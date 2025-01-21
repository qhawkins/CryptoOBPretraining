from fp8_models import DeepNarrowTransformerModelPT
from training_classes import PretrainingDataset
import torch

def apply_mask(inputs: torch.Tensor, mask_percentage=0.15, mask_value=0.0, device='cuda'):
    """
    Applies masking to the input tensor.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, features).
        mask_percentage (float): Fraction of entries to mask.
        mask_value (float): Value to replace masked entries with.
        device (str): Device to perform masking on.

    Returns:
        masked_inputs (torch.Tensor): Tensor with masked entries.
        labels (torch.Tensor): Tensor with original values for masked entries and -100 elsewhere.
        mask (torch.Tensor): Boolean mask indicating which entries were masked.
    """
    # Generate a mask for 15% of the entries
    mask = torch.rand(inputs.shape, device=device) < mask_percentage

    # Replace masked entries in inputs with mask_value
    masked_inputs = inputs.clone()
    masked_inputs[mask] = mask_value

    return masked_inputs.cuda(), mask.cuda()

def parse_state_dict(state_dict: dict) -> dict:
    """
    This function will perform key replacements from the transformer-engine
    style naming convention to the vanilla PyTorch naming convention found in
    the OB model. 
    
    Modify this carefully to handle:
    1) ._extra_state   => skip these keys (they are TE metadata).
    2) fc1_weight      => linear1.weight
    3) fc1_bias        => linear1.bias
    4) fc2_weight      => linear2.weight
    5) fc2_bias        => linear2.bias
    6) The QKV-projection in TE => in_proj_{weight,bias} and out_proj_{weight,bias} in PyTorch.
    7) The layernorm in TE => the appropriate norm{1,2}.weight/bias in the OB model.
    
    If any keys obviously do not map (like "core_attention._extra_state"), we skip them
    and/or raise an Exception if they look essential but can’t be matched.

    """
    output_state_dict = {}

    for original_key in state_dict.keys():
        # -------------
        # 1) skip TE internal metadata states
        # -------------
        if "._extra_state" in original_key:
            continue

        new_key = original_key  # We will progressively modify new_key

        # -------------
        # 2) fc1/2 => linear1/2 for MLP
        # -------------
        if "fc1_weight" in new_key:
            new_key = new_key.replace("fc1_weight", "linear1.weight")
        elif "fc1_bias" in new_key:
            new_key = new_key.replace("fc1_bias", "linear1.bias")
        elif "fc2_weight" in new_key:
            new_key = new_key.replace("fc2_weight", "linear2.weight")
        elif "fc2_bias" in new_key:
            new_key = new_key.replace("fc2_bias", "linear2.bias")

        # -------------
        # 3) TE uses "layernorm_qkv" for the layernorm done before QKV-projection,
        #    plus "proj" for the final out-projection. 
        #
        #    The OB model typically does multi-head self-attention as follows:
        #       self_attn.in_proj_weight, self_attn.in_proj_bias
        #       self_attn.out_proj.weight, self_attn.out_proj.bias
        #
        #    TE separates the QKV projection from the "layernorm" part, 
        #    so we need to handle both:
        #
        #    a) QKV projection => in_proj_weight, in_proj_bias
        #    b) final projection => out_proj.weight, out_proj.bias
        #
        #    c) The "layernorm_qkv" in TE => norm1 or norm2 in the OB model 
        #       depending on the design. Typically, the norm that precedes self-attn 
        #       is norm1 in a standard "encoderX" block.  
        # -------------
        if "self_attention.layernorm_qkv.layer_norm_weight" in new_key:
            # This typically is norm1.weight in many PyTorch Transformers
            new_key = new_key.replace("self_attention.layernorm_qkv.layer_norm_weight", "norm1.weight")
        elif "self_attention.layernorm_qkv.layer_norm_bias" in new_key:
            new_key = new_key.replace("self_attention.layernorm_qkv.layer_norm_bias", "norm1.bias")

        # QKV-projection weight in TE is "layernorm_qkv.weight"
        #    shape is usually [3 * hidden_size, hidden_size]
        # => must map to self_attn.in_proj_weight
        elif "self_attention.layernorm_qkv.weight" in new_key:
            new_key = new_key.replace("self_attention.layernorm_qkv.weight", "self_attn.in_proj_weight")
        
        # QKV-projection bias => self_attn.in_proj_bias
        elif "self_attention.layernorm_qkv.bias" in new_key:
            new_key = new_key.replace("self_attention.layernorm_qkv.bias", "self_attn.in_proj_bias")

        # -------------
        # TE final MHA projection => "proj.weight"/"proj.bias" => out_proj
        # -------------
        elif "self_attention.proj.weight" in new_key:
            new_key = new_key.replace("self_attention.proj.weight", "self_attn.out_proj.weight")
        elif "self_attention.proj.bias" in new_key:
            new_key = new_key.replace("self_attention.proj.bias", "self_attn.out_proj.bias")

        # -------------
        # 4) For the MLP portion, "layernorm_mlp" is typically the second LN in a standard 
        #    transformer block => norm2
        # -------------
        elif "layernorm_mlp.layer_norm_weight" in new_key:
            new_key = new_key.replace("layernorm_mlp.layer_norm_weight", "norm2.weight")
        elif "layernorm_mlp.layer_norm_bias" in new_key:
            new_key = new_key.replace("layernorm_mlp.layer_norm_bias", "norm2.bias")

        elif "layernorm_mlp" in new_key:
            new_key = new_key.replace("layernorm_mlp.", "")

        # -------------
        # Done rewriting the key. Now store it
        # -------------
        output_state_dict[new_key] = state_dict[original_key]

    return output_state_dict

def load_model(path: str):
    ob_model = DeepNarrowTransformerModelPT((256, 96, 2), (256, 96, 2), 0.25)
    state_dict = torch.load(path)  # Addressing FutureWarning
    state_dict = state_dict['model_state_dict']
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    means1 = []
    means2 = []
    for key, v in state_dict.items():
        try:
            means1.append(torch.mean(v))
            print(torch.mean(v))
        except:
            continue
    #print(f"OB model keys: {ob_model.state_dict().keys()}")
    #print(f"State dict keys (transformerengine model): {state_dict.keys()}")
    #exit()
    print(100*"=")
    state_dict = parse_state_dict(state_dict)
    for _, v in state_dict.items():
        means2.append(torch.mean(v))
        print(torch.mean(v))
    for i in range(len(means1)):
        print(f"Mean difference: {means1[i] - means2[i]}")
    ob_model.load_state_dict(state_dict)
    ob_model.to("cuda")
    ob_model = ob_model.eval()
    return ob_model

if __name__ == "__main__":
    #len_dataset = np.load("/home/qhawkins/Desktop/CryptoOBPretraining/test_indices.npy", mmap_mode='r').shape[0]
    model = load_model("/media/qhawkins/SSD3/single_models/pretrained_ddp_val_loss_000116003_epoch_5_mse_deep_narrow_transformer.pth")
    print("Model loaded")

    shared_test_dataset = (
			"/home/qhawkins/Desktop/CryptoOBPretraining/eth_btc_test_indices.npy",
			"/home/qhawkins/Desktop/CryptoOBPretraining/btc_usdt_test_indices.npy"
			)
		

    dataset = PretrainingDataset(shared_test_dataset, (0, 0), (2048*32, 2048*32), 256, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=8)
    loss_fn = torch.nn.MSELoss().cuda()
    #model.compile()


    loss_values = []
    for idx, data in enumerate(dataloader):
        masked_inputs, mask = apply_mask(
            data,
            mask_percentage=.25,
            mask_value=0,  # You can choose a different mask value if needed
            device='cuda'
        )
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                data = data.cuda()
                outputs = model(masked_inputs)  # Shape: (batch_size, seq_length -1, features)
                loss_val = loss_fn(outputs[mask], data[mask])
                print(f"Data masked: {data[mask]}, outputs masked: {outputs[mask]}")
                #exit()
            mean_loss = torch.mean(loss_val).cpu()
            print(f"Mean loss: {mean_loss} for epoch {idx}")
            loss_values.append(mean_loss)
            
                #exit()

    print(f"Mean loss: {torch.mean(torch.tensor(loss_values))}")