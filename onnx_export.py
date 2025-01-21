import torch
from fp8_models import DeepNarrowTransformerModelPT, PPOModel
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from onnxruntime.training import artifacts
import onnx
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16)

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
    #print(f"OB model keys: {ob_model.state_dict().keys()}")
    #print(f"State dict keys (transformerengine model): {state_dict.keys()}")
    #exit()
    state_dict = parse_state_dict(state_dict)
    
    ob_model.load_state_dict(state_dict)
    ppo_model = PPOModel((256, 96, 2), (256, 96, 2), 0.25, 16, ob_model)
    # ppo_model.eval()  # Typically used for evaluation mode
    return ppo_model

if __name__ == "__main__":
    model: PPOModel = load_model("/media/qhawkins/SSD3/single_models/pretrained_ddp_val_loss_000116003_epoch_5_mse_deep_narrow_transformer.pth")
    grad_param_list = []
    frozen_params = []
    for param in model.named_parameters():
        if param[1].requires_grad:
            name = param[0]
            print(param[1].dtype)
            if name.startswith("ob_encoder"):
                frozen_params.append(name)
            else:
                grad_param_list.append(name)
    
    model = model.to("cuda")

    # Optionally, avoid using graphed callables if they introduce scripting issues
    # model = te.make_graphed_callables(model, (ob_example, state_example))
    print("Model graphed")
    #model = model.train(True)
    print("Model set to training mode")
    with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        for i in range(128):
            ob_example = torch.rand(1, 256, 96, 2).to("cuda")
            state_example = torch.rand(1, 256, 16).to("cuda")
            x = model(ob_example, state_example)  # Warm-up pass
            print(f"{i}, {x}")
    print("Model warmed up")
    with torch.inference_mode(), te.fp8_autocast(
            enabled=True, fp8_recipe=recipe
        ):
        #with te.onnx_export(enabled=True):
        ob_example = torch.rand(1, 256, 96, 2).to("cuda")
        state_example = torch.rand(1, 256, 16).to("cuda")
        #model_jit = torch.jit.trace(model, (ob_example, state_example))
        model_jit = torch.jit.script(model)
        print("Model traced")
            #model_jit = model_jit.to("cpu")
    torch.jit.save(model_jit, "ppo_model.pt")
    #torch.save(model_jit, "ppo_model.pt")
    print("saved successfully!")
    exit()
    #model = torch.onnx.export(model, (ob_example, state_example, ), "ppo_model.onnx", export_params=True, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)
    model_path = "ppo_model.onnx"
    base_model = onnx.load(model_path)

    requires_grad = grad_param_list

    # Generate the training artifacts
    artifacts.generate_artifacts(base_model, requires_grad = requires_grad, frozen_params = frozen_params,
                                loss = artifacts.LossType.MSELoss, optimizer = artifacts.OptimType.AdamW)