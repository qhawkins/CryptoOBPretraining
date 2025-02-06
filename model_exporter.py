import torch
import numpy as np
from fp8_models import DeepNarrowTransformerModelPT, PPOModel
import transformer_engine.pytorch as te

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

def load_model(path: str, dropout: float, shapes: tuple, state_features: int) -> PPOModel:
    ob_model = DeepNarrowTransformerModelPT(shapes, shapes, dropout)
    #state_dict = torch.load(path)  # Addressing FutureWarning
    #state_dict = state_dict['model_state_dict']
    #state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    #state_dict = parse_state_dict(state_dict)
    #ob_model.load_state_dict(state_dict)
    ppo_model = PPOModel(shapes, shapes, dropout, state_features, ob_model)
    # ppo_model.eval()  # Typically used for evaluation mode
    return ppo_model

if __name__ == "__main__":
    temporal_dim = 1536
    shapes = (temporal_dim, 96, 2)
    dropout = 0.0
    state_features = 16
    model: PPOModel = load_model("/media/qhawkins/SSD3/single_models/pretrained_ddp_val_loss_000116003_epoch_5_mse_deep_narrow_transformer.pth", dropout, shapes, state_features)
    ob_example = torch.rand(32, temporal_dim, 96, 2).to("cuda")
    state_example = torch.rand(32, 16).to("cuda")

    model = model.to("cuda")

    # Optionally, avoid using graphed callables if they introduce scripting issues
    # model = te.make_graphed_callables(model, (ob_example, state_example))
    print("Model graphed")
    model = model.train(True)
    print("Model set to training mode")

    # Use tracing instead of scripting
    traced_script_module = torch.jit.trace(model, (ob_example, state_example))

    output: torch.Tensor = traced_script_module(ob_example, state_example)
    print(output[0].shape)
    print(output[1].shape)

    # Save the traced model
    torch.jit.save(traced_script_module, "ppo_model.pt")
    print("Model saved")

