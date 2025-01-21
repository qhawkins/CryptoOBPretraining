import torch
from fp8_models import DeepNarrowTransformerModel, PPOModel
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from onnxruntime.training import artifacts
import onnx

fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
recipe = DelayedScaling(fp8_format=fp8_format)
def load_model(path: str):
    ob_model = DeepNarrowTransformerModel((256, 96, 2), (256, 96, 2), 0.25)
    state_dict = torch.load(path)  # Addressing FutureWarning
    state_dict = state_dict['model_state_dict']
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    ob_model.load_state_dict(state_dict)
    ppo_model = PPOModel((256, 96, 2), (256, 96, 2), 0.25, 16, ob_model)
    # ppo_model.eval()  # Typically used for evaluation mode
    return ppo_model

if __name__ == "__main__":
    model: PPOModel = load_model("/media/qhawkins/SSD3/single_models/pretrained_ddp_val_loss_000116003_epoch_5_mse_deep_narrow_transformer.pth")
    ob_example = torch.rand(1, 256, 96, 2).to("cuda")
    state_example = torch.rand(1, 256, 16).to("cuda")
    grad_param_list = []
    frozen_params = []
    for param in model.named_parameters():
        if param[1].requires_grad:
            name = param[0]
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
    with torch.inference_mode(), te.fp8_autocast(
            enabled=True, fp8_recipe=recipe
        ):
        with te.onnx_export(enabled=True):
            model = torch.onnx.export(model, (ob_example, state_example, ), "ppo_model.onnx", export_params=True, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)

    model_path = "ppo_model.onnx"
    base_model = onnx.load(model_path)

    requires_grad = grad_param_list

    # Generate the training artifacts
    artifacts.generate_artifacts(base_model, requires_grad = requires_grad, frozen_params = frozen_params,
                                loss = artifacts.LossType.MSELoss, optimizer = artifacts.OptimType.AdamW)