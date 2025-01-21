import torch
from fp8_models import DeepNarrowTransformerModel, PPOModel
import transformer_engine.pytorch as te

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

    model = model.to("cuda")

    # Optionally, avoid using graphed callables if they introduce scripting issues
    # model = te.make_graphed_callables(model, (ob_example, state_example))
    print("Model graphed")
    model = model.train(True)
    print("Model set to training mode")

    # Use tracing instead of scripting
    traced_script_module = torch.jit.trace(model, (ob_example, state_example))

    output: torch.Tensor = traced_script_module(ob_example, state_example)
    print(output.shape)

    # Save the traced model
    traced_script_module.save("traced_transformer_model.pt")
    print("Model saved")

