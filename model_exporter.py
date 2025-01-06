import torch
from models import TinyTransformerModel

def load_model(path: str):
    model = TinyTransformerModel((128, 96, 2), (128, 96, 2), 0.25)
    state_dict = torch.load(path)
    state_dict = state_dict['model_state_dict']
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model



if __name__ == "__main__":
    model = load_model("/home/qhawkins/Downloads/pretrained_ddp_val_loss_000055124_epoch_7_mse_tiny_transformer.pth")
    example = torch.rand(128, 96, 2)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("traced_transformer_model.pt")
    print("Model saved")