from models import SmallFCModel, MediumFCModel, LargeFCModel, TinyLSTMModel, DeepLSTMModel, ShallowLSTMModel, TinyTransformerModel
from training_classes import normalize_data, PretrainingDataset
import torch
import numpy as np

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

def load_model(path: str):
    model = TinyTransformerModel((128, 96, 2), (128, 96, 2), 0.25)
    state_dict = torch.load(path)
    state_dict = state_dict['model_state_dict']
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model

if __name__ == "__main__":
    len_dataset = np.load("/home/qhawkins/Desktop/CryptoOBDataExploration/test_dataset.npy", mmap_mode='r').shape[0]
    model = TinyTransformerModel((128, 96, 2), (128, 96, 2), 0.25)
    state_dict = torch.load("/home/qhawkins/Downloads/pretrained_ddp_val_loss_004381035_epoch_44_mse_tiny_transformer.pth")
    state_dict = state_dict['model_state_dict']
    print(state_dict.keys())
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    print("Model loaded")
    dataset = PretrainingDataset("/home/qhawkins/Desktop/CryptoOBDataExploration/test_dataset.npy", 0, len_dataset, 128, 96)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=8)
    loss_fn = torch.nn.MSELoss().cuda()
    model.compile()


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