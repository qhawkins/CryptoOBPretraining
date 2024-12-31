import torch
import numpy as np
from models import SmallFCModel, MediumFCModel, LargeFCModel, SmallLSTMModel

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    fake_data = np.random.rand(4096, 128, 128, 2)
    #small = SmallFCModel((128, 2, 128), (128, 2, 128), 0.25)
    #medium = MediumFCModel((128, 2, 128), (128, 2, 128), 0.25)
    #large = LargeFCModel((128, 2, 128), (128, 2, 128), 0.25)
    small_lstm = SmallLSTMModel((128, 128, 2), (128, 128, 2), 0.25)
    #print(f"Small model has {count_parameters(small)/1e6} M parameters")
    #print(f"Medium model has {count_parameters(medium)/1e6} M parameters")
    #print(f"Large model has {count_parameters(large)/1e6} M parameters")
    print(f"Small LSTM model has {count_parameters(small_lstm)/1e6} M parameters")
    lstm_test = small_lstm(torch.tensor(fake_data, dtype=torch.float32))
    print(lstm_test.shape)