import torch
import numpy as np
from models import SmallFCModel, MediumFCModel, LargeFCModel, TinyLSTMModel, DeepLSTMModel, ShallowLSTMModel

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    fake_data = np.random.rand(4096, 128, 128, 2)
    #small = SmallFCModel((128, 2, 128), (128, 2, 128), 0.25)
    #medium = MediumFCModel((128, 2, 128), (128, 2, 128), 0.25)
    #large = LargeFCModel((128, 2, 128), (128, 2, 128), 0.25)
    tiny_lstm = TinyLSTMModel((128, 128, 2), (128, 128, 2), 0.25)
    small_lstm = ShallowLSTMModel((128, 128, 2), (128, 128, 2), 0.25)
    deep_lstm = DeepLSTMModel((128, 128, 2), (128, 128, 2), 0.25)
    #print(f"Small model has {count_parameters(small)/1e6} M parameters")
    #print(f"Medium model has {count_parameters(medium)/1e6} M parameters")
    #print(f"Large model has {count_parameters(large)/1e6} M parameters")
    print(f"Tiny LSTM model has {count_parameters(tiny_lstm)/1e6} M parameters")
    print(f"Shallow LSTM model has {count_parameters(small_lstm)/1e6} M parameters")
    print(f"Deep LSTM model has {count_parameters(deep_lstm)/1e6} M parameters")
