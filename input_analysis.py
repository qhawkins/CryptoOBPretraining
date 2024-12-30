import numpy as np
import torch

inputs: np.dtype = np.load("test.npy")

def normalize_slice(data: torch.Tensor):
    mean = torch.mean(data)
    std = torch.std(data)
    return (data - mean) / std


def normalize_data(data: torch.Tensor):
    """
    Normalizes the input data to have zero mean and unit variance.
    
    :param data: A torch.Tensor of shape (time, depth, features)
    :return: A torch.Tensor of shape (time, depth, features) with zero mean and unit variance
    """

    data[:, :, 0] = normalize_slice(data[:, :, 0])
    data[:, :, 1] = normalize_slice(data[:, :, 1])
    data[:, :, 2] = normalize_slice(data[:, :, 2])

    return data

print(inputs[:, :, :])
print(inputs.shape)
#print average number of each feature
print(np.mean(inputs[:, :, 0]))
print(np.mean(inputs[:, :, 1]))
print(np.mean(inputs[:, :, 2]))
normalized = normalize_data(torch.tensor(inputs))
print(np.mean(normalized[:, :, 0]))
print(np.mean(normalized[:, :, 1]))
print(np.mean(normalized[:, :, 2]))
