import numpy as np
import torch

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
    #data[:, :, 2] = normalize_slice(data[:, :, 2])
    data = data[:, :, :-1]

    return data

data = np.load("/home/qhawkins/Desktop/CryptoOBDataExploration/train_dataset.npy")
#print(f"Data nan count: {np.sum(np.isnan(data))}")
data_torch = torch.from_numpy(data)
print(f"Data torch nan count: {torch.sum(torch.isnan(data_torch))}")
data_torch = data_torch.float()

print(data.shape)



normalized_data = normalize_data(data_torch)
print(normalized_data[0, :, :])
print(100*"-")
print(normalized_data[-1, :, :])

print(f"Normalized data nan count: {torch.sum(torch.isnan(normalized_data))}")