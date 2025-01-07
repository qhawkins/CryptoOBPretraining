import numpy as np
import torch

def z_score(data: torch.Tensor):
    mean = torch.mean(data)
    std = torch.std(data)
    return (data - mean) / std

def min_max_normalize(data: torch.Tensor):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val)

def volume_normalize(data: torch.Tensor):
    #divide all volume values by the total volume in the slice and then min-max normalize the data
    cumulative_volume = torch.sum(data)
    data = data / cumulative_volume
    return min_max_normalize(data)

def price_normalize(data: torch.Tensor):
    #divide all price values by the mid price in the slice and then min-max normalize the data
    beginning = data[0]
    print(beginning.shape)
    data = torch.div(data, beginning)
    return min_max_normalize(data)

def normalize_data(data: torch.Tensor):
    """
    Normalizes the input data to have zero mean and unit variance.
    
    :param data: A torch.Tensor of shape (time, depth, features)
    :return: A torch.Tensor of shape (time, depth, features) with zero mean and unit variance
    """
    data[:, :, 0] = price_normalize(data[:, :, 0])
    data[:, :, 1] = volume_normalize(data[:, :, 1])
    #data[:, :, 2] = normalize_slice(data[:, :, 2])

    return data

data = np.load("/home/qhawkins/Desktop/CryptoOBDataExploration/test_dataset.npy", mmap_mode='r')[:16384]
#print(f"Data nan count: {np.sum(np.isnan(data))}")
data_torch = torch.from_numpy(data.copy())
print(f"Data torch zeros count: {torch.sum(torch.sum(data_torch==0))}")
print(f"Data torch nan count: {torch.sum(torch.isnan(data_torch))}")
data_torch = data_torch.float()

print(data.shape)

print(f"data zeros count: {np.sum(data == 0)}")
exit()

normalized_data = normalize_data(data_torch)

print(f"Normalized data feature 0 mean: {torch.mean(normalized_data[:, :, 0])}")
print(f"Normalized data feature 0 std: {torch.std(normalized_data[:, :, 0])}")
print(f"Normalized data feature 1 mean: {torch.mean(normalized_data[:, :, 1])}")
print(f"Normalized data feature 1 std: {torch.std(normalized_data[:, :, 1])}")
print(f"Normalized data feature 0 min: {torch.min(normalized_data[:, :, 0])}")
print(f"Normalized data feature 0 max: {torch.max(normalized_data[:, :, 0])}")
print(f"Normalized data feature 1 min: {torch.min(normalized_data[:, :, 1])}")
print(f"Normalized data feature 1 max: {torch.max(normalized_data[:, :, 1])}")


print(f"Normalized data nan count: {torch.sum(torch.isnan(normalized_data))}")
exit()

for dp in data_torch:
    normalized_data = normalize_data(dp)
    print(normalized_data[0, :, :])
    print(100*"-")
    print(normalized_data[-1, :, :])
    exit()

