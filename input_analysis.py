import numpy as np
import torch

inputs: torch.Tensor = torch.load("test.pt")

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
#    data[:, :, 2] = normalize_slice(data[:, :, 2])
    return data

#inputs = inputs[1032117:1032309, :, :]
#print(f"Data nan count: {np.sum(np.isnan(inputs))}")
#normalized_data = normalize_data(torch.tensor(inputs))
#print(f"Normalized data nan count: {torch.sum(torch.isnan(normalized_data))}")
#exit()
print(inputs.shape)
#print average number of each feature
print(torch.mean(inputs[:, :, 0]))
print(torch.mean(inputs[:, :, 1]))

print(inputs[0, :, 0])
print(inputs[0, :, 1])

#inputs = torch.tensor(inputs*10, dtype=torch.bfloat16)

#print(inputs[0, :, 0])
#print(inputs[0, :, 1])


normalized = normalize_data(inputs)
print(torch.mean(normalized[:, :, 0]))
print(torch.mean(normalized[:, :, 1]))
