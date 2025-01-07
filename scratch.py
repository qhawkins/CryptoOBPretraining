import numpy as np


a = np.load("/home/azureuser/data/train_dataset.npy", mmap_mode='r')
print(np.sum(np.isnan(a)))
print(a.shape)
print(f"num zeros in a: {np.sum(a == 0)}")