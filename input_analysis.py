import numpy as np
import torch


data = np.load("/home/qhawkins/Desktop/CryptoOBDataExploration/test_dataset.npy", mmap_mode='r')[:16384, :, 0]
print(f"Data zeros count: {np.sum(data == 0)}")