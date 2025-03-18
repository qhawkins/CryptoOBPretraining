import numpy as np
import torch


data = np.load("/media/qhawkins/SSD3/training_data/ETH_BTC_full_parsed.npy", mmap_mode='r')[1160:1416, :, 0]
print(f"Data zeros count: {np.sum(data == 0)}")