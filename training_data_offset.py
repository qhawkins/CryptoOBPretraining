import numpy as np

data = np.load("/home/qhawkins/Desktop/CryptoOBPretraining/full_parsed.npy", mmap_mode='r+')
print(data.shape)
exit()
data = data[1280000, :, :]

np.save("/home/azureuser/data/full_parsed.npy", data)