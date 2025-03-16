import numpy as np

#code to create a small slice of the full dataset for testing purposes
if __name__ == "__main__":
    #pair = "BTC_USDT"
    pair = "ETH_BTC"
    
    path = f"/media/qhawkins/SSD3/{pair}_20240101_20241201.npy"
    save_path = f"/media/qhawkins/SSD3/{pair}_20240101_20241201_sliced.npy"
    data = np.load(path, mmap_mode="r")
    np.save(save_path, data[:4096])