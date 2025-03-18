import numpy as np

#code to create a small slice of the full raw dataset for testing purposes
if __name__ == "__main__":
    #pair = "BTC_USDT"
    #pair = "ETH_BTC"
    pair = "XRP_BTC"
    path = f"/media/qhawkins/SSD3/{pair}_20240101_20241201.npy"
    save_path = f"./training_data/raw_sliced/{pair}_20240101_20241201_sliced.npy"
    data = np.load(path, mmap_mode="r")
    #data = data.astype(np.float32)
    print(data.shape)
    print(f"Data zeros count: {np.sum(data[:, 3] == 0)}")
    print(f"Data nans count: {np.sum(np.isnan(data[:, 3]))}")
    print(data[0, :])
    exit()
    
    np.save(save_path, data)

    #np.save(save_path, data[:32768])
