import pandas as pd
import numpy as np
import torch
import os
import glob
import pandas as pd
from copy import deepcopy
import numba

@numba.njit(cache=True)
def find_price_index(prices, price):
    """
    Linear search to find the index of a given price.
    Returns the index if found, else -1.
    """
    for i in range(len(prices)):
        if prices[i] == price:
            return i
    return -1

@numba.njit(cache=True)
def sort_levels(levels: np.array)-> np.array:
    return levels[np.argsort(levels[:, 0])]

def map_order_type(update_type: str) -> np.int64:
    if update_type == 'ADD':
        return 0

    elif update_type == 'SUB':
        return 1

    elif update_type == 'MATCH':
        return 2

    elif update_type == 'SET':
        return 3

    elif update_type == 'DELETE':
        return 4

    elif update_type == 'SNAPSHOT':
        return 5
    else:
        raise ValueError(f"Unknown update type: {update_type}")

@numba.njit(cache=True)
def order_book_update(levels: np.array, update_type: float, price_idx: float, price: float,  size: float):
    if update_type == 5.0:
        levels[price_idx, 0] = price
        levels[price_idx, 1] = size
    elif update_type == 0.0:
        levels[price_idx, 1] += size
        #print(f"Adding size {size} to price {levels[price_idx, 0]}")
    elif update_type == 1.0 or update_type == 2.0:
        levels[price_idx, 1] -= size
    elif update_type == 3.0:
        levels[price_idx, 0] = price
        levels[price_idx, 1] = size
    elif update_type == 4.0:
        levels[price_idx, 0] = 0
        levels[price_idx, 1] = 0
    else:
        levels[price_idx, 0] = price
        levels[price_idx, 1] = size
    return levels

@numba.njit(cache=True)
def add_slice(levels: np.array, price: float, size: float):
    idx = 0
    storage = np.zeros((len(levels), 2), dtype=np.float32)
    for i in range(len(levels)-1):
        if price > levels[i, 0] and price < levels[i+1, 0]:
            idx = i
    if price < levels[0, 0]:
        storage[0, 0] = price
        storage[0, 1] = size
        storage[1:, :] = levels[:-1, :]
        return storage
    elif idx == 0:
        levels[idx, 0] = price
        levels[idx, 1] = size
        return levels
    else:
        storage[:idx, :] = levels[:idx, :]
        storage[idx, 0] = price
        storage[idx, 1] = size
        storage[idx+1:, :] = levels[idx:-1, :]
        return storage

@numba.njit(cache=True)
def switch_padding(levels: np.array):
    first_non_zero = -1
    for i in range(len(levels)):
        if levels[i, 0] != 0:
            first_non_zero = i

    if first_non_zero == 0:
        #print("First non-zero is 0, no need to switch padding")
        return levels
    idx_2 = 0
    levels_copy = np.zeros_like(levels)
    for i in range(first_non_zero, len(levels)):
        if levels[i, 0] != 0:
            levels_copy[idx_2, 0] = levels[i, 0]
            levels_copy[idx_2, 1] = levels[i, 1]
            idx_2 += 1
    for i in range(idx_2, len(levels)):
        levels_copy[i, 0] = 0
        levels_copy[i, 1] = 0
    return levels_copy


@numba.njit(cache=True)
def optimized_order_book(arr: np.array, snapshots: np.array, max_size: int = 128, subsample: int = 10):
    flag = False
    max_size = max_size + 16
    buys = np.zeros((max_size//2, 2), dtype=np.float32)
    sells = np.zeros((max_size//2, 2), dtype=np.float32)
    consolidated = np.zeros((max_size, 2), dtype=np.float32)
    
    for idx, row in enumerate(arr):
        if idx % 10000000 == 0:
            print(f"Processing row {idx//10}, number of 0s in snapshots: {np.sum(snapshots[(idx//10)-1000000:(idx//10)] == 0)}")        
        price = row[2]
        size = row[3]
        update_type = row[0]
        is_buy = row[1]
        if is_buy == 1.0:
            price_idx = find_price_index(buys[:, 0], price)
            if price_idx != -1:
                buys = order_book_update(buys, update_type, price_idx, price, size)
            else:
                buys = add_slice(buys, price, size)        
                buys = sort_levels(buys)


        elif is_buy == 0.0:
            price_idx = find_price_index(sells[:, 0], price)
            if price_idx != -1:
                sells = order_book_update(sells, update_type, price_idx, price, size)
            else:
                sells = add_slice(sells, price, size)
                sells = sort_levels(sells)
            
        consolidated[:max_size//2, :] = buys
        consolidated[max_size//2:, :] = sells

        #consolidated = np.concatenate((buys, sells), axis=0)
        if flag is False:
            start_idx = idx
            flag = True
        if (idx+1) % subsample == 0:
            snapshots[idx//subsample] = consolidated[8:-8, :]

    return snapshots, start_idx
            
if __name__ == "__main__":
    depth = 96
    subsample = 10
    azure = False
    if azure:
        raw_data = pd.read_csv("/home/azureuser/data/eth_btc_20231201_20241201.csv", engine="pyarrow", low_memory=True)
    else:
        raw_data = np.load("/media/qhawkins/SSD3/btc_usdt_20231201_20241201.npy", mmap_mode='r')
    print("loaded")
    print(raw_data.shape)
    print(raw_data[0, :])
    print(raw_data[-1, :])
    #exit()
    results = np.zeros((len(raw_data)//subsample, depth, 2), dtype=np.float32, order="C")
    ob_state, start_idx = optimized_order_book(raw_data, results, depth, subsample=subsample)

    #drop any rows with 0s
    print(f"Ob_state shape: {ob_state.shape}")
    #ob_state = ob_state[~np.any(ob_state == 0, axis=(1, 2))]
    #print("Sliced")
    #print(ob_state[-1, :, 0])
    #ob_state = ob_state/1e7
    #ob_state = ob_state[:, :, :-1]
    #ob_state_bf16 = torch.tensor(ob_state, dtype=torch.bfloat16, requires_grad=False)
    #print(f"Ob_state shape: {ob_state.shape}")
    if azure:
        np.save("/home/azureuser/datadrive/full_parsed.npy", ob_state)
    
    else:
        np.save("/media/qhawkins/SSD3/btc_usdt_full_parsed.npy", ob_state)
    
    
    #ob_state = torch.tensor(ob_state, dtype=torch.float32, requires_grad=False)
    #print(f"bf16 {ob_state_bf16[-1, :, :]}")
    for idx, entry in enumerate(ob_state[-1, :, :]):
        print(f"Slice {idx}, price: {entry[0]}, size: {entry[1]}")
    
    #torch.save(ob_state, "full_parsed.pt")

    #np.save("test.npy", ob_state)

    #sorted_levels = sorted(ob_state[-1].keys())
    #print(ob_state[-1])