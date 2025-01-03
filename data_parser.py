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
def find_min_price_index(prices):
    """
    Find the index of the minimum price.
    """
    min_idx = 0
    for i in range(len(prices)):
        if prices[i] < prices[min_idx]:
            min_idx = i
    return min_idx

@numba.njit(cache=True)
def find_max_price_index(prices):
    """
    Find the index of the maximum price.
    """
    max_idx = 0
    for i in range(len(prices)):
        if prices[i] > prices[max_idx]:
            max_idx = i
    return max_idx

@numba.njit(cache=True)
def sort_levels(levels: np.array)-> np.array:
    return levels[np.argsort(levels[:, 0])]


@numba.njit(cache=True)
def optimized_order_book(arr: np.array, snapshots: np.array, max_size: int = 128):
    flag = False
    is_buy_sum = 0
    is_sell_sum = 0
    #levels is price, size, is_buy
    levels = np.zeros((max_size, 3), dtype=np.float32)
    n_levels = 0

    #find_max_price_index
    #find_min_price_index
    #find_price_index

    for idx, row in enumerate(arr):
        if idx % 1000000 == 0:
            print(f"Processing row {idx}, length of levels: {len(levels)}")
        price = row[2]
        size = row[3]
        update_type = row[0]
        is_buy = row[1]
        price_in_levels = find_price_index(levels[:, 0], price)
        if price_in_levels != -1:
            if update_type == 5.0:
                levels[price_in_levels][0] = price
                levels[price_in_levels][1] = size
                levels[price_in_levels][2] = is_buy
            elif update_type == 0.0:
                levels[price_in_levels][1] += size
            elif update_type == 1.0:
                levels[price_in_levels][1] -= size
            elif update_type == 2.0:
                levels[price_in_levels][1] -= size
            elif update_type == 3.0:
                levels[price_in_levels][0] = price
                levels[price_in_levels][1] = size
                levels[price_in_levels][2] = is_buy
            elif update_type == 4.0:
                levels[price_in_levels][0] = 0
                levels[price_in_levels][1] = 0
                levels[price_in_levels][2] = 0
            else:
                levels[price_in_levels][0] = price
                levels[price_in_levels][1] = size
                levels[price_in_levels][2] = is_buy
            levels = sort_levels(levels)
        else:
            levels = sort_levels(levels)
            num_buys = np.sum(levels[:, 2])
            num_sells = max_size - num_buys
            if n_levels < max_size:
                levels[0] = [price, size, is_buy]
                n_levels += 1
            else:
                if num_buys > num_sells:
                    min_idx = find_min_price_index(levels[:, 0])
                    levels[min_idx] = [price, size, is_buy]

                elif num_sells > num_buys:
                    max_idx = find_max_price_index(levels[:, 0])
                    levels[max_idx] = [price, size, is_buy]
                else:
                    #find mid price
                    mid_price = (levels[max_size//2][0] + levels[max_size//2 + 1][0])/2
                    max_distance_from_mid = levels[find_max_price_index(levels[:, 0])][0] - mid_price
                    min_distance_from_mid = mid_price - levels[find_min_price_index(levels[:, 0])][0]
                    if max_distance_from_mid > min_distance_from_mid:
                        max_idx = find_max_price_index(levels[:, 0])
                        levels[max_idx] = [price, size, is_buy]
                    else:
                        min_idx = find_min_price_index(levels[:, 0])
                        levels[min_idx] = [price, size, is_buy]

                    
        if n_levels == max_size:
            if flag is False:
                start_idx = idx
                flag = True
            snapshots[idx] = levels[:, :2]
    return snapshots, start_idx
            

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

if __name__ == "__main__":
    depth = 128
    raw_data = pd.read_csv("/home/qhawkins/Desktop/eth_btc_20231201_20241201.csv", engine="pyarrow")
    raw_data.dropna(axis=0, inplace=True)
    #print(raw_data.value_counts("update_type"))
    #exit()
    #the start of the first order book snapshot is the first row of the data that doesnt have an update type of snapshot
    #starting_ob_state = raw_data.loc[raw_data["update_type"] != "snapshot"].index[0]
    #raw_data['time_exchange_int'] = raw_data['time_exchange'].apply(lambda x: int(x.timestamp()*1000))
    #raw_data['time_coinapi_int'] = raw_data['time_coinapi'].apply(lambda x: int(x.timestamp()*1000))

    #raw_data.set_index("time_coinapi_int", inplace=True)

    raw_data.drop(columns=["time_exchange", "time_coinapi"], inplace=True)

    print(raw_data.describe())

    #raw_data.rename(columns={"time_exchange_int": "time_exchange", "time_coinapi_int": "time_coinapi"}, inplace=True)

    raw_data['update_type'] = raw_data['update_type'].apply(map_order_type)
    raw_data['is_buy'] = raw_data['is_buy'].apply(lambda x: 1 if x else 0)
    raw_data['entry_px'] = (raw_data['entry_px']).astype(np.float32)
    raw_data['entry_sx'] = (raw_data['entry_sx']).astype(np.float32)


    raw_data = raw_data.to_numpy(dtype=np.float32)
    #raw_data = raw_data[:10000, :]
    #raw_data = raw_data[:, :4]
    print(raw_data.shape)
    print(raw_data[0, :])
    #exit()
    results = np.zeros((len(raw_data), depth, 2), dtype=np.float32, order="C")
    ob_state, start_idx = optimized_order_book(raw_data, results, depth)
    ob_state = ob_state[start_idx+1024:]
    #ob_state = ob_state/1e7
    #ob_state = ob_state[:, :, :-1]
    #ob_state_bf16 = torch.tensor(ob_state, dtype=torch.bfloat16, requires_grad=False)
    
    np.save("full_parsed.npy", ob_state)
    #ob_state = torch.tensor(ob_state, dtype=torch.float32, requires_grad=False)
    #print(f"bf16 {ob_state_bf16[-1, :, :]}")
    print(f"fp32 {ob_state[-1, :, :]}")
    print(f"fp32 {ob_state[0, :, :]}")
    #torch.save(ob_state, "full_parsed.pt")

    #np.save("test.npy", ob_state)

    #sorted_levels = sorted(ob_state[-1].keys())
    #print(ob_state[-1])