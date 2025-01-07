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
def find_last_is_buy(levels: np.array):
    """
    Find the index of the last is_buy level.
    """
    for i in range(len(levels)-1, -1, -1):
        if levels[i][2] == 1:
            return i
    return -1 

@numba.njit(cache=True)
def optimized_order_book(arr: np.array, snapshots: np.array, max_size: int = 128):
    flag = False
    max_size = max_size + 16
    #levels is price, size, is_buy
    levels = np.zeros((max_size, 3), dtype=np.float32)
    n_levels = 0
    #find_max_price_index
    #find_min_price_index
    #find_price_index

    for idx, row in enumerate(arr):
        if idx % 1000000 == 0:
            print(f"Processing row {idx}, length of levels: {len(levels)}, number of 0s in snapshots: {np.sum(snapshots[idx-1000000:idx] == 0)}")
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
            if n_levels < max_size:
                levels[0] = [price, size, is_buy]
                n_levels += 1
            else:
                #find mid price

                num_buys = np.sum(levels[:, 2])
                num_sells = np.count_nonzero(levels[:, 0]) - num_buys

                min_idx = find_min_price_index(levels[:, 0])
                max_idx = find_max_price_index(levels[:, 0])


                buy_orders = levels[min_idx:min_idx+num_buys, :]
                sell_orders = levels[max_idx-num_sells:max_idx, :]

                if len(buy_orders) > max_size//2:
                    buy_orders = buy_orders[len(buy_orders)-(max_size//2):, :]
                    buy_orders = sort_levels(buy_orders)                  
                if len(sell_orders) > max_size//2:
                    sell_orders = sell_orders[:(max_size//2), :]
                    sell_orders = sort_levels(sell_orders)

                levels[(max_size//2)-len(buy_orders):(max_size//2), :] = buy_orders
                levels[(max_size//2):(max_size//2)+len(sell_orders), :] = sell_orders

                if num_buys > num_sells:
                    levels[find_min_price_index(levels[:, 0]), :] = [0, 0, 0]
                    n_levels -= 1
                elif num_sells > num_buys:
                    levels[find_max_price_index(levels[:, 0]), :] = [0, 0, 0]
                    n_levels -= 1                    
        if n_levels == max_size:
            if flag is False:
                start_idx = idx
                flag = True
            if 0.0 in levels[8:-8, :2]:
                continue
            snapshots[idx] = levels[8:-8, :2]
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
    depth = 96
    raw_data = pd.read_csv("/home/qhawkins/Desktop/eth_btc_20231201_20241201.csv", nrows=2000000) #engine="pyarrow", low_memory=True, nrows=10000000)
    #raw_data = raw_data.iloc
    raw_data.dropna(axis=0, inplace=True)
    #print(raw_data.value_counts("update_type"))
    #exit()
    #the start of the first order book snapshot is the first row of the data that doesnt have an update type of snapshot
    #starting_ob_state = raw_data.loc[raw_data["update_type"] != "snapshot"].index[0]
    #raw_data['time_exchange_int'] = raw_data['time_exchange'].apply(lambda x: int(x.timestamp()*1000))
    #raw_data['time_coinapi_int'] = raw_data['time_coinapi'].apply(lambda x: int(x.timestamp()*1000))

    #raw_data.set_index("time_coinapi_int", inplace=True)

    raw_data.drop(columns=["time_exchange", "time_coinapi"], inplace=True)
    raw_data = raw_data[raw_data["entry_sx"] != 0]
    raw_data.reset_index(drop=True, inplace=True)
    print(raw_data.describe())

    #raw_data.rename(columns={"time_exchange_int": "time_exchange", "time_coinapi_int": "time_coinapi"}, inplace=True)

    raw_data['update_type'] = raw_data['update_type'].apply(map_order_type)
    raw_data['is_buy'] = raw_data['is_buy'].apply(lambda x: 1 if x else 0)
    raw_data['entry_px'] = (raw_data['entry_px']).astype(np.float32)
    raw_data['entry_sx'] = (raw_data['entry_sx']).astype(np.float32)


    raw_data = raw_data.to_numpy(dtype=np.float32)
    print(f"0s in raw data: {np.sum(raw_data==0)}")
    #raw_data = raw_data[:10000, :]
    #raw_data = raw_data[:, :4]
    print(raw_data.shape)
    print(raw_data[0, :])
    #exit()
    results = np.zeros((len(raw_data), depth, 2), dtype=np.float32, order="C")
    ob_state, start_idx = optimized_order_book(raw_data, results, depth)
    ob_state = ob_state[start_idx+1280000:]
    print("Sliced")
    #print(ob_state[-1, :, 0])
    #ob_state = ob_state/1e7
    #ob_state = ob_state[:, :, :-1]
    #ob_state_bf16 = torch.tensor(ob_state, dtype=torch.bfloat16, requires_grad=False)
    np.save("/home/qhawkins/Desktop/CryptoOBPretraining/full_parsed.npy", ob_state)
    
    
    
    #ob_state = torch.tensor(ob_state, dtype=torch.float32, requires_grad=False)
    #print(f"bf16 {ob_state_bf16[-1, :, :]}")
    for idx, entry in enumerate(ob_state[-1, :, :]):
        print(f"Slice {idx}, price: {entry[0]}, size: {entry[1]}")
    
    #torch.save(ob_state, "full_parsed.pt")

    #np.save("test.npy", ob_state)

    #sorted_levels = sorted(ob_state[-1].keys())
    #print(ob_state[-1])