import pandas as pd
import numpy as np
import torch
import os
import glob
import pandas as pd
from copy import deepcopy
import numba

def build_order_book_snapshots(df: pd.DataFrame):
    """
    Builds a list of order book snapshots from the given DataFrame.
    Each row in `df` represents one update to the order book.
    
    :param df: A pandas DataFrame with columns:
               ['time_exchange', 'time_api', 'update_type',
                'is_buy', 'entry_px', 'entry_sz']
    :return: A list of dictionaries, where each dictionary represents
             the state of the order book after applying a single row's update.
    """
    
    # We store order book levels in a dictionary:
    #   levels[price] = {
    #       'price'         : float,
    #       'size'          : float,
    #       'is_buy'        : bool,
    #       'time_exchange' : float or datetime,
    #       'time_api'      : float or datetime
    #   }
    levels = {}
    
    # List to store a "snapshot" (deep copy) of the order book after each update.
    snapshots = []
    
    # Iterate over rows; each row is an update event
    for idx, row in df.iterrows():
        print(f"Processing row {idx}, length of levels: {len(levels)}")
        price = row['entry_px']
        size = row['entry_sx']
        update_type = row['update_type']
        #current_time = row['time_coinapi']
        #print(f"Price: {price}, Size: {size}, Update Type: {update_type}, Current Time: {current_time}, ")
        #for key in levels.keys():
        #    levels[key]['time_since_last_update'] = current_time - levels[key]['time_coinapi'] + 1
        
        # If this price level already exists in the order book:
        if price in levels:
            # Update the timestamps
            levels[price]['time_exchange'] = row['time_exchange']
            levels[price]['time_coinapi'] = row['time_coinapi']
            #levels[price]['time_since_last_update'] = current_time - levels[price]['time_coinapi'] + 1
            
            # Apply update type logic
            if update_type == 'ADD':
                # ADD means increment the size
                levels[price]['size'] += size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 'SUB':
                # SUB means decrement the size
                levels[price]['size'] -= size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 'MATCH':
                # MATCH also means decrement the size
                levels[price]['size'] -= size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 'SET':
                # SET means overwrite the size and set is_buy
                levels[price]['size'] = size
                levels[price]['is_buy'] = bool(row['is_buy'])
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 'DELETE':
                # DELETE means remove the price level entirely
                del levels[price]
                
            elif update_type == 'SNAPSHOT':
                # SNAPSHOT: first set the existing level's size,
                # then create a new level object and assign it.
                
                # 1) Overwrite existing size
                levels[price]['size'] = size
                levels[price]['is_buy'] = bool(row['is_buy'])
                
                # 2) Create a new level from scratch and assign
                new_level = {
                    'price': row['entry_px'],
                    'size': size,
                    'is_buy': bool(row['is_buy']),
                    'time_exchange': row['time_exchange'],
                    'time_coinapi': row['time_coinapi'],
                    #'time_since_last_update': 1
                }
                levels[price] = new_level
                continue
                
            else:
                # Unknown update type (optional: handle error/logging)
                pass
                
        else:
            # If this price level does not exist in the book:
            # We create a brand-new level and insert it
            new_level = {
                'price': price,
                'size': size,
                'is_buy': bool(row['is_buy']),
                'time_exchange': row['time_exchange'],
                'time_coinapi': row['time_coinapi'],
                #'time_since_last_update': 1
            }
            levels[price] = new_level

        # If the number of levels exceeds 256, find the target index to remove
        # to do this, find the mid price and if the remove the key that is farthest from the mid price
        # to find the mid price, find the index where the is_buy column converts from True to False, making sure that the levels are ordered by price (in ascending order)

        if len(levels) > 64:
            count_of_is_buy = sum([1 for key in levels.keys() if levels[key]['is_buy']])
            count_of_is_sell = sum([1 for key in levels.keys() if not levels[key]['is_buy']])
            sorted_keys = sorted(levels.keys())
            if count_of_is_buy > count_of_is_sell:
                min_key = min(sorted_keys)
                del levels[min_key]
            else:
                max_key = max(sorted_keys)
                del levels[max_key]

            #min_key = min(levels.keys())
            #max_key = max(levels.keys())
            #mid_price = find_mid(levels)
            #distance_to_min = abs(mid_price - min_key)
            #distance_to_max = abs(mid_price - max_key)
            #if distance_to_min > distance_to_max:
            #    del levels[min_key]
            #else:
            #    del levels[max_key]

            #print(f"Count of is_buy: {sum([1 for key in levels.keys() if levels[key]['is_buy']])}")
            #print(f"Deleted key: {min_key if distance_to_min > distance_to_max else max_key}, Mid price: {mid_price}, Distance to min: {distance_to_min}, Distance to max: {distance_to_max}")

        if len(levels) == 64:
            snapshots.append(pd.DataFrame.from_dict(deepcopy(levels), orient='index').drop(columns=["time_exchange", "time_coinapi"]).sort_values(by="price", ascending=True).reset_index(drop=True))
    
    return snapshots

@numba.njit(cache=True, fastmath=True)
def convert_dict_to_arr(levels: dict):
    intermediary_array = np.zeros((len(levels), 3), dtype=np.int64)
    for idx, key in enumerate(sorted(levels.keys())):
        intermediary_array[idx, 0] = levels[key]['price']
        intermediary_array[idx, 1] = levels[key]['size']
        intermediary_array[idx, 2] = levels[key]['is_buy']
    return intermediary_array

@numba.njit(cache=True, fastmath=True)
def nb_build_order_book_snapshots(arr: np.array, snapshots: np.array):
    levels = {}
    flag = False
    for idx, row in enumerate(arr):
        if idx % 1000000 == 0:
            print(f"Processing row {idx}, length of levels: {len(levels)}")
        #print(len(row))
        price = row[2]
        size = row[3]
        update_type = row[0]
        is_buy = row[1]
        if price in levels:            
            # Apply update type logic
            if update_type == 0:
                # ADD means increment the size
                levels[price]['size'] += size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 1:
                # SUB means decrement the size
                levels[price]['size'] -= size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 2:
                # MATCH also means decrement the size
                levels[price]['size'] -= size
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 3:
                # SET means overwrite the size and set is_buy
                levels[price]['size'] = size
                levels[price]['is_buy'] = is_buy
                #levels[price]['time_since_last_update'] = 1
                
            elif update_type == 4:
                # DELETE means remove the price level entirely
                del levels[price]
                
            elif update_type == 5:
                # SNAPSHOT: first set the existing level's size,
                # then create a new level object and assign it.
                
                # 1) Overwrite existing size
                levels[price]['size'] = size
                levels[price]['is_buy'] = is_buy
                
                # 2) Create a new level from scratch and assign
                new_level = {
                    'price': price,
                    'size': size,
                    'is_buy': is_buy,
                    #'time_since_last_update': 1
                }
                levels[price] = new_level
                continue
                
            else:
                # Unknown update type (optional: handle error/logging)
                pass
                
        else:
            # If this price level does not exist in the book:
            # We create a brand-new level and insert it
            #print("Creating new level")
            new_level = {
                'price': price,
                'size': size,
                'is_buy': is_buy,
                #'time_since_last_update': 1
            }
            levels[price] = new_level

        # If the number of levels exceeds 256, find the target index to remove
        # to do this, find the mid price and if the remove the key that is farthest from the mid price
        # to find the mid price, find the index where the is_buy column converts from True to False, making sure that the levels are ordered by price (in ascending order)

        if len(levels) > 64:
            #print("Excess length")
            count_of_is_buy = sum([1 for key in levels.keys() if levels[key]['is_buy']])
            count_of_is_sell = sum([1 for key in levels.keys() if not levels[key]['is_buy']])
            sorted_keys = sorted(levels.keys())
            if count_of_is_buy > count_of_is_sell:
                min_key = min(sorted_keys)
                del levels[min_key]
            else:
                max_key = max(sorted_keys)
                del levels[max_key]

        if len(levels) == 64:
            if flag is False:
                start_idx = idx
                flag = True
            #print(f"Snapshot {idx}")
            snapshots[idx] = convert_dict_to_arr(levels)
    
    return snapshots, start_idx

def center_ob_slices(data: pd.DataFrame, target_depth: int):
    """
    Center the order book slices around the target depth.
    If the target depth is 5, the output will be a slice of the order book
    with 5 levels on each side of the target depth.
    
    :param data: A pandas DataFrame with columns:
                 ['price', 'size', 'is_buy', 'time_since_last_update']
    :param target_depth: The target depth to center the slices around
    :return: A pandas DataFrame with columns:
             ['price', 'size', 'is_buy', 'time_since_last_update']
    """
    len_data = len(data)

    # find the index where the is_buy column converts from True to False
    target_idx = data[data['is_buy'] == False].index[0]
    data['is_buy'] = data['is_buy'].apply(lambda x: 1 if x else 0).astype(np.int64)
    data['price'] = (data['price']*1e9).astype(np.int64)
    data['size'] = (data['size']*1e9).astype(np.int64)
    #data['time_since_last_update'] = data['time_since_last_update'].astype(np.int64)
    data = data.to_numpy(dtype=np.int64)
    
    ## Calculate the start and end indices for the slice
    #start_idx = target_idx - target_depth/2
    #end_idx = target_idx + target_depth/2
    
    # If the slice is out of bounds, add padding to the start and/or end
    #if start_idx < 0:
    #    padding_needed = int(abs(start_idx))
    #    start_idx = 0
    #    beginning_padding = np.zeros((padding_needed, 3), dtype=np.int64)
    #    data = np.vstack((beginning_padding, data))
    #if end_idx > len_data:
    #    padding_needed = int(end_idx - len_data)
    #    ending_padding = np.zeros((padding_needed, 3), dtype=np.int64)
    #    data = np.vstack((data, ending_padding))
    #sliced_data = data[int(start_idx):int(start_idx+target_depth)]
    
    #return sliced_data
    return data

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
    depth = 64
    raw_data = pd.read_csv("/home/qhawkins/Desktop/eth_btc_20231201_20241201_fragment.csv", engine="pyarrow")
    raw_data.dropna(axis=0, inplace=True)
    #print(raw_data.value_counts("update_type"))
    #exit()
    #the start of the first order book snapshot is the first row of the data that doesnt have an update type of snapshot
    #starting_ob_state = raw_data.loc[raw_data["update_type"] != "snapshot"].index[0]
    raw_data['time_exchange_int'] = raw_data['time_exchange'].apply(lambda x: int(x.timestamp()*1000))
    raw_data['time_coinapi_int'] = raw_data['time_coinapi'].apply(lambda x: int(x.timestamp()*1000))

    #raw_data.set_index("time_coinapi_int", inplace=True)

    raw_data.drop(columns=["time_exchange", "time_coinapi"], inplace=True)

    print(raw_data.describe())

    raw_data.rename(columns={"time_exchange_int": "time_exchange", "time_coinapi_int": "time_coinapi"}, inplace=True)

    raw_data['update_type'] = raw_data['update_type'].apply(map_order_type)
    raw_data['is_buy'] = raw_data['is_buy'].apply(lambda x: 1 if x else 0)
    raw_data['entry_px'] = (raw_data['entry_px']*1e9).astype(np.int64)
    raw_data['entry_sx'] = (raw_data['entry_sx']*1e9).astype(np.int64)

    print(raw_data.info())
    #exit()

    raw_data = raw_data.to_numpy(dtype=np.int64)
    raw_data = raw_data[:, :4]
    print(raw_data.shape)
    print(raw_data[0, :])
    #exit()
    results = np.zeros((len(raw_data), depth, 3), dtype=np.int64)
    ob_state, start_idx = nb_build_order_book_snapshots(raw_data, results)
    ob_state = ob_state[start_idx+1:]
    ob_state = ob_state/1e7
    ob_state = ob_state[:, :, :-1]
    #ob_state_bf16 = torch.tensor(ob_state, dtype=torch.bfloat16, requires_grad=False)
    ob_state = torch.tensor(ob_state, dtype=torch.float32, requires_grad=False)
    #print(f"bf16 {ob_state_bf16[-1, :, :]}")
    print(f"fp32 {ob_state[-1, :, :]}")
    print(f"fp32 {ob_state[0, :, :]}")
    torch.save(ob_state, "test.pt")

    #np.save("test.npy", ob_state)

    #sorted_levels = sorted(ob_state[-1].keys())
    #print(ob_state[-1])