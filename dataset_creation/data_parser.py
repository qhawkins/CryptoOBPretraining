import pandas as pd
import numpy as np
import numba
from datetime import datetime
import time
from copy import deepcopy

#
# C++ Parsing Functions Converted to Python with Numba
#

def get_time_from_string(time_string):
    """
    Convert a time string to a Unix timestamp
    
    Python implementation of C++ function:
    std::chrono::time_point<std::chrono::system_clock> get_time_from_string(const std::string& time_string)
    """
    try:
        dt = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
        return time.mktime(dt.timetuple())
    except Exception as e:
        print(f"Error converting time string: {e}")
        return 0

@numba.njit
def get_time_from_double(time_double):
    """
    Convert a double to a timestamp (nanoseconds)
    
    Python implementation of C++ function:
    std::chrono::time_point<std::chrono::system_clock> get_time_from_double(double time_double)
    """
    return np.int64(time_double)

def numpy_to_flat_vector(numpy_array):
    """
    Convert a 2D numpy array to a flat 1D vector
    
    Python implementation of C++ function:
    std::vector<double> numpyToFlatVector(const py::array_t<double>& numpyArray)
    """
    if len(numpy_array.shape) != 2:
        raise ValueError("Input array must have 2 dimensions")
    
    n, cols = numpy_array.shape
    print(f"n: {n}, cols: {cols}")
    
    # Create a flattened copy of the array
    total_size = n * cols
    result = np.zeros(total_size, dtype=numpy_array.dtype)
    result[:] = numpy_array.flatten()
    
    return result

@numba.njit(cache=True)
def parse_into_slices(data, result):
    """
    Parse flat data into order objects.
    
    Python implementation of C++ function:
    void parse_into_slices(const std::vector<double>& data, std::vector<Order>& result)
    
    Parameters:
    data (np.array): Flat array of order data
    result (np.array): Output array to store parsed orders
    
    Returns:
    np.array: Updated result array
    """
    for i in range(0, len(data), 5):
        idx = i // 5
        if idx < len(result):
            # C++ order: time_exchange, update_type, is_buy, price, size
            result[idx, 0] = data[i]      # time
            result[idx, 1] = data[i+1]    # update_type
            result[idx, 2] = data[i+2]    # is_buy (0 for ask, 1 for bid, 2 for null)
            result[idx, 3] = data[i+3]    # price
            result[idx, 4] = data[i+4]    # size
    return result

@numba.njit(cache=True)
def get_slice(vec, idx, features):
    """
    Extract a slice from an array
    
    Python implementation of C++ function:
    void getSlice(const std::vector<double>& vec, std::vector<double>& result, int idx, int features)
    
    Parameters:
    vec (np.array): Input array
    idx (int): Slice index
    features (int): Number of features per slice
    
    Returns:
    np.array: Extracted slice
    """
    start = idx * features
    end = start + features
    if end <= len(vec):
        return vec[start:end].copy()
    else:
        return np.zeros(features, dtype=vec.dtype)

def write_vector_to_csv(data, file_name):
    """
    Write a numpy array to a CSV file
    
    Python implementation of C++ function:
    void write_vector_to_csv(const std::vector<float>& data, const std::string& file_name)
    
    Parameters:
    data (np.array): Data to write
    file_name (str): Output file path
    """
    try:
        with open(file_name, 'w') as f:
            for i in range(len(data)):
                f.write(str(data[i]))
                if i < len(data) - 1:
                    f.write(',')
        print(f"Successfully wrote data to {file_name}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

#
# Existing Helper Functions
#

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
    storage = np.zeros((len(levels), 2), dtype=np.float64)
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

#
# Main Order Book Processing Function
#

@numba.njit(cache=True)
def optimized_order_book(arr: np.array, snapshots: np.array, max_size: int = 128, subsample: int = 10):
    flag = False
    max_size = max_size + 16
    buys = np.zeros((max_size//2, 2), dtype=np.float64)
    sells = np.zeros((max_size//2, 2), dtype=np.float64)
    consolidated = np.zeros((max_size, 2), dtype=np.float64)
    
    # Separate counter for tracking snapshots
    snapshot_counter = 0
    
    for idx, row in enumerate(arr):
        if idx % 10000000 == 0:
            print(f"Processing row {idx}, snapshots saved: {snapshot_counter}")
        
        # Extract order details from row
        price = row[3]
        size = row[4]
        update_type = row[1]
        is_buy = row[2]
        
        # Skip invalid entries
        if price <= 0 or np.isnan(price):
            continue
            
        if is_buy == 1.0:  # Bid side
            price_idx = find_price_index(buys[:, 0], price)
            if price_idx != -1:
                buys = order_book_update(buys, update_type, price_idx, price, size)
                buys = sort_levels(buys)
            else:
                buys = add_slice(buys, price, size)        
                buys = sort_levels(buys)
        elif is_buy == 0.0:  # Ask side
            price_idx = find_price_index(sells[:, 0], price)
            if price_idx != -1:
                sells = order_book_update(sells, update_type, price_idx, price, size)
                sells = sort_levels(sells)
            else:
                sells = add_slice(sells, price, size)
                sells = sort_levels(sells)
            
        consolidated[:max_size//2, :] = buys
        consolidated[max_size//2:, :] = sells
        
        if flag is False:
            start_idx = idx
            flag = True
        
        # Take snapshots at specified intervals
        if (idx+1) % subsample == 0:
            # Check if the consolidated array has valid data
            valid_section = consolidated[8:-8]
            has_valid_data = not np.any(valid_section == 0)
            
            if has_valid_data and snapshot_counter < len(snapshots):
                #print(f"Adding snapshot at idx {idx}, snapshot_counter {snapshot_counter}")
                snapshots[snapshot_counter] = valid_section
                snapshot_counter += 1

    print(f"Total snapshots saved: {snapshot_counter}")
    return snapshots, start_idx

#
# Main Execution
#

if __name__ == "__main__":
    # price level depth of the parsed order book
    depth = 96
    
    # level of subsampling for the parsed order book (1 means no subsampling)
    '''XRP'''
    #subsample = 5
    '''ETH'''
    subsample = 15

    # whether the parsing is happening on Azure VM or not
    azure = False
    
    # the pair to parse
    #pair = "XRP_BTC"
    pair = "ETH_BTC"
    # Load raw data

    input_path = f"./training_data/raw_sliced/{pair}_20240101_20241201_sliced.npy"
    raw_data = np.load(input_path, mmap_mode='r')

    print("loaded")
    print(raw_data.shape)

    # creating results array for performance reasons, c contiguous for faster reads/writes
    results = np.zeros((len(raw_data)//subsample, depth, 2), dtype=np.float64, order="C")
    
    # numba optimized function to create/parse the order book
    ob_state, start_idx = optimized_order_book(raw_data, results, depth, subsample=subsample)

    print(f"ob_state shape: {ob_state.shape}")

    # remove any rows where all entries are 0, which would mess up training
    ob_state = ob_state[~np.any(ob_state == 0, axis=(1, 2))]

    print(f"Final ob_state shape: {ob_state.shape}")

    np.save(f"/media/qhawkins/SSD3/training_data/{pair}_full_parsed.npy", ob_state)
        
    # diagnostic printing to verify integrity of parsing
    for idx, entry in enumerate(ob_state[-1, :, :]):
        print(f"Slice {idx}, price: {entry[0]}, size: {entry[1]}")