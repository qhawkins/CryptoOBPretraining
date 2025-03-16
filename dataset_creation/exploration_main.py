import numpy as np
import numba as nb

@nb.njit(cache=True)
def find_slice_mid(slice: np.dtype, bid_level: int, ask_level: int):
    return (slice[bid_level, 0]+slice[ask_level, 0])/2

@nb.njit(cache=True)
def find_average_slice_spread(slice: np.dtype, bid_level: int, ask_level: int):
    return np.mean(slice[:, ask_level, 0]-slice[:, bid_level, 0])

@nb.njit(cache=True)
def std_of_tick_returns(data: np.dtype, bid_level: int, ask_level: int):
    tick_returns = np.zeros(data.shape[0]-1, dtype=np.float32)
    for i in range(data.shape[0]-1):
        tick_returns[i] = data[i+1, bid_level, 0]-data[i, bid_level, 0]
    return np.std(tick_returns)

@nb.njit(cache=True)
def mean_of_tick_returns(data: np.dtype, bid_level: int, ask_level: int):
    tick_returns = np.zeros(data.shape[0]-1, dtype=np.float32)
    for i in range(data.shape[0]-1):
        tick_returns[i] = data[i+1, bid_level, 0]-data[i, bid_level, 0]
    return np.mean(tick_returns)


@nb.njit(cache=True, parallel=True)
def create_stride_statistic(data: np.dtype, temporal_dim: int, depth: int):
    bid_level = (depth//2)-1
    ask_level = (depth//2)
    results = np.zeros((data.shape[0]-temporal_dim, 6), dtype=np.float32)
    for i in nb.prange(temporal_dim, data.shape[0]):
        slice = data[i-temporal_dim:i]
        start_mid = find_slice_mid(data[i-temporal_dim], bid_level, ask_level)
        end_mid = find_slice_mid(data[i], bid_level, ask_level)
        mid_price_change = end_mid-start_mid
        average_spread = find_average_slice_spread(slice, bid_level, ask_level)
        std_tick_returns = std_of_tick_returns(slice, bid_level, ask_level)
        mean_tick_returns = mean_of_tick_returns(slice, bid_level, ask_level)
        results[i-temporal_dim] = np.array([mid_price_change, average_spread, std_tick_returns, mean_tick_returns, start_mid, i], dtype=np.float32)
    return results

@nb.njit(cache=True, parallel=True)
def extract_statistics_from_slices(data: np.dtype, outputs: np.dtype, depth: int):
    bid_level = (depth//2)-1
    ask_level = (depth//2)
    
    for i in nb.prange(data.shape[0]):
        price_levels = data[i, :, 0]
        volumes = data[i, :, 1]
        mid_prices = (price_levels[bid_level]+price_levels[ask_level])/2
        spreads = price_levels[ask_level]-price_levels[bid_level]
        bid_spread = mid_prices - price_levels[bid_level]
        ask_spread = price_levels[ask_level] - mid_prices
        average_volume = np.mean(volumes)
        outputs[i] = np.array([mid_prices, spreads, bid_spread, ask_spread, average_volume, i], dtype=np.float32)
    return outputs

if __name__ == "__main__":
    # whether to use the Azure filesystem or not
    on_azure = False

    # temporal dim, how long the each slice goes through time
    temporal_dim = 256

    # select which pair you want to parse
    #pair = "BTC_USDT"
    #pair = "ETH_BTC"
    pair = "XRP_BTC"
    
    if on_azure:
        ds_path = "/home/azureuser/datadrive/full_parsed.npy"
    else:
        ds_path = f"./training_data/semi_parsed/{pair}_full_parsed.npy"

    #loading with mmap mode for memory efficiency (important with large datasets)
    data = np.load(ds_path, mmap_mode="r")

    #dynamic depth selecton based on the shape of the parsed data
    depth = data.shape[1]
    print(f"Shape of input data: {data.shape}")

    # numba acceleration for speed, not as important compared to the other parsing files but still beneficial
    #output schema: (mid price, spread, bid spread, ask spread, volume at price level)
    results = create_stride_statistic(data, temporal_dim, depth)

    print(f"Output statistics for first slice: {results[0]}")
    
    if on_azure:
        np.save("/home/azureuser/datadrive/stride_statistics.npy", results)
    else:
        np.save(f"./training_data/stride_statistics/{pair}_stride_statistics.npy", results)