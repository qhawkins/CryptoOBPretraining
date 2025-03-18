import numba as nb
import numpy as np

@nb.njit(cache=True, parallel=True)
def create_slices(indices: list[int], data: np.dtype, temporal_dim: int, slices: np.dtype):
    for i in nb.prange(len(indices)):
        slices[i] = data[indices[i]-temporal_dim:indices[i]]
    return slices

@nb.njit(cache=True, parallel=True)
def find_zeros_in_slices(indices: list[int], data: np.dtype, temporal_dim: int):
    for i in nb.prange(len(indices)):
        if np.sum(data[indices[i]-temporal_dim:indices[i]] == 0) > 0:
            indices[i] = -1
    return indices

@nb.njit
def compute_mean_std(data):
    """
    Compute mean and standard deviation for each column in the data.
    """
    N, D = data.shape
    means = np.zeros(D)
    stds = np.zeros(D)
    
    # Compute means
    for d in range(D):
        sum_ = 0.0
        for n in range(N):
            sum_ += data[n, d]
        means[d] = sum_ / N
    
    # Compute standard deviations
    for d in range(D):
        sum_sq = 0.0
        for n in range(N):
            diff = data[n, d] - means[d]
            sum_sq += diff * diff
        stds[d] = np.sqrt(sum_sq / N)
    
    return means, stds

@nb.njit
def filter_outliers_mask(data, means, stds, k):
    """
    Create a boolean mask where True indicates non-outlier data points.
    A data point is considered non-outlier if all its statistics are within mean ± k*std.
    """
    N, D = data.shape
    mask = np.ones(N, dtype=np.bool_)
    
    for n in range(N):
        for d in range(D):
            if np.abs(data[n, d] - means[d]) > k * stds[d]:
                mask[n] = False
                break  # No need to check other dimensions
    return mask

@nb.njit
def compute_composite_metric_weighted_sum(data, weights):
    """
    Compute a composite metric as a weighted sum of the statistics.
    """
    N, D = data.shape
    composite = np.zeros(N)
    
    for n in range(N):
        sum_ = 0.0
        for d in range(D):
            sum_ += data[n, d] * weights[d]
        composite[n] = sum_
    
    return composite

@nb.njit
def assign_bins(composite_metric, bin_edges, num_bins):
    """
    Assign each data point to a bin based on the composite metric.
    """
    N = composite_metric.shape[0]
    bins = np.zeros(N, dtype=np.int32)
    
    for n in range(N):
        for b in range(1, num_bins + 1):
            if composite_metric[n] <= bin_edges[b]:
                bins[n] = b - 1  # Bin indices start at 0
                break
    return bins

@nb.njit(cache=True, parallel=True)
def compute_bin_edges(composite_metric, num_bins):
    """
    Compute bin edges based on quantiles for stratification.
    """
    bin_edges = np.zeros(num_bins + 1)
    for b in nb.prange(1, num_bins):
        percentile = b * 100.0 / num_bins
        bin_edges[b] = np.percentile(composite_metric, percentile)
    bin_edges[0] = composite_metric.min()
    bin_edges[num_bins] = composite_metric.max()
    return bin_edges

def balanced_sampling(bins, num_bins, samples_per_bin, rng):
    """
    Perform balanced sampling by selecting an equal number of samples from each bin.
    """
    balanced_indices = []
    for b in range(num_bins):
        bin_indices = np.where(bins == b)[0]
        if bin_indices.size == 0:
            continue
        if bin_indices.size >= samples_per_bin:
            sampled = rng.choice(bin_indices, size=samples_per_bin, replace=False)
        else:
            sampled = rng.choice(bin_indices, size=samples_per_bin, replace=True)
        balanced_indices.extend(sampled.tolist())
    return np.array(balanced_indices, dtype=np.int64)


if __name__ == "__main__":
    # CREATE INDICES TO BE USED BY THE DDP TRAINER DATA LOADER FOR TRAINING IN A MEMORY EFFICIENT AND MULTI-THREADED MANNER
    
    #using random number generator
    rng = np.random.default_rng()
    
    #flag for Azure
    on_azure = False

    # Constants
    K_STD = 4  # Threshold for outlier detection (mean ± K_STD * std)
    NUM_BINS = 100  # Number of bins for stratification (e.g., deciles)
    TRAIN_RATIO = 0.95  # Proportion of data to be used for training

    # define column indices for clarity
    MID_PRICE_CHANGE = 0
    AVERAGE_SPREAD = 1
    STDEV_TICK_RETURNS = 2
    MEAN_TICK_RETURNS = 3

    # composite metric weights (must sum to 1.0)
    COMPOSITE_WEIGHTS = np.array([0.4, 0.2, 0.3, 0.1])

    #number of indices to create
    TARGET_ENTRIES = 12800000

    # temporal dimension of each created slice
    temporal_dim = 256

    #depth of each created slice
    depth = 96

    #number of features in each slice (price, volume), would like to add more in the future
    features = 2
    
    #pair of the data to be used
    #pair = "BTC_USDT"
    pair = "ETH_BTC"
    #pair = "XRP_BTC"

    if on_azure:
        raw_data = np.load("/home/azureuser/datadrive/full_parsed.npy", mmap_mode="r")
        stats_data  = np.load("/home/azureuser/datadrive/stride_statistics.npy")
    else:
        raw_data = np.load(f"/media/qhawkins/SSD3/training_data/{pair}_full_parsed.npy", mmap_mode="r")
        stats_data  = np.load(f"./training_data/stride_statistics/{pair}_stride_statistics.npy")
        
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Stride statistics shape: {stats_data.shape}")
    assert raw_data.shape[0] == stats_data.shape[0]+temporal_dim, "Raw data and statistics must have the same number of entries."
    print("Data loaded")

    N_total = raw_data.shape[0]
    # with the full dataset, I kept getting OOM issues, so memory management is crucial
    del raw_data  # Release memory
    print(f"Total data points loaded: {N_total}")
    
    # Step 2: Outlier Detection and Filtering
    print("Computing mean and standard deviation for outlier detection...")
    means, stds = compute_mean_std(stats_data)
    
    print("Filtering outliers...")
    non_outlier_mask = filter_outliers_mask(stats_data, means, stds, K_STD)
    N_filtered = np.sum(non_outlier_mask)
    print(f"Data points after outlier removal: {N_filtered} ({(N_filtered / N_total) * 100:.2f}%)")
    
    # filtered statistics
    stats_data = stats_data[non_outlier_mask]
    
    # Step 3: Composite Metric Calculation
    print("Computing composite metric...")
    composite_metric = compute_composite_metric_weighted_sum(stats_data, COMPOSITE_WEIGHTS)
    del stats_data  # Release memory
    composite_metric = composite_metric.squeeze()
    print(f"Composite metric: {composite_metric.shape}")   

    # Step 4: Stratification into Bins
    print("Computing bin edges based on composite metric...")
    bin_edges = compute_bin_edges(composite_metric, NUM_BINS)
    print(f"Bin edges: {bin_edges}")
    
    print("Assigning data points to bins...")
    bins = assign_bins(composite_metric, bin_edges, NUM_BINS)
    del composite_metric  # Release memory
    
    # Step 5: Balanced Sampling
    print("Performing balanced sampling...")
    # Determine the number of samples per bin (using the minimum bin size)
    bin_counts = np.array([np.sum(bins == b) for b in range(NUM_BINS)])
    non_zero_bins = np.count_nonzero(bin_counts)
    print(f"Bin counts: {bin_counts}")
    samples_per_bin = int(TARGET_ENTRIES // non_zero_bins)
    print(f"Sampling {samples_per_bin} data points from each of the {NUM_BINS} bins.")
    
    balanced_indices = balanced_sampling(bins, NUM_BINS, samples_per_bin, rng)
    N_balanced = balanced_indices.size
    print(f"Total balanced data points: {N_balanced} ({(N_balanced / N_total) * 100:.2f}%)")
    print(f"Minimum index: {np.min(balanced_indices)}, Maximum index: {np.max(balanced_indices)}")
    #rng.shuffle(balanced_indices)
    #print(f"Shuffled indices: {balanced_indices[:10]}")
    if on_azure:
        raw_data = np.load("/home/azureuser/datadrive/full_parsed.npy")
    else:
        raw_data = np.load(f"./training_data/semi_parsed/{pair}_full_parsed.npy", mmap_mode="r")
    
    
    print(f"Data loaded, Total 0 data points: {np.sum(raw_data == 0)}")
    balanced_indices = find_zeros_in_slices(balanced_indices, raw_data, temporal_dim)
    del raw_data
    # Removing any 0s that happend to slip through
    balanced_indices = balanced_indices[balanced_indices != -1]
    N_balanced = balanced_indices.size
    print(f"Total balanced data points after zero removal: {N_balanced} ({(N_balanced / N_total) * 100:.2f}%)")
    train_indices = balanced_indices[:int(N_balanced * TRAIN_RATIO)]
    test_indices = balanced_indices[int(N_balanced * TRAIN_RATIO):]
    print(f"Training set size: {train_indices.size}")
    print(f"Testing set size: {test_indices.size}")

    if on_azure:
        np.save("/home/azureuser/datadrive/train_indices.npy", train_indices)
        np.save("/home/azureuser/datadrive/test_indices.npy", test_indices)

    else:
        np.save(f"./training_data/parsed_indices/{pair}_train_indices.npy", train_indices)
        np.save(f"./training_data/parsed_indices/{pair}_test_indices.npy", test_indices)