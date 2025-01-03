from ray.tune.stopper import Stopper
from torch.utils.data import Dataset
import numpy as np
import torch

def normalize_slice(data: torch.Tensor):
	mean = torch.mean(data)
	std = torch.std(data)
	return (data - mean) / (std + 1e-6)


def normalize_data(data: torch.Tensor):
	"""
	Normalizes the input data to have zero mean and unit variance.
	
	:param data: A torch.Tensor of shape (time, depth, features)
	:return: A torch.Tensor of shape (time, depth, features) with zero mean and unit variance
	"""

	data[:, :, 0] = normalize_slice(data[:, :, 0])
	data[:, :, 1] = normalize_slice(data[:, :, 1])
	#data[:, :, 2] = normalize_slice(data[:, :, 2])

	return data

class CombinedNanPlateauStopper(Stopper):
	def __init__(self, metric, std, num_results, grace_period, metric_threshold, mode):
		self.nan_stopper = NanStopper(metric=metric)
		self.loss_stopper = LossIncreaseStopper(metric=metric, grace_period=grace_period)
		self.high_loss_stopper = HighLossStopper(metric=metric, metric_threshold=metric_threshold, mode=mode, grace_period=grace_period)

	def __call__(self, trial_id, result):
		return self.nan_stopper(result) or self.loss_stopper(trial_id, result) or self.high_loss_stopper(result)

	def stop_all(self):
		return self.loss_stopper.stop_all()

class HighLossStopper(Stopper):
	def __init__(self, metric="loss", metric_threshold=1.0, mode="min", grace_period=3):
		self.metric = metric
		self.metric_threshold = metric_threshold
		self.mode = mode
		self.grace_period = grace_period

	def __call__(self, result):
		if result['iteration'] > self.grace_period:
			return result[self.metric] > self.metric_threshold
		else:
			return False

	def stop_all(self):
		return False

class NanStopper(Stopper):
	def __init__(self, metric="loss"):
		self.metric = metric

	def __call__(self, result):
		return np.isnan(result[self.metric])

	def stop_all(self):
		return False
	
class LossIncreaseStopper(Stopper):
	def __init__(self, metric="loss", grace_period=20):
		self.metric = metric
		self.grace_period = grace_period
		self.losses = {}

	def __call__(self, trial_id, result):
		if trial_id not in self.losses.keys():
			self.losses[trial_id] = []
		if len(self.losses[trial_id]) < self.grace_period:
			self.losses[trial_id].append(result[self.metric])
			return False
		else:
			self.losses[trial_id].pop(0)
			self.losses[trial_id].append(result[self.metric])
			if self.losses[trial_id][0] < self.losses[trial_id][-1]:
				return True
			else:
				return False

	def stop_all(self):
		return False

class PretrainingDataset(Dataset):
	def __init__(self, start_idx: int, end_idx: int, data_path: str, temporal_offset: int = 512):
		self.data = np.load(data_path, mmap_mode="r")
		self.start_idx = start_idx
		self.end_idx = end_idx
		self.offset = temporal_offset
		#self.length = len(self.data)-self.offset
		self.length = self.end_idx - self.start_idx - self.offset
		print(f"PretrainingDataset initialized with {self.length} rows on {'cuda'}")
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx):
		if isinstance(idx, int):
			if idx < 0 or idx >= self.length:
				raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.length}")
			data_slice = torch.from_numpy(self.data[idx+self.start_idx:idx+self.start_idx+self.offset].copy())
			normalized = normalize_data(data_slice)
			nan_count = torch.sum(torch.isnan(normalized))
			#print(f"Shape of normalized slice {idx}:{idx+self.offset} is {normalized.shape}")
			if nan_count > 0:
				print(f"Found {nan_count} nans in slice {idx}")
				print(normalized)
				raise ValueError(f"Nans found in normalized slice with indices {idx}:{idx+self.offset}")
			return normalized.pin_memory()
		elif isinstance(idx, slice):
			normalized = self.data[idx]
			#print(f"Shape of normalized slice in slice {idx} is {normalized.shape}")
			return normalized
		else:
			raise TypeError(f"Invalid index type: {type(idx)}. Expected int or slice.")
