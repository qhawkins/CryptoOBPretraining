import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"


# Import your existing classes and functions
from training_classes import PretrainingDataset
from models import (
	SmallFCModel, MediumFCModel, LargeFCModel,
	ShallowLSTMModel, DeepLSTMModel, TinyLSTMModel, TinyTransformerModel
)

# Define your apply_mask function (unchanged)
def apply_mask(inputs: torch.Tensor, mask_percentage=0.15, mask_value=0.0, device='cuda'):
	"""
	Applies masking to the input tensor.
	
	Args:
		inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, features).
		mask_percentage (float): Fraction of entries to mask.
		mask_value (float): Value to replace masked entries with.
		device (str): Device to perform masking on.
	
	Returns:
		masked_inputs (torch.Tensor): Tensor with masked entries.
		mask (torch.Tensor): Boolean mask indicating which entries were masked.
	"""
	# Generate a mask for 15% of the entries
	mask = torch.rand(inputs.shape, device=device, requires_grad=False, dtype=torch.bfloat16) < mask_percentage
	mask = mask.to(device, non_blocking=True)
	
	# Replace masked entries in inputs with mask_value
	masked_inputs = inputs.clone()
	masked_inputs[mask] = mask_value
	
	return masked_inputs, mask

# Adapted Trainer class with DDP support
class Trainer:
	def __init__(self, config, rank, world_size, data_parallel_group, train_dataset, test_dataset):
		self.config = config
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.rank = rank
		self.world_size = world_size
		self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		self.data_parallel_group = data_parallel_group
		# Initialize the model
		self.initialize_model()
		
		# Initialize loss function and optimizer
		self.initialize_criterion_optimizer()
		
		# Load data
		self.load_data()
		
		# Initialize other components
		self.initialize_training_components()
		
	def load_model(self, path: str):
		self.model = TinyTransformerModel((self.config["temporal_dim"], self.config["depth_dim"], 2), (self.config["temporal_dim"], self.config["depth_dim"], 2), 0.25)
		state_dict = torch.load(path)
		state_dict = state_dict['model_state_dict']
		state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
		self.model.load_state_dict(state_dict)
		self.model = self.model.to(self.device)
		self.model = self.model.train()
		#self.model = self.model.compile()

	def initialize_model(self):
		model_size = self.config['model_size']
		temporal_dim = self.config['temporal_dim']
		depth_dim = self.config['depth_dim']
		dropout = self.config['dropout']
		
		model_classes = {
			'small': SmallFCModel,
			'medium': MediumFCModel,
			'large': LargeFCModel,
			'shallow_lstm': ShallowLSTMModel,
			'deep_lstm': DeepLSTMModel,
			'tiny_lstm': TinyLSTMModel,
			'tiny_transformer': TinyTransformerModel
		}
		
		if model_size not in model_classes:
			raise ValueError(f"Unsupported model size: {model_size}")
		
		if self.config['load_model']:
			self.load_model(self.config['model_path'])
			print(f"Model loaded from {self.config['model_path']}")

		else:
			self.model = model_classes[model_size](
				(temporal_dim, depth_dim, 2),
				(temporal_dim, depth_dim, 2),
				dropout
			).to(self.device)
			
			# Compile the model for potential performance benefits
			self.model = torch.compile(self.model)
			
			# Wrap the model with DDP
		self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True, process_group=self.data_parallel_group)
		
	def initialize_criterion_optimizer(self):
		loss_type = self.config['loss']
		if loss_type == 'mse':
			self.criterion = torch.nn.MSELoss().to(self.device)
		elif loss_type == 'mae':
			self.criterion = torch.nn.L1Loss().to(self.device)
		elif loss_type == 'huber':
			self.criterion = torch.nn.HuberLoss().to(self.device)
		elif loss_type == 'smoothl1':
			self.criterion = torch.nn.SmoothL1Loss().to(self.device)
		else:
			raise ValueError(f"Unsupported loss type: {loss_type}")
		
		optimizer_type = self.config['optimizer']
		lr = self.config['lr']
		if optimizer_type == 'adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		elif optimizer_type == 'sgd':
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
		elif optimizer_type == 'adamw':
			self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
		else:
			raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
		
	def load_data(self):
		# Load your dataset
		
		# Create dataset splits
		self.split_data()
		
		# Create DistributedSampler
		self.train_sampler = DistributedSampler(
			self.train_ds,
			num_replicas=self.world_size,
			rank=self.rank,
			shuffle=True
		)
		
		self.val_sampler = DistributedSampler(
			self.val_ds,
			num_replicas=self.world_size,
			rank=self.rank,
			shuffle=False
		)
		
		self.test_sampler = DistributedSampler(
			self.test_ds,
			num_replicas=self.world_size,
			rank=self.rank,
			shuffle=False
		)
		
		# Create DataLoaders with DistributedSampler
		self.train_dataloader = DataLoader(
			self.train_ds,
			batch_size=self.config['batch_size'],
			sampler=self.train_sampler,
			drop_last=True,
			num_workers=8,
			pin_memory=True,
		)
		
		self.val_dataloader = DataLoader(
			self.val_ds,
			batch_size=self.config['batch_size'],
			sampler=self.val_sampler,
			drop_last=True,
			num_workers=8,
			pin_memory=True,
		)
		
		self.test_dataloader = DataLoader(
			self.test_ds,
			batch_size=self.config['batch_size'],
			sampler=self.test_sampler,
			drop_last=True,
			num_workers=8,
			pin_memory=True,
		)
		
	def split_data(self):
		total_size = np.load(self.train_dataset, mmap_mode="r").shape[0]

		train_size = int(self.config['split_ratios'][0] * total_size)
		val_size = int(self.config['split_ratios'][1] * total_size)

		test_size = np.load(self.test_dataset, mmap_mode="r").shape[0]
		
		self.train_ds = PretrainingDataset(
			self.train_dataset,
			0, 
			train_size, 
			self.config['temporal_dim'],
			self.config['depth_dim']
		)
		self.val_ds = PretrainingDataset(
			self.train_dataset,
			train_size,
			train_size + val_size,
			self.config['temporal_dim'],
			self.config['depth_dim']
		)
		self.test_ds = PretrainingDataset(
			self.test_dataset,
			0, 
			test_size, 
			self.config['temporal_dim'],
			self.config['depth_dim']
		)
		
	def initialize_training_components(self):
		self.model_name = self.config["model_name"]
		self.train_loss_history = []
		self.val_loss_history = []
		self.test_loss_history = []
		self.ratios = self.config['split_ratios']
		self.lr_decay_factor = self.config['lr_decay_factor']
		self.lr_decay_patience = self.config['lr_decay_patience']
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=self.lr_decay_factor,
			patience=self.lr_decay_patience
		)
		self.scaler = torch.amp.GradScaler(f"cuda:{self.rank}")
		self.early_stopping_patience = self.config['early_stopping_patience']
		self.saved_models = deque(maxlen=self.early_stopping_patience + 1)
		self.best_val_loss = float('inf')
		self.best_model_path = self.config['best_model_path']
		self.step_losses = []
		
	def save_model(self, epoch, val_loss):
		if self.rank != 0:
			return  # Only the master process saves the model
		
		model_path = (
			f"/home/azureuser/single_models/{self.model_name}_val_loss_"
			f"{str(round(val_loss, 8)).replace('.', '')}_epoch_{epoch}_"
			f"{self.config['loss']}_{self.config['model_size']}.pth"
		)
		torch.save({
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),  # Access the underlying model
			'optimizer_state_dict': self.optimizer.state_dict(),
			'val_loss': val_loss,
			'loss_function': self.config['loss'],
			"step_losses": self.step_losses
		}, model_path)
		self.saved_models.append((model_path, val_loss))
		
		if val_loss < self.best_val_loss:
			self.best_val_loss = val_loss
			self.best_model_path = model_path
	
	def print_model_params(self):
		if self.rank == 0:
			print(f"Model: {self.model_name}")
			print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
	
	def train(self, epochs):
		self.print_model_params()
		best_val_loss = float('inf')
		epochs_without_improvement = 0
		
		for epoch in range(epochs):
			if epoch % 100 == 0:
				print(f"Epoch {epoch} started.")
			self.train_sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
			self.model.train()
			avg_train_loss = 0
			self.step_losses = []

			for i, data in enumerate(self.train_dataloader):
				data = data.to(self.device, non_blocking=True)
				masked_inputs, mask = apply_mask(
					data,
					mask_percentage=self.config['mask_perc'],
					mask_value=0.0,
					device=self.device
				)
				
				self.optimizer.zero_grad()
				
				with torch.amp.autocast(f"cuda:{self.rank}"):
					outputs = self.model(masked_inputs)
					loss = self.criterion(outputs[~mask], data[~mask])
				
				self.scaler.scale(loss).backward()
				self.scaler.step(self.optimizer)
				self.scaler.update()
				
				avg_train_loss += loss.item()
				self.step_losses.append(loss.item())
			
			avg_train_loss /= (i + 1)
			
			# Validation Phase
			self.model.eval()
			avg_val_loss = 0
			with torch.no_grad():
				for i, data in enumerate(self.val_dataloader):
					data = data.to(self.device, non_blocking=True)
					masked_inputs, mask = apply_mask(
						data,
						mask_percentage=self.config['mask_perc'],
						mask_value=0.0,
						device=self.device
					)
					
					with torch.amp.autocast(f"cuda:{self.rank}"):
						outputs = self.model(masked_inputs)
						loss = self.criterion(outputs[~mask], data[~mask])
					
					avg_val_loss += loss.item()
				avg_val_loss /= (i + 1)
			
			# Only the master process logs and saves models
			if self.rank == 0:
				self.train_loss_history.append(avg_train_loss)
				self.val_loss_history.append(avg_val_loss)
				self.scheduler.step(avg_val_loss)
				print(f'Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}')
				self.save_model(epoch, avg_val_loss)
			
				if avg_val_loss < best_val_loss:
					best_val_loss = avg_val_loss
					epochs_without_improvement = 0
					if avg_val_loss < 0.45:
						self.save_model(epoch, avg_val_loss)
				else:
					epochs_without_improvement += 1
				
				# Early Stopping Logic
				if epochs_without_improvement >= self.early_stopping_patience:
					print("Early stopping triggered.")
					break
		
		if self.rank == 0:
			print(f"Training completed. Best model saved at: {self.best_model_path}")
	
	def test(self):
		if self.rank != 0:
			return  # Only the master process performs testing
		
		if self.best_model_path:
			checkpoint = torch.load(self.best_model_path, map_location=self.device)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			print(f"Loaded best model from epoch {checkpoint['epoch']} for testing.")
		
		self.model.eval()
		test_loss = 0
		mae_loss = torch.nn.L1Loss().to(self.device)
		with torch.no_grad():
			for i, data in enumerate(self.test_dataloader):
				data = data.to(self.device)
				masked_inputs, mask = apply_mask(
					data,
					mask_percentage=self.config['mask_perc'],
					mask_value=0.0,
					device=self.device
				)
				
				with torch.amp.autocast(f"cuda:{self.rank}"):
					outputs = self.model(masked_inputs)
					loss = self.criterion(outputs[~mask], data[~mask])
					test_loss += loss.item()
			
			avg_test_loss = test_loss / (i + 1)
			print(f'{self.model_name} Test Loss ({self.config["loss"]}): {avg_test_loss:.6f}')
		
		# Additionally compute MAE loss
		test_loss = 0
		with torch.no_grad():
			for i, data in enumerate(self.test_dataloader):
				data = data.to(self.device)
				masked_inputs, mask = apply_mask(
					data,
					mask_percentage=self.config['mask_perc'],
					mask_value=0.0,
					device=self.device
				)
				
				with torch.amp.autocast(f"cuda:{self.rank}"):
					outputs = self.model(masked_inputs)
					loss = mae_loss(outputs[~mask], data[~mask])
					test_loss += loss.item()
		
		avg_test_mae_loss = test_loss / (i + 1)
		print(f'{self.model_name} Test Loss (MAE): {avg_test_mae_loss:.6f}')
	
	def plot_loss(self):
		if self.rank != 0:
			return  # Only the master process plots the loss
		
		plt.figure(figsize=(10, 6))
		plt.plot(self.train_loss_history, label='Train Loss')
		plt.plot(self.val_loss_history, label='Validation Loss')
		plt.title(f'{self.model_name} Loss History')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()

def main_worker(rank, world_size, config, train_dataset, test_dataset):
	# Initialize the process group
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)
	data_parallel_group = torch.distributed.new_group(ranks=[rank], backend="nccl")

	# Create a Trainer instance
	trainer = Trainer(config, rank, world_size, data_parallel_group=data_parallel_group, train_dataset=train_dataset, test_dataset=test_dataset)
	
	# Train the model
	trainer.train(epochs=config['epochs'])
	
	# Test the model
	#trainer.test()
	
	# Plot loss (only on master process)
	trainer.plot_loss()
	
	# Cleanup
	dist.destroy_process_group()

def setup_env_variables():
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

def main():
	world_size = 2  # Number of GPUs
	
	config = {
		'model_name': 'pretrained_ddp',
		'split_ratios': [0.7, 0.25, 0.05],
		'lr_decay_factor': 0.5,  # Fixed value instead of tune.choice
		'lr_decay_patience': 5,
		'early_stopping_patience': 25,
		'best_model_path': "best_model.pth",
		'dropout': 0.25,  # Fixed value instead of tune.choice
		'optimizer': 'adamw',  # Fixed choice
		'lr': 1e-4,  # Fixed or configurable as needed
		'batch_size': 1472,  # Fixed value
		'loss': 'mse',  # Fixed choice
		'model_size': "tiny_transformer",
		'temporal_dim': 128,
		'mask_perc': 0.25,  # Fixed choice
		'depth_dim': 96,
		'epochs': 250,  # Define the number of epochs
		'load_model': True,
		'model_path': "/home/azureuser/single_models/pretrained_ddp_val_loss_000064671_epoch_195_mse_tiny_transformer.pth"
	}
	
	setup_env_variables()
	
	torch.multiprocessing.set_sharing_strategy('file_system')
	
	#train_dataset_len = 1599976 #= np.load("/home/azureuser/data/train_dataset.npy", mmap_mode="r").shape[0]
	#test_dataset_len = 399994#np.load("/home/azureuser/data/test_dataset.npy", mmap_mode="r").shape[0]
	shared_train_dataset = "/home/azureuser/data/train_dataset.npy"
	shared_test_dataset = "/home/azureuser/data/test_dataset.npy"
	#print("numpy loading finished")
	#shared_dataset = torch.from_numpy(shared_dataset)
	#shared_train_dataset = torch.from_file("/home/azureuser/data/train_dataset.npy", dtype = torch.float32, size=train_dataset_len*config["temporal_dim"]*config["depth_dim"]*2)
	#shared_test_dataset = torch.from_file("/home/azureuser/data/test_dataset.npy", dtype = torch.float32, size=test_dataset_len*config["temporal_dim"]*config["depth_dim"]*2)
	#shared_train_dataset = shared_train_dataset.share_memory_()
	#shared_test_dataset = shared_test_dataset.share_memory_()
	#shared_dataset.reshape((shared_dataset_len, config["depth_dim"], 2))
	#print('shared_created')
	# Spawn one process per GPU
	children = []

	for i in range(world_size):
		subproc = mp.Process(target=main_worker, args=(i, world_size, config, shared_train_dataset, shared_test_dataset))
		children.append(subproc)
		subproc.start()
		print(f"Process {i} started.")

	for i in range(world_size):
		children[i].join()

	"""
	torch.multiprocessing.spawn(
		main_worker,
		args=(world_size, config, shared_train_dataset, shared_test_dataset),
		nprocs=world_size,
		join=True
	)
	"""
	
if __name__ == '__main__':
	main()
