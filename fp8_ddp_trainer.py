import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import time

from training_classes import PretrainingDataset
from fp8_models import TinyTransformerModel, MediumTransformerModel, DeepNarrowTransformerModel

from transformer_engine.common.recipe import Format, DelayedScaling
import transformer_engine.pytorch as te

# allow tf32 format for increased performance on modern Nvidia GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# tensorboard for logging and diagnostics
from tensorboardX import SummaryWriter


# Define your apply_mask function
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
	mask = torch.rand(inputs.shape, device=device, requires_grad=False, dtype=torch.float32) < mask_percentage
	#mask = mask.to(device, non_blocking=True)
	
	# Replace masked entries in inputs with mask_value
	masked_inputs = inputs.clone()
	masked_inputs[mask] = mask_value
	
	return masked_inputs, mask

# Adapted Trainer class with adaptive DDP support for both single and multi-GPU setups
class Trainer:
	def __init__(self, config, rank, world_size, train_dataset: tuple, test_dataset: tuple, dp_group=None, use_ddp=True):
		"""
		Initialize trainer with support for both single and multi-GPU setups.
		
		Args:
			config: Configuration dictionary
			rank: GPU rank (0 for single GPU)
			world_size: Number of GPUs (1 for single GPU)
			train_dataset: Training dataset
			test_dataset: Testing dataset
			dp_group: Data parallel group (None for single GPU)
			use_ddp: Whether to use DDP (False for single GPU)
		"""
		self.config = config
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.rank = rank
		self.world_size = world_size
		self.dp_group = dp_group
		self.use_ddp = use_ddp
		self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		
		# Initialize the model
		self.initialize_model()
		
		# Initialize loss function and optimizer
		self.initialize_criterion_optimizer()
		
		# Load data
		self.load_data()
		
		# Initialize other components
		self.initialize_training_components()
		self.train_losses = []
		self.val_losses = []
		
		fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
		self.recipe = DelayedScaling(fp8_format=fp8_format)

		if rank == 0:
			self.logger = SummaryWriter()
	
	def log_gradients_in_model(self, model, logger, step):
		for tag, value in self.model.named_parameters():
			if value.grad is not None:
				self.logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

	def load_model(self, path: str):
		self.model = MediumTransformerModel((self.config["temporal_dim"], self.config["depth_dim"], 2), (self.config["temporal_dim"], self.config["depth_dim"], 2), self.config['dropout'])
		state_dict = torch.load(path)
		state_dict = state_dict['model_state_dict']
		# Handle loading state dict for both DDP and non-DDP models
		state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
		self.model.load_state_dict(state_dict)
		self.model = self.model.to(self.device)
		self.model = self.model.train()

	def initialize_model(self):
		model_size = self.config['model_size']
		temporal_dim = self.config['temporal_dim']
		depth_dim = self.config['depth_dim']
		dropout = self.config['dropout']
		
		model_classes = {
			'tiny_transformer': TinyTransformerModel,
			'medium_transformer': MediumTransformerModel,
			'deep_narrow_transformer': DeepNarrowTransformerModel
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
				dropout,
			).to(self.device)
			
		# Wrap the model with DDP only if using multiple GPUs
		if self.use_ddp and self.world_size > 1:
			self.model = DDP(self.model, device_ids=[self.rank])
		
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
		# Create dataset splits
		self.split_data()
		
		# Use DistributedSampler for multi-GPU, regular DataLoader for single-GPU
		if self.use_ddp and self.world_size > 1:
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
				num_workers=5,
				pin_memory=True,
				prefetch_factor=4,
			)
			
			self.val_dataloader = DataLoader(
				self.val_ds,
				batch_size=self.config['batch_size'],
				sampler=self.val_sampler,
				drop_last=True,
				num_workers=4,
				pin_memory=True,
			)
			
			self.test_dataloader = DataLoader(
				self.test_ds,
				batch_size=self.config['batch_size'],
				sampler=self.test_sampler,
				drop_last=True,
				num_workers=4,
				pin_memory=True,
			)
		else:
			# For single GPU setup, use regular DataLoader with shuffle instead of sampler
			self.train_dataloader = DataLoader(
				self.train_ds,
				batch_size=self.config['batch_size'],
				shuffle=True,
				drop_last=True,
				num_workers=5,
				pin_memory=True,
				prefetch_factor=4,
			)
			
			self.val_dataloader = DataLoader(
				self.val_ds,
				batch_size=self.config['batch_size'],
				shuffle=False,
				drop_last=True,
				num_workers=4,
				pin_memory=True,
			)
			
			self.test_dataloader = DataLoader(
				self.test_ds,
				batch_size=self.config['batch_size'],
				shuffle=False,
				drop_last=True,
				num_workers=4,
				pin_memory=True,
			)
		
	def split_data(self):
		train_sizes = (
			int(np.load(self.train_dataset[0], mmap_mode="r").shape[0]*self.config['split_ratios'][0]),
			int(np.load(self.train_dataset[1], mmap_mode="r").shape[0]*self.config['split_ratios'][0])
		)
		test_sizes = (
			int(np.load(self.test_dataset[0], mmap_mode="r").shape[0]),
			int(np.load(self.test_dataset[1], mmap_mode="r").shape[0])
		)
		val_sizes = (
			int((np.load(self.train_dataset[0], mmap_mode="r").shape[0] * self.config['split_ratios'][1]) + train_sizes[0]),
			int((np.load(self.train_dataset[1], mmap_mode="r").shape[0] * self.config['split_ratios'][1]) + train_sizes[1])
		)

		self.train_ds = PretrainingDataset(
			self.train_dataset,
			(0, 0), 
			train_sizes, 
			self.config['temporal_dim'],
			azure=self.config['azure']
		)
		self.val_ds = PretrainingDataset(
			self.train_dataset,
			train_sizes,
			val_sizes,
			self.config['temporal_dim'],
			azure=self.config['azure']

		)
		self.test_ds = PretrainingDataset(
			self.test_dataset,
			(0, 0), 
			test_sizes, 
			self.config['temporal_dim'],
			azure=self.config['azure']
		)
		
	def initialize_training_components(self):
		self.model_name = self.config["model_name"]
		self.train_loss_history = []
		self.val_loss_history = []
		self.test_loss_history = []
		self.ratios = self.config['split_ratios']
		self.lr_decay_factor = self.config['lr_decay_factor']
		self.lr_decay_patience = self.config['lr_decay_patience']
		if self.config['use_scheduler']:
			self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
				self.optimizer,
				max_lr=self.config["max_lr"],
				epochs=self.config["epochs"],
				steps_per_epoch=len(self.train_dataloader)
			)
		self.early_stopping_patience = self.config['early_stopping_patience']
		self.saved_models = deque(maxlen=self.early_stopping_patience + 1)
		self.best_val_loss = float('inf')
		self.best_model_path = self.config['best_model_path']
		self.step_losses = []
		
	def save_model(self, epoch, val_loss):
		if self.rank != 0:
			return  # Only the master process saves the model
		if self.config['azure']:
			model_path = (
				f"/home/azureuser/single_models/{self.model_name}_val_loss_"
				f"{str(round(val_loss, 8)).replace('.', '')}_epoch_{epoch}_"
				f"{self.config['loss']}_{self.config['model_size']}.pth"
			)
		else:
			model_path = (
				f"/media/qhawkins/SSD3/single_models/{self.model_name}_val_loss_"
				f"{str(round(val_loss, 8)).replace('.', '')}_epoch_{epoch}_"
				f"{self.config['loss']}_{self.config['model_size']}.pth"
			)

		torch.save({
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),  # Handle both DDP and non-DDP models
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
			print(f"Using DDP: {self.use_ddp}, Number of GPUs: {self.world_size}")
	
	def train(self, epochs):
		self.print_model_params()
		best_val_loss = float('inf')
		epochs_without_improvement = 0
		if self.rank == 0:
			print(f"Start time for training: {time.ctime()}")
		
		for epoch in range(epochs):
			epoch_start_time = time.time()
			# Set epoch for DistributedSampler (only in DDP mode)
			if self.use_ddp and self.world_size > 1:
				self.train_sampler.set_epoch(epoch)
				
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
				
				# Handle gradient accumulation differently for DDP vs non-DDP
				if self.use_ddp and self.world_size > 1:
					# DDP mode with gradient accumulation
					if (i + 1) % self.config["accumulation_steps"] != 0:
						with self.model.no_sync():				
							with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
								outputs = self.model(masked_inputs)
								loss: torch.Tensor = self.criterion(outputs[mask], data[mask])
							loss.backward()
					else:
						with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
							outputs = self.model(masked_inputs)
							loss: torch.Tensor = self.criterion(outputs[mask], data[mask])
						loss.backward()
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
						self.optimizer.step()
						if self.config['use_scheduler']:
							self.scheduler.step()
						self.optimizer.zero_grad(set_to_none=True)
				else:
					# Non-DDP mode with gradient accumulation (simpler)
					with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
						outputs = self.model(masked_inputs)
						loss: torch.Tensor = self.criterion(outputs[mask], data[mask])
						loss = loss / self.config["accumulation_steps"]
					
					loss.backward()
					
					if (i + 1) % self.config["accumulation_steps"] == 0:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
						self.optimizer.step()
						if self.config['use_scheduler']:
							self.scheduler.step()
						self.optimizer.zero_grad(set_to_none=True)

				if self.rank == 0 and (i+1) % 1000 == 0:
					self.log_gradients_in_model(self.model, self.logger, i)

				# Save loss regardless of GPU count
				loss_val = loss.item() * (self.config["accumulation_steps"] if not self.use_ddp else 1)
				if self.config['azure']:
					with open(f"/home/azureuser/single_models/{self.model_name}_epoch_train_losses.txt", "a+") as f:
						f.write(f"{loss_val}\n")
				else:
					with open(f"/media/qhawkins/SSD3/single_models/{self.model_name}_epoch_train_losses.txt", "a+") as f:
						f.write(f"{loss_val}\n")

				avg_train_loss += loss_val
				self.step_losses.append(loss_val)
			
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

					with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
						outputs = self.model(masked_inputs)
						loss = self.criterion(outputs[mask], data[mask])

					if self.config['azure']:
						with open(f"/home/azureuser/single_models/{self.model_name}_epoch_val_losses.txt", "a+") as f:
							f.write(f"{loss.item()}\n")
					else:
						with open(f"/media/qhawkins/SSD3/single_models/{self.model_name}_epoch_val_losses.txt", "a+") as f:
							f.write(f"{loss.item()}\n")

					avg_val_loss += loss.item()
				avg_val_loss /= (i + 1)
			
			# Only the master process logs and saves models
			epoch_end_time = time.time()
			if self.rank == 0:
				self.train_loss_history.append(avg_train_loss)
				self.val_loss_history.append(avg_val_loss)
				if self.config['azure']:
					with open(f"/home/azureuser/single_models/{self.model_name}_train_losses.txt", "a+") as f:
						f.write(f"{avg_train_loss}\n")
					with open(f"/home/azureuser/single_models/{self.model_name}_val_losses.txt", "a+") as f:
						f.write(f"{avg_val_loss}\n")
					if self.config['use_scheduler']:
						with open(f"/home/azureuser/single_models/{self.model_name}_lr.txt", "a+") as f:
							f.write(f"{self.scheduler.get_last_lr()[0]}\n")
				else:
					with open(f"/media/qhawkins/SSD3/single_models/{self.model_name}_train_losses.txt", "a+") as f:
						f.write(f"{avg_train_loss}\n")
					with open(f"/media/qhawkins/SSD3/single_models/{self.model_name}_val_losses.txt", "a+") as f:
						f.write(f"{avg_val_loss}\n")
					if self.config['use_scheduler']:
						with open(f"/media/qhawkins/SSD3/single_models/{self.model_name}_lr.txt", "a+") as f:
							f.write(f"{self.scheduler.get_last_lr()[0]}\n")

				epoch_completion_time = epoch_end_time-epoch_start_time
				next_epoch_estimated_time = time.ctime(time.time() + epoch_completion_time)

				if self.config['use_scheduler']:
					print(f'Epoch {epoch+1}/{epochs} finished in {round(epoch_completion_time, 2)} seconds, ETA of next epoch is {next_epoch_estimated_time}, Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}, Epoch learning rate: {self.scheduler.get_last_lr()[0]}')
				else:		
					print(f'Epoch {epoch+1}/{epochs} finished in {round(epoch_completion_time, 2)} seconds, ETA of next epoch is {next_epoch_estimated_time}, Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}, Epoch learning rate: {self.config["lr"]}')
				
				self.save_model(epoch, avg_val_loss)
				if avg_val_loss < best_val_loss:
					best_val_loss = avg_val_loss
					epochs_without_improvement = 0
					if avg_val_loss < 0.45:
						self.save_model(epoch, avg_val_loss)
				else:
					epochs_without_improvement += 1
		
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
				with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
					outputs = self.model(masked_inputs)
					loss = self.criterion(outputs[mask], data[mask])
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
				
				with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe, fp8_group=self.dp_group):
					outputs = self.model(masked_inputs)
					loss = mae_loss(outputs[mask], data[mask])
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

def single_gpu_training(config, train_dataset, test_dataset):
    """
    Training workflow for single GPU setup.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        test_dataset: Testing dataset
    """
    # Set device to cuda:0 for single GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize trainer with single GPU setup (no DDP)
    trainer = Trainer(
        config=config,
        rank=0,
        world_size=1,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dp_group=None,
        use_ddp=False
    )
    
    # Train the model
    trainer.train(epochs=config['epochs'])
    
    # Test the model
    trainer.test()
    
    # Plot loss
    trainer.plot_loss()

def main_worker(rank, world_size, config, train_dataset, test_dataset):
    """
    Worker function for multi-GPU distributed training.
    
    Args:
        rank: GPU rank
        world_size: Number of GPUs
        config: Configuration dictionary
        train_dataset: Training dataset
        test_dataset: Testing dataset
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5000"
    
    # Initialize the process group
    dist.init_process_group(
        backend=config['backend'],
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Create a data parallel group for FP8
    data_parallel_group = torch.distributed.new_group(ranks=[rank], backend=config['backend'])
    
    # Set device
    torch.cuda.set_device(rank)

    # Initialize DDP trainer
    trainer = Trainer(
        config=config, 
        rank=rank, 
        world_size=world_size, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        dp_group=data_parallel_group,
        use_ddp=True
    )
    
    # Train the model
    trainer.train(epochs=config['epochs'])
    
    # Test the model (only on master process)
    if rank == 0:
        trainer.test()
        trainer.plot_loss()
    
    # Cleanup
    dist.destroy_process_group()

def main():
    """
    Main entry point for training.
    Determines whether to use single or multi-GPU setup based on available GPUs.
    """
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    world_size = min(num_gpus, 2)  # Use at most 2 GPUs (as per original code)
    
    config = {
        'azure': False,
        'model_name': 'pretrained_ddp',
        'split_ratios': [0.7, 0.25, 0.05],
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 5,
        'early_stopping_patience': 15,
        'best_model_path': "best_model.pth",
        'dropout': 0.0,
        'optimizer': 'adamw',
        'lr': 1e-4,
		#reduced batch size for explanatory purposes
        'batch_size': 48, #used to be 96
        'loss': 'mse',
        "model_size": "deep_narrow_transformer",
        'temporal_dim': 256,
        'mask_perc': 0.25,
        'depth_dim': 96,
        'epochs': 10,
        'load_model': False,
        'model_path': "/media/qhawkins/SSD3/single_models/pretrained_ddp_val_loss_000135047_epoch_5_mse_medium_transformer.pth",
        'max_lr': 2.5e-4,
        "backend": "nccl",
        "accumulation_steps": 4,
        "max_grad_norm": 1.5,
        "use_scheduler": True
    }
    
    # Define datasets
    if config["azure"]:
        shared_train_dataset = "/home/azureuser/datadrive/train_indices.npy"
        shared_test_dataset = "/home/azureuser/datadrive/test_indices.npy"
    else:
        shared_train_dataset = (
            "/home/qhawkins/Desktop/CryptoOBPretraining/eth_btc_train_indices.npy",
            "/home/qhawkins/Desktop/CryptoOBPretraining/btc_usdt_train_indices.npy"
        )
        shared_test_dataset = (
            "/home/qhawkins/Desktop/CryptoOBPretraining/eth_btc_test_indices.npy",
            "/home/qhawkins/Desktop/CryptoOBPretraining/btc_usdt_test_indices.npy"
        )
    
    # Use single GPU if only one is available, otherwise use DDP
    if world_size <= 1:
        print("Single GPU detected. Using single GPU training...")
        single_gpu_training(config, shared_train_dataset, shared_test_dataset)
    else:
        print(f"Multiple GPUs detected ({world_size}). Using DDP for multi-GPU training...")
        # Set up for multi-GPU training
        mp.set_start_method('spawn')
        
        # Spawn one process per GPU
        children = []
        for i in range(world_size):
            subproc = mp.Process(
                target=main_worker, 
                args=(i, world_size, config, shared_train_dataset, shared_test_dataset)
            )
            children.append(subproc)
            subproc.start()
            print(f"Process {i} started.")

        for i in range(world_size):
            children[i].join()
            
if __name__ == '__main__':
    main()