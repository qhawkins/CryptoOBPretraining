# FP8-DDP Transformer Pretraining for Financial Order Books

## Overview

This repository implements a framework for pre-training transformer models on financial order book data using NVIDIA's Transformer Engine Tensor Cores with FP8 precision. The pre-trained transformer encoder serves as the foundation for a Proximal Policy Optimization (PPO) agent designed to trade cryptocurrency markets.

The workflow consists of three main data processing steps followed by the transformer pre-training stage:

1. Order book state parsing (`data_parser.py`)
2. Feature extraction and statistics calculation (`exploration_main.py`)
3. Balanced dataset sampling and creation (`create_balanced_dataset.py`)
4. Transformer pre-training with FP8 precision (`fp8_ddp_trainer.py`)

## Why This Matters

Traditional cryptocurrency trading strategies often struggle with the high-frequency, non-stationary nature of volatilie markets. By leveraging transformer architectures that have proven effective in sequential data domains, this project provides:

1. **Efficient market state encoding:** The transformer learns meaningful representations of order book states, capturing complex patterns and relationships.
2. **Foundation for reinforcement learning:** The pre-trained encoder serves as one of the heads of the policy network in a PPO agent, encoding important state information.
3. **Computational efficiency:** FP8 precision and distributed training enable faster training while maintaining accuracy.

## System Requirements

- NVIDIA GPU with Tranformer Engine Tensor Cores (Ada Lovelace/RTX 40 series or newer)
- CUDA toolkit compatible with PyTorch
- PyTorch
- NVIDIA Transformer Engine
- NumPy, Pandas, Matplotlib, Numba
- TensorboardX for logging

**Important:** This code will not run on older NVIDIA GPUs or AMD GPUs as it requires specific hardware support for FP8 precision.

**I ran this code on an Azure 4xA100SXM VM with good results, but Ampere generation GPUs do not support Transformer Engine acceleration so this code will not work on Azure. However, I have still included the original Azure logic inside of FP8 DDP trainer to show how I originally implemented it.**

## Attached Data

Relevant cryptocurrency data has been attached to this repo under the **training_data** directory. I have not included the original raw data because each asset pair has between 40-100**GB** of raw data. I have attached a small subset of the original data for use with the data parsing pipeline.

## Data Processing Pipeline

### 1. Order Book Parsing (`data_parser.py`)

Transforms raw market data into complete order book states with specified depth.

### 2. Statistical Feature Extraction (`exploration_main.py`)

Calculates critical time-series statistics for each order book state.

### 3. Balanced Dataset Creation (`create_balanced_dataset.py`)

Ensures a balanced representation of market conditions through stratified sampling based on calculated statistics.

## Transformer Pre-Training

### Masking-Based Training Approach

The transformer model is trained using a masked prediction approach similar to BERT:

```python
# From fp8_ddp_trainer.py
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
    # Generate a mask for specified percentage of entries
    mask = torch.rand(inputs.shape, device=device, requires_grad=False, dtype=torch.float32) < mask_percentage
    
    # Replace masked entries in inputs with mask_value
    masked_inputs = inputs.clone()
    masked_inputs[mask] = mask_value
    
    return masked_inputs, mask
```

### FP8 Precision & Distributed Training

The trainer dynamically supports both single and multi-GPU training with FP8 precision.

### Automatic Training Mode Selection

The code automatically determines whether to use DDP based on available GPUs:

```python
# From fp8_ddp_trainer.py
# Detect available GPUs
num_gpus = torch.cuda.device_count()
world_size = min(num_gpus, 2)  # Use at most 2 GPUs (as per original code)

# Use single GPU if only one is available, otherwise use DDP
if world_size <= 1:
    print("Single GPU detected. Using single GPU training...")
    single_gpu_training(config, shared_train_dataset, shared_test_dataset)
else:
    print(f"Multiple GPUs detected ({world_size}). Using DDP for multi-GPU training...")
    # Set up for multi-GPU training
    mp.set_start_method('spawn')

```

## Performance Results

### Training Metrics

Results from a 10-epoch training run (very brief, just showing the capabilities of the architecture):

| Epoch | Duration (s) | Train Loss | Val Loss  | Learning Rate |
|-------|--------------|------------|-----------|---------------|
| 1     | 44.72        | 0.084256   | 0.052023  | 1.41e-05      |
| 2     | 41.29        | 0.040036   | 0.035269  | 2.60e-05      |
| 3     | 42.00        | 0.030934   | 0.028887  | 4.49e-05      |
| 4     | 41.66        | 0.027861   | 0.027467  | 6.97e-05      |
| 5     | 41.63        | 0.026084   | 0.024786  | 9.85e-05      |
| 6     | 40.61        | 0.024683   | 0.023994  | 1.29e-04      |
| 7     | 41.54        | 0.025567   | 0.022346  | 1.60e-04      |
| 8     | 41.53        | 0.020492   | 0.018917  | 1.89e-04      |
| 9     | 40.89        | 0.043318   | 0.058942  | 2.14e-04      |
| 10    | 41.29        | 0.031816   | 0.023543  | 2.33e-04      |

**In the past I ran the training with the full dataset and 2 Nvidia RTX 4090s for around a week to get substantially better results, which generalized well out of sample.**  

## Usage

### Configuration

The training parameters can be modified in the `config` dictionary within `fp8_ddp_trainer.py`:

```python
config = {
    'azure': False,                     # Whether running on Azure
    'model_name': 'pretrained_ddp',     # Name for saving models
    'split_ratios': [0.7, 0.25, 0.05],  # Train/val/test split
    'lr_decay_factor': 0.5,             # Learning rate decay
    'lr_decay_patience': 5,             # Epochs before decay
    'early_stopping_patience': 15,      # Epochs before early stop
    'dropout': 0.0,                     # Dropout rate
    'optimizer': 'adamw',               # Optimizer type
    'lr': 1e-4,                         # Base learning rate
    'batch_size': 48,                   # Batch size
    'loss': 'mse',                      # Loss function
    'model_size': 'deep_narrow_transformer', # Model architecture
    'temporal_dim': 256,                # Sequence length
    'mask_perc': 0.25,                  # Masking percentage
    'depth_dim': 96,                    # Order book depth
    'epochs': 10,                       # Training epochs
    'max_lr': 2.5e-4,                   # Max learning rate for scheduler
    'accumulation_steps': 4,            # Gradient accumulation steps
    'max_grad_norm': 1.5,               # Gradient clipping
    'use_scheduler': True               # Whether to use LR scheduler
}
```

### Running the Pipeline

1. Parse order book data:
   ```
   python data_parser.py
   ```

2. Extract statistics:
   ```
   python exploration_main.py
   ```

3. Create balanced dataset:
   ```
   python create_balanced_dataset.py
   ```

4. Train the transformer model:
   ```
   python fp8_ddp_trainer.py
   ```

## Next Steps

After pre-training, the transformer encoder can be used as the foundation for a RL-based trading agent, where:

1. The encoder provides state representations for the policy network
2. The policy network is trained using PPO to maximize trading returns
3. The agent adapts to changing market conditions through continuous learning

## Additional Files

**This repo contains other files that arent directly used by the FP8 DDP Trainer. These files include:**

### Training Results Analysis

This file allows for analysis of the training results to fine-tune your training approach and hyperparameters. 

### Inference Testing

This file allows you to perform testing on the inference performance and accuracy of the trained models to verify the effectiveness of training.

### Model Exporter

This file contains logic to export the model to TorchScript for inference or further training. I implemented this functionality to allow for the pre-trained model to be implemented into my C++ accelerated RL training environment. 

**Enjoy the code and feel free to reach out with any questions!!!**