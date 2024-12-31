import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
from collections import deque
import numpy as np
from ray import train, tune
from training_classes import CombinedNanPlateauStopper, PretrainingDataset
import ray
from models import SmallFCModel, MediumFCModel, LargeFCModel, ShallowLSTMModel, DeepLSTMModel, TinyLSTMModel
from torch.masked import MaskedTensor

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
        labels (torch.Tensor): Tensor with original values for masked entries and -100 elsewhere.
        mask (torch.Tensor): Boolean mask indicating which entries were masked.
    """
    # Generate a mask for 15% of the entries
    mask = torch.rand(inputs.shape, device=device) < mask_percentage

    # Replace masked entries in inputs with mask_value
    masked_inputs = inputs.clone()
    masked_inputs[mask] = mask_value

    return masked_inputs.cuda(), mask

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dropout = self.config['dropout']
        self.temporal_dim = self.config['temporal_dim']
        self.mask_perc = self.config['mask_perc']
        self.depth_dim = self.config['depth_dim']
        
        if self.config['model_size'] == 'small':
            self.model = SmallFCModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')

        elif self.config['model_size'] == 'medium':
            self.model = MediumFCModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')

        elif self.config['model_size'] == 'large':
            self.model = LargeFCModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')

        elif self.config['model_size'] == 'shallow_lstm':
            self.model = ShallowLSTMModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')

        elif self.config['model_size'] == 'deep_lstm':
            self.model = DeepLSTMModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')
            
        elif self.config['model_size'] == 'tiny_lstm':
            self.model = TinyLSTMModel((self.temporal_dim, self.depth_dim, 2), (self.temporal_dim, self.depth_dim, 2), self.dropout).to('cuda')

        #compiling
        self.model = torch.compile(self.model)

        if config['loss'] == 'mse':
            self.criterion = torch.nn.MSELoss().to('cuda')
        
        elif config['loss'] == 'mae':
            self.criterion = torch.nn.L1Loss().to('cuda')

        elif config['loss'] == 'huber':
            self.criterion = torch.nn.HuberLoss().to('cuda')

        elif config['loss'] == 'smoothl1':
            self.criterion = torch.nn.SmoothL1Loss().to('cuda')

        if self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])

        self.data = np.load("/home/qhawkins/Desktop/CryptoOBPretraining/test.npy")
        self.model_name = self.config["model_name"]
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.ratios = self.config['split_ratios']
        self.lr_decay_factor = self.config['lr_decay_factor']
        self.lr_decay_patience = self.config['lr_decay_patience']
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_decay_factor, patience=self.lr_decay_patience)
        self.scaler = torch.amp.GradScaler()
        self.early_stopping_patience = self.config['early_stopping_patience']
        self.saved_models = deque(maxlen=self.early_stopping_patience+1)
        self.best_val_loss = float('inf')
        self.best_model_path = self.config['best_model_path']        

    def save_model(self, epoch, val_loss):
        model_path = f"/media/qhawkins/SSD3/ray_models/{self.model_name}_val_loss_{str(round(val_loss, 8)).replace('.', '')}_epoch_{epoch}_{self.config['loss']}_{self.config['model_size']}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'loss_function': self.config['loss']
        }, model_path)
        self.saved_models.append((model_path, val_loss))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_path = model_path

    def split_data(self):
        total_size = len(self.data)
        train_size = int(self.ratios[0] * total_size)
        val_size = int(self.ratios[1] * total_size)
        test_size = total_size - train_size - val_size

        #self.train_ds, self.val_ds, self.test_ds = random_split(
        #    self.dataset, [train_size, val_size]
        #)
        self.train_ds = PretrainingDataset(self.data[:train_size], self.temporal_dim)
        self.val_ds = PretrainingDataset(self.data[train_size:train_size+val_size], self.temporal_dim)
        self.test_ds = PretrainingDataset(self.data[:test_size], self.temporal_dim)

        self.train_dataloader = DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True, num_workers=6, prefetch_factor=4, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_ds, batch_size=self.config['batch_size'], shuffle=False, drop_last=True, num_workers=6, prefetch_factor=4, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_ds, batch_size=self.config['batch_size'], shuffle=False, drop_last=True, num_workers=6, prefetch_factor=4, pin_memory=True)

    def print_model_params(self):
        print(f"Model: {self.model_name}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
    def train(self, epochs):
        self.split_data()
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            avg_train_loss = 0
            avg_val_loss = 0

            for i, data in enumerate(self.train_dataloader):
                # Assuming data shape: (batch_size, seq_length, features)
                # No need for separate labels in MLM; labels are derived from inputs
                #print(f"Initial data nan count: {torch.isnan(data).sum()}")
                masked_inputs, mask = apply_mask(
                    data,
                    mask_percentage=self.mask_perc,
                    mask_value=0.0,  # You can choose a different mask value if needed
                    device='cuda'
                )
                #print(f"masked inputs nan count: {torch.isnan(masked_inputs).sum()}")

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    data = data.cuda()
                    #print(50*"=-")
                    #print(data.shape)
                    outputs = self.model(masked_inputs)  # Shape: (batch_size, seq_length -1, features)
                    # Compute loss only on masked positions
                    #print(f"Outputs shape: {outputs.shape}, mask shape: {mask.shape}, data shape: {data.shape}")
                    loss = self.criterion(outputs[~mask], data[~mask])
                    #print(f"Loss: {loss.item()}, outputs shape: {outputs.shape}, data shape: {data.shape}, mask shape: {mask.shape}, outputs nan count: {torch.isnan(outputs).sum()}, data nan count: {torch.isnan(data).sum()}, mask nan count: {torch.isnan(mask).sum()}")
                    # For tracking purposes, compute tracking loss as before

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                avg_train_loss += loss.item()

            avg_train_loss /= (i + 1)

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.val_dataloader):
                    data = data.cuda()
                    masked_inputs, mask = apply_mask(
                        data,
                        mask_percentage=self.mask_perc,
                        mask_value=0.0,
                        device='cuda'
                    )

                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(masked_inputs)
                        loss = self.criterion(outputs[~mask], data[~mask])

                    self.val_loss_history.append(loss.item())
                    avg_val_loss += loss.item()
                avg_val_loss /= (i + 1)

            # Reporting
            train.report({'loss': avg_val_loss, "iteration": epoch, "train_loss": avg_train_loss})

            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                if avg_val_loss < 0.45:
                    self.save_model(epoch, avg_val_loss)
            else:
                epochs_without_improvement += 1

            # Early Stopping Logic (if applicable)
            if epochs_without_improvement >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

            print(f'Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}')

        print(f"Training completed. Best model saved at: {self.best_model_path}")

    def test(self):
        if self.best_model_path:
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} for testing.")

        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                inputs = data[:, :-1].to('cuda')
                labels = data[:, -1].to('cuda')

                with torch.amp.autocast(device_type='cuda'):
                    outputs = torch.flatten(self.model(inputs))
                    loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
        avg_test_loss = test_loss / (i + 1)
        #print(f'{self.model_name} Test Loss {config['loss']}: {avg_test_loss:.6f}')
        test_loss = 0
        mae_loss = torch.nn.L1Loss().to('cuda')
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                inputs = data[:, :-1].to('cuda')
                labels = data[:, -1].to('cuda')

                with torch.amp.autocast(device_type='cuda'):
                    outputs = torch.flatten(self.model(inputs))
                    loss = mae_loss(outputs, labels)
                
                test_loss += loss.item()
        avg_test_loss = test_loss / (i + 1)
        print(f'{self.model_name} Test Loss (MAE): {avg_test_loss:.6f}')



    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.title(f'{self.model_name} Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def train_model(config):
    model = Trainer(config)
    model.train(1000)
    model.test()
    return model.best_val_loss

if __name__ == '__main__':

    ray.init()
    combined_stopper = CombinedNanPlateauStopper(metric="loss", std=0.05, num_results=10, grace_period=25, metric_threshold=.5, mode='min')

    config = {
        'model_name': 'pretrained1',
        'split_ratios': [0.7, 0.25, 0.05],
        'lr_decay_factor': tune.choice([0.5, 0.75, 0.95]),
        'lr_decay_patience': 5,
        'early_stopping_patience': 25,
        'best_model_path': "best_model.pth",
        'dropout': tune.choice([0.1, 0.15, 0.25, 0.5]),
        'optimizer': tune.choice(['adam', 'adamw']),
        'lr': tune.uniform(1e-6, 1e-3),
        'batch_size': tune.choice([2048, 4096]),
        'loss': tune.choice(['mse']),#,'mae', 'huber']),
        "model_size": tune.choice(["shallow_lstm", "deep_lstm", "tiny_lstm", "small", "medium", "large"]),
        'temporal_dim': 128,
        'mask_perc': tune.choice([0.15, 0.25, 0.35]),
        'depth_dim': 64
    }

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources=tune.PlacementGroupFactory([{'CPU': 7, 'GPU': 1}] + [{'CPU': 1.0}])
        ),
        run_config=train.RunConfig(
            stop=combined_stopper
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            #scheduler=scheduler,
            num_samples=1000,
            #reuse_actors=True
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min", "all")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")