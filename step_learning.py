import torch
import matplotlib.pyplot as plt

base_path = "/media/qhawkins/SSD3/ray_models/"
model_id = "pretrained1_val_loss_015957976_epoch_0_mse_tiny_transformer.pth"
model_path = base_path + model_id
state_dict = torch.load(model_path)
step_losses = state_dict['step_losses']

plt.plot(step_losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.show()