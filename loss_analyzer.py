import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os

# Configuration
azure = False
#most_recent = 5000  # Adjust this value based on your needs
most_recent = 250000
# File paths
if azure:
    loss_path = "/home/qhawkins/Downloads/pretrained_ddp_epoch_train_losses.txt"
    #loss_path = "/home/qhawkins/Downloads/pretrained_ddp_epoch_val_losses.txt"
    # lr_path = "/home/qhawkins/Downloads/pretrained_ddp_lr.txt"
else:
    loss_path = "/media/qhawkins/SSD3/single_models/pretrained_ddp_epoch_train_losses.txt"
    #loss_path = "/media/qhawkins/SSD3/single_models/pretrained_ddp_epoch_val_losses.txt"
    # lr_path = "/media/qhawkins/SSD3/single_models/pretrained_ddp_lr.txt"

def read_train_losses():
    """
    Reads the training losses from the specified file.
    Returns a list of floats representing the loss values.
    """
    if not os.path.exists(loss_path):
        print(f"Train loss file not found: {loss_path}")
        return []
    
    try:
        with open(loss_path, "r") as f:
            lines = f.readlines()
        # Convert lines to floats, ignoring any malformed lines
        train_losses = [float(line.strip()) for line in lines if line.strip()]
        return train_losses
    except Exception as e:
        print(f"Error reading train loss file: {e}")
        return []

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Train Loss", color='blue')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Real-Time Training Loss")
ax.legend()
ax.grid(True)

def init():
    """Initialize the plot limits and line data."""
    ax.set_xlim(0, most_recent)
    ax.set_ylim(0, 1)  # Initial y-axis limit; will be autoscaled
    line.set_data([], [])
    return line,

def update(frame):
    """Update the plot with new loss data."""
    train_losses = read_train_losses()
    if not train_losses:
        return line,

    # If you want to display only the most recent N points
    if len(train_losses) > most_recent:
        train_losses_display = train_losses[-most_recent:]
        x = range(len(train_losses_display))
    else:
        train_losses_display = train_losses
        x = range(len(train_losses_display))
    
    # Update the data of the plot
    line.set_data(x, train_losses_display)
    
    # Adjust the x-axis limit if necessary
    ax.set_xlim(max(0, len(train_losses_display) - most_recent), len(train_losses_display))
    
    # Adjust the y-axis to fit the data
    ax.set_ylim(min(train_losses_display) * 0.95, max(train_losses_display) * 1.05)
    
    return line,

# Create the animation
ani = FuncAnimation(
    fig, 
    update, 
    init_func=init, 
    interval=1000,    # Update every 1000 milliseconds (1 second)
    blit=True
)

plt.show()
