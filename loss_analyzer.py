import matplotlib.pyplot as plt


with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_train_losses.txt", "r") as f:
    train_losses = f.readlines()

with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_val_losses.txt", "r") as f:
    val_losses = f.readlines()

train_losses = [float(loss) for loss in train_losses]
val_losses = [float(loss) for loss in val_losses]
#print(len(train_losses))
#exit()

plt.plot(range(0, len(train_losses)), train_losses, label="Train Loss")
plt.show()

plt.plot(range(0, len(val_losses)), val_losses, label="Val Loss")
plt.show()