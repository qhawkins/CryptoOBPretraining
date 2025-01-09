import matplotlib.pyplot as plt
import numpy as np

with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_train_losses.txt", "r") as f:
    train_losses = f.readlines()

with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_val_losses.txt", "r") as f:
    val_losses = f.readlines()

with open("/home/qhawkins/Downloads/pretrained_ddp_lr.txt", "r") as f:
    lr = f.readlines()


#lr = [float(l) for l in lr]
#plt.plot(range(0, len(lr)), lr, label="Learning Rate")
#plt.show()
#exit()

train_losses = [float(loss) for loss in train_losses]
val_losses = [float(loss) for loss in val_losses]

log_train_losses = [np.log(loss) for loss in train_losses]
log_val_losses = [np.log(loss) for loss in val_losses]
#print(len(train_losses))
#exit()

plt.plot(range(0, len(train_losses[40000:])), train_losses[40000:], label="Train Loss")
plt.show()

plt.plot(range(0, len(val_losses[10000:])), val_losses[10000:], label="Val Loss")
plt.show()