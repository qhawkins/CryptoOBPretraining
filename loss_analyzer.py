import matplotlib.pyplot as plt
import numpy as np

azure = False

if azure:
    with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_train_losses.txt", "r") as f:
        train_losses = f.readlines()

    with open("/home/qhawkins/Downloads/pretrained_ddp_epoch_val_losses.txt", "r") as f:
        val_losses = f.readlines()

    with open("/home/qhawkins/Downloads/pretrained_ddp_lr.txt", "r") as f:
        lr = f.readlines()

else:
    with open("/media/qhawkins/SSD3/single_models/pretrained_ddp_epoch_train_losses.txt", "r") as f:
        train_losses = f.readlines()
    
    #with open("/media/qhawkins/SSD3/single_models/pretrained_ddp_epoch_val_losses.txt", "r") as f:
    #    val_losses = f.readlines()

    #with open("/media/qhawkins/SSD3/single_models/pretrained_ddp_lr.txt", "r") as f:
    #    lr = f.readlines()

#lr = [float(l) for l in lr]
#plt.plot(range(0, len(lr)), lr, label="Learning Rate")
#plt.show()
#exit()

train_losses = [float(loss) for loss in train_losses]
#val_losses = [float(loss) for loss in val_losses]

log_train_losses = [np.log(loss) for loss in train_losses]
#log_val_losses = [np.log(loss) for loss in val_losses]
#print(len(train_losses))
#exit()
most_recent = 1000000

plt.plot(range(0, len(train_losses[-most_recent:])), train_losses[-most_recent:], label="Train Loss")
plt.show()

#plt.plot(range(0, len(val_losses[-most_recent:])), val_losses[-most_recent:], label="Val Loss")
#plt.show()