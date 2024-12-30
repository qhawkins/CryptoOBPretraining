import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    dir_path = "/home/qhawkins/ray_results"
    training_runs = os.listdir(dir_path)
    sorted_runs = sorted(training_runs)
    
    selected_run = sorted_runs[-1]

    for folder in glob.glob(os.path.join(os.path.join(dir_path, selected_run), "*")):
        if not os.path.exists(os.path.join(folder, "progress.csv")):
            continue    
        progress = pd.read_csv(os.path.join(folder, "progress.csv"))
        parameters = json.load(open(os.path.join(folder, "params.json")))
        batch_size = parameters.get("batch_size", 0)
        depth_dim = parameters.get("depth_dim", 0)
        dropout = parameters.get("dropout", 0)
        loss_fn = parameters.get("loss", "")
        lr = parameters.get("lr", 0)
        lr_decay = parameters.get("lr_decay", 0)
        mask_percentage = parameters.get("mask_perc", 0)
        model_size = parameters.get("model_size", "")
        optimizer = parameters.get("optimizer", "")
        temporal_dim = parameters.get("temporal_dim", 0)

        if progress.iloc[-1]["loss"] > .5:
            continue

        if batch_size == 2048:
            plt.plot(progress["training_iteration"], progress["loss"], "red")
        elif batch_size == 4096:
            plt.plot(progress["training_iteration"], progress["loss"], "blue")

        print(progress.info())        

    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.show()