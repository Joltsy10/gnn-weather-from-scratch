import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gnn import GNN
import yaml

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)
    
def inference(timestep=0):
    config = load_config()
    K = 4

    node_features = torch.tensor(
        np.load('data/node_features.npy'), dtype=torch.float32
    )
    edge_index = torch.tensor(
        np.load('data/edge_index.npy'), dtype=torch.long
    )
    edge_features = torch.tensor(
        np.load('data/edge_features.npy'), dtype=torch.float32
    )
    test = node_features[4032:]
    mean = np.load('data/mean.npy')
    std = np.load('data/std.npy')

    model = GNN(node_dim=7, edge_dim=3)
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model.eval()

    x = test[timestep]
    with torch.no_grad():
        delta = model(x, edge_index, edge_features)
        pred = x + delta

    pred_denorm = pred.numpy() * std + mean
    actual_denorm = test[timestep + 1].numpy() * std + mean

    mae_per_step = np.zeros(K)
    counts = 0
    with torch.no_grad():
        for t in range(len(test) - K):
            x = test[t]
            for k in range(K):
                delta = model(x, edge_index, edge_features)
                pred = x + delta
                mae_per_step[k] += torch.mean(torch.abs(pred - test[t + k + 1])).item()
                x = pred
            counts += 1

    mae_per_step /= counts
    return pred_denorm, actual_denorm, mae_per_step

if __name__ == "__main__":
    pred, actual, mae_per_step = inference(timestep=0)
    std = np.load('data/std.npy')

    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    for i, name in enumerate(var_names):
        mae = np.mean(np.abs(pred[:, i] - actual[:, i]))
        print(f"{name}: MAE = {mae:.4f} (normalized: {mae/std[i]:.4f})")

    print("\nRollout MAE by lead time:")
    for k in range(len(mae_per_step)):
        hours = (k + 1) * 6
        print(f"  T+{k+1} ({hours}h): {mae_per_step[k]:.6f}")