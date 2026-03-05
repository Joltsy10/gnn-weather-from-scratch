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
        pred = model(x, edge_index, edge_features)

    pred_denorm = pred.numpy() * std + mean
    actual_denorm = test[timestep + 1].numpy() * std + mean

    loss_fn = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for t in range(len(test)-1):
            p = model(test[t], edge_index, edge_features)
            total += loss_fn(p, test[t+1]).item()
    test_mse = total / (len(test)-1)

    return pred_denorm, actual_denorm, test_mse

if __name__ == "__main__":
    pred, actual, test_mse = inference(timestep=0)
    std = np.load('data/std.npy')

    print(f"Predicted shape: {pred.shape}")
    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    for i, name in enumerate(var_names):
        mae = np.mean(np.abs(pred[:, i] - actual[:, i]))
        print(f"{name}: MAE = {mae:.4f} (normalized: {mae/std[i]:.4f})")

    print(f"\nTest MSE:         {test_mse:.6f}")
    print(f"Test persistence: 0.101400")