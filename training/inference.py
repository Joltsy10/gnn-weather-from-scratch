import torch
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
    mean = np.load('data/mean.npy')
    std = np.load('data/std.npy')

    model = GNN(
        node_dim=7,
        edge_dim=3,
    )
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model.eval()

    x = node_features[timestep]
    with torch.no_grad():
        pred = model(x, edge_index, edge_features)

    pred_denorm = pred.numpy() * std + mean
    actual_denorm = node_features[timestep + 1].numpy() * std + mean

    return pred_denorm, actual_denorm

if __name__ == "__main__":
    pred, actual = inference(timestep=0)
    print(f"Predicted shape: {pred.shape}")
    std = np.load('data/std.npy')

    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    for i, name in enumerate(var_names):
        mae = np.mean(np.abs(pred[:, i] - actual[:, i]))
        mae_normalized = mae / std[i]
        print(f"{name}: MAE = {mae:.4f} (normalized: {mae_normalized:.4f})")



