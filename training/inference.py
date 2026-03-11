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
    config   = load_config()
    domain   = config['domain']
    data_dir = f'data/{domain}'
    K = 4

    node_features = torch.load(f'{data_dir}/node_features.pt')
    edge_index    = torch.load(f'{data_dir}/edge_index.pt')
    edge_features = torch.load(f'{data_dir}/edge_features.pt')
    mean          = torch.load(f'{data_dir}/mean.pt').numpy()
    std           = torch.load(f'{data_dir}/std.pt').numpy()

    test = node_features[4032:]

    node_dim = config['model']['node_dim']
    edge_dim = edge_features.shape[1]

    model = GNN(node_dim=node_dim, edge_dim=edge_dim)
    model.load_state_dict(torch.load(f'{data_dir}/model.pt', map_location='cpu'))
    model.eval()

    x = test[timestep]
    with torch.no_grad():
        delta = model(x, edge_index, edge_features)
        pred  = x + delta

    pred_denorm   = pred.numpy() * std + mean
    actual_denorm = test[timestep + 1].numpy() * std + mean

    mae_per_step = np.zeros(K)
    counts = 0
    with torch.no_grad():
        for t in range(len(test) - K):
            x = test[t]
            for k in range(K):
                delta = model(x, edge_index, edge_features)
                pred  = x + delta
                mae_per_step[k] += torch.mean(torch.abs(pred - test[t + k + 1])).item()
                x = pred
            counts += 1

    mae_per_step /= counts
    return pred_denorm, actual_denorm, mae_per_step

if __name__ == "__main__":
    pred, actual, mae_per_step = inference(timestep=0)
    std = torch.load(f'data/{load_config()["domain"]}/std.pt').numpy()

    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    for i, name in enumerate(var_names):
        mae = np.mean(np.abs(pred[:, i] - actual[:, i]))
        print(f"{name}: MAE = {mae:.4f} (normalized: {mae/std[i]:.4f})")

    print("\nRollout MAE by lead time:")
    for k in range(len(mae_per_step)):
        hours = (k + 1) * 6
        print(f"  T+{k+1} ({hours}h): {mae_per_step[k]:.6f}")