import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gnn import GNN
from model.hi_gnn import HiGNN
import yaml

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_global_graph(graph_dir, device):
    m2m_edge_index = torch.load(f'{graph_dir}/m2m_edge_index.pt', map_location=device)
    num_levels = len(m2m_edge_index)
    graph = {
        'g2m_edge_index':  torch.load(f'{graph_dir}/g2m_edge_index.pt',       map_location=device),
        'g2m_features':    torch.load(f'{graph_dir}/g2m_features.pt',          map_location=device),
        'm2g_edge_index':  torch.load(f'{graph_dir}/m2g_edge_index.pt',        map_location=device),
        'm2g_features':    torch.load(f'{graph_dir}/m2g_features.pt',          map_location=device),
        'm2m_edge_index':  m2m_edge_index,
        'm2m_features':    torch.load(f'{graph_dir}/m2m_features.pt',          map_location=device),
        'up_edge_index':   torch.load(f'{graph_dir}/mesh_up_edge_index.pt',    map_location=device),
        'up_features':     torch.load(f'{graph_dir}/mesh_up_features.pt',      map_location=device),
        'down_edge_index': torch.load(f'{graph_dir}/mesh_down_edge_index.pt',  map_location=device),
        'down_features':   torch.load(f'{graph_dir}/mesh_down_features.pt',    map_location=device),
        'mesh_features':   torch.load(f'{graph_dir}/mesh_features.pt',         map_location=device),
    }
    return graph, num_levels

def inference(timestep=0, device='cpu'):
    config   = load_config()
    domain   = config['domain']
    data_dir = f'data/{domain}'
    node_dim = config['model']['node_dim']
    K = 1

    node_features = torch.load(f'{data_dir}/node_features.pt')
    mean = torch.load(f'{data_dir}/mean.pt').numpy()
    std  = torch.load(f'{data_dir}/std.pt').numpy()

    test = node_features[4032:]

    if domain == 'lam':
        edge_index    = torch.load(f'{data_dir}/edge_index.pt').to(device)
        edge_features = torch.load(f'{data_dir}/edge_features.pt').to(device)
        edge_dim = edge_features.shape[1]
        model = GNN(node_dim=node_dim, edge_dim=edge_dim)
        model.load_state_dict(torch.load(f'{data_dir}/tmodel.pt', map_location=device))
        model = model.to(device)
    else:
        graph, num_levels = load_global_graph(data_dir, device)
        edge_dim = graph['g2m_features'].shape[1]
        model = HiGNN(node_dim=node_dim, edge_dim=edge_dim, num_levels=num_levels)
        model.load_state_dict(torch.load(f'{data_dir}/tmodel.pt', map_location=device))
        model = model.to(device)

    model.eval()

    # Single step prediction at given timestep
    x = test[timestep].to(device)
    with torch.no_grad():
        if domain == 'lam':
            delta = model(x, edge_index, edge_features)
        else:
            delta = model(x, graph)
        pred = x + delta

    pred_denorm   = pred.cpu().numpy() * std + mean
    actual_denorm = test[timestep + 1].numpy() * std + mean

    # Rollout MAE over test set
    mae_per_step = np.zeros(K)
    counts = 0
    with torch.no_grad():
        for t in range(len(test) - K):
            x = test[t].to(device)
            for k in range(K):
                if domain == 'lam':
                    delta = model(x, edge_index, edge_features)
                else:
                    delta = model(x, graph)
                pred_t = x + delta
                mae_per_step[k] += torch.mean(
                    torch.abs(pred_t - test[t + k + 1].to(device))
                ).item()
                x = pred_t
            counts += 1

    mae_per_step /= counts
    return pred_denorm, actual_denorm, mae_per_step, std, mean

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    pred, actual, mae_per_step, std, mean = inference(timestep=0, device=device)

    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    units     = ['m/s', 'm/s', 'Pa', 'K', 'K', 'm²/s²', 'm²/s²']

    print("\nPer-variable MAE at T+1 (physical units):")
    for i, (name, unit) in enumerate(zip(var_names, units)):
        mae_phys = np.mean(np.abs(pred[:, i] - actual[:, i]))
        mae_norm = mae_phys / std[i]
        print(f"  {name}: {mae_phys:.4f} {unit} (normalized: {mae_norm:.4f})")

    print("\nRollout MAE (normalized):")
    for k in range(len(mae_per_step)):
        hours = (k + 1) * 6
        print(f"  T+{k+1} ({hours}h): {mae_per_step[k]:.6f}")