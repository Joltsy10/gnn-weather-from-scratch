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

def inference(device='cpu'):
    config   = load_config()
    domain   = config['domain']
    data_dir = f'data/{domain}'
    node_dim = config['model']['node_dim']
    val_end  = config['data']['val_end']
    rollout_steps = config['training'][rollout_steps]

    node_features = torch.from_numpy(
        np.load(f'{data_dir}/node_features.npy', mmap_mode='r')
    )
    mean = torch.load(f'{data_dir}/mean.pt').numpy()
    std  = torch.load(f'{data_dir}/std.pt').numpy()

    test = node_features[val_end:]

    if domain == 'lam':
        edge_index    = torch.load(f'{data_dir}/edge_index.pt').to(device)
        edge_features = torch.load(f'{data_dir}/edge_features.pt').to(device)
        edge_dim      = edge_features.shape[1]
        model         = GNN(node_dim=node_dim, edge_dim=edge_dim)
        model.load_state_dict(torch.load(f'{data_dir}/tmodel.pt', map_location=device))
        model         = model.to(device)
    else:
        graph, num_levels = load_global_graph(data_dir, device)
        edge_dim          = graph['g2m_features'].shape[1]
        model             = HiGNN(node_dim=node_dim, edge_dim=edge_dim, num_levels=num_levels)
        model.load_state_dict(torch.load(f'{data_dir}/tmodel.pt', map_location=device))
        model             = model.to(device)

    model.eval()

    mae_model       = np.zeros(rollout_steps)
    mae_persistence = np.zeros(rollout_steps)
    counts          = 0

    with torch.no_grad():
        for t in range(len(test) - rollout_steps):
            x = test[t].to(device)

            for k in range(rollout_steps):
                target = test[t + k + 1].to(device)

                if domain == 'lam':
                    delta = model(x, edge_index, edge_features)
                else:
                    delta = model(x, graph)

                pred = x + delta
                mae_model[k] += torch.mean(torch.abs(pred - target)).item()

                persistence = test[t].to(device)
                mae_persistence[k] += torch.mean(torch.abs(persistence - target)).item()

                x = pred

            counts += 1

    mae_model       /= counts
    mae_persistence /= counts

    return mae_model, mae_persistence, std, mean

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config   = load_config()
    rollout_steps = config['training']['rollout_steps']
    mae_model, mae_persistence, std, mean = inference(device=device)

    var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
    units     = ['m/s', 'm/s', 'Pa', 'K',   'K',    'm²/s²','m²/s²']

    print("\nRollout MAE — model vs persistence (normalized):")
    print(f"  {'Step':<8} {'Hours':<8} {'Model':<12} {'Persistence':<12} {'Skill'}")
    print(f"  {'-'*52}")
    for k in range(rollout_steps):
        hours = (k + 1) * 6
        skill = (mae_persistence[k] - mae_model[k]) / mae_persistence[k] * 100
        print(f"  T+{k+1:<6} {hours:<8} {mae_model[k]:<12.6f} {mae_persistence[k]:<12.6f} {skill:+.1f}%")

    print("\nT+1 MAE per variable (physical units):")
    print(f"  {'Var':<8} {'Model':<14} {'Persistence':<14} {'Unit'}")
    print(f"  {'-'*48}")
    for i, (name, unit) in enumerate(zip(var_names, units)):
        model_phys = mae_model[0] * std[i]
        pers_phys  = mae_persistence[0] * std[i]
        print(f"  {name:<8} {model_phys:<14.4f} {pers_phys:<14.4f} {unit}")