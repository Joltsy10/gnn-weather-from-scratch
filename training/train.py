import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from model.gnn import GNN
from model.hi_gnn import HiGNN
from data.dataset import WeatherDataset

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_lam_graph(data_dir, device):
    from torch_geometric.data import HeteroData
    graph = torch.load(f'{data_dir}/graph.pt', map_location=device, weights_only=False)
    edge_dim = graph['grid', 'knn', 'grid'].edge_attr.shape[1]
    return graph, edge_dim

def load_global_graph(data_dir, device):
    graph = torch.load(f'{data_dir}/graph.pt', map_location=device, weights_only=False)
    edge_dim    = graph['grid', 'g2m', graph.node_types[-1]].edge_attr.shape[1]
    num_levels  = sum(1 for nt in graph.node_types if nt.startswith('mesh_'))
    return graph, edge_dim, num_levels

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, gcs_path=None):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val':             best_val,
    }, path)
    if gcs_path:
        os.system(f'gcloud storage cp {path} {gcs_path}')

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_val']

def train(device='cpu', resume=False):
    config         = load_config()
    domain         = config['domain']
    data_dir       = f'data/{domain}'
    node_dim       = config['model']['node_dim']
    hidden_dim     = config['model']['hidden_dim']
    gcs_checkpoint = config['training'].get('gcs_checkpoint', None)
    use_bf16       = device == 'cuda'
    train_end      = config['data']['train_end']
    val_end        = config['data']['val_end']
    npy_path       = f'{data_dir}/node_features.npy'

    if domain == 'lam':
        graph, edge_dim = load_lam_graph(data_dir, device)
        model = GNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    else:
        graph, edge_dim, num_levels = load_global_graph(data_dir, device)
        model = HiGNN(node_dim=node_dim, edge_dim=edge_dim,
                      hidden_dim=hidden_dim, num_levels=num_levels).to(device)

    train_dataset = WeatherDataset(npy_path, 0, train_end)
    val_dataset   = WeatherDataset(npy_path, train_end, val_end)
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    optimizer    = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn      = nn.MSELoss()
    best_val     = float('inf')
    start_epoch  = 0
    checkpoint_path = f'{data_dir}/checkpoint_latest.pt'

    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_val = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        total_loss = 0.0

        for x, x_next in train_loader:
            x      = x.squeeze(0).to(device)
            x_next = x_next.squeeze(0).to(device)

            graph['grid'].x = x

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                delta = model(x, graph)
                pred  = x + delta
                loss  = loss_fn(pred, x_next)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, x_next in val_loader:
                x      = x.squeeze(0).to(device)
                x_next = x_next.squeeze(0).to(device)

                graph['grid'].x = x

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    delta      = model(x, graph)
                    pred       = x + delta
                    val_total += loss_fn(pred, x_next).item()

        val_loss = val_total / len(val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} — train: {avg_loss:.6f} val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'{data_dir}/tmodel.pt')
            print(f"  new best val: {best_val:.6f}")

        save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, best_val, gcs_checkpoint)
        scheduler.step(val_loss)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(device=device, resume=True)