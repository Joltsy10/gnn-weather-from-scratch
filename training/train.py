import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn import GNN
from model.hi_gnn import HiGNN

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(data_dir):
    node_features = torch.load(f'{data_dir}/node_features.pt')
    edge_index    = torch.load(f'{data_dir}/edge_index.pt')
    edge_features = torch.load(f'{data_dir}/edge_features.pt')
    return node_features, edge_index, edge_features

def load_global_graph(graph_dir, device):
    m2m_edge_index = torch.load(f'{graph_dir}/m2m_edge_index.pt', map_location=device)
    num_levels = len(m2m_edge_index)
    graph = {
        'g2m_edge_index':  torch.load(f'{graph_dir}/g2m_edge_index.pt',      map_location=device),
        'g2m_features':    torch.load(f'{graph_dir}/g2m_features.pt',         map_location=device),
        'm2g_edge_index':  torch.load(f'{graph_dir}/m2g_edge_index.pt',       map_location=device),
        'm2g_features':    torch.load(f'{graph_dir}/m2g_features.pt',         map_location=device),
        'm2m_edge_index':  m2m_edge_index,
        'm2m_features':    torch.load(f'{graph_dir}/m2m_features.pt',         map_location=device),
        'up_edge_index':   torch.load(f'{graph_dir}/mesh_up_edge_index.pt',   map_location=device),
        'up_features':     torch.load(f'{graph_dir}/mesh_up_features.pt',     map_location=device),
        'down_edge_index': torch.load(f'{graph_dir}/mesh_down_edge_index.pt', map_location=device),
        'down_features':   torch.load(f'{graph_dir}/mesh_down_features.pt',   map_location=device),
        'mesh_features':   torch.load(f'{graph_dir}/mesh_features.pt',        map_location=device),
    }
    return graph, num_levels

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

def forward(model, x, graph, domain, edge_index=None, edge_features=None):
    if domain == 'lam':
        return model(x, edge_index, edge_features)
    return model(x, graph)

def train(device='cpu', resume=False):
    config   = load_config()
    domain   = config['domain']
    data_dir = f'data/{domain}'
    node_dim = config['model']['node_dim']
    accum_steps = config['training'].get('accum_steps', 16)
    gcs_checkpoint = config['training'].get('gcs_checkpoint', None)
    use_bf16 = device == 'cuda'

    edge_index, edge_features, graph = None, None, None

    if domain == 'lam':
        node_features, edge_index, edge_features = load_data(data_dir)
        edge_index    = edge_index.to(device)
        edge_features = edge_features.to(device)
        edge_dim      = edge_features.shape[1]
        model         = GNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
    else:
        node_features      = node_features = torch.from_numpy(np.load(f'{data_dir}/node_features.npy', mmap_mode='r'))
        graph, num_levels  = load_global_graph(data_dir, device)
        edge_dim           = graph['g2m_features'].shape[1]
        model              = HiGNN(node_dim=node_dim, edge_dim=edge_dim,
                                   num_levels=num_levels).to(device)

    node_features = node_features
    train_end     = config['data']['train_end']
    val_end       = config['data']['val_end']
    train_data    = node_features[:train_end]
    val_data      = node_features[train_end:val_end]

    optimizer  = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn    = nn.MSELoss()
    best_val   = float('inf')
    start_epoch = 0
    K = 1

    checkpoint_path = f'{data_dir}/checkpoint_latest.pt'

    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_val = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        total_loss  = 0.0
        optimizer.zero_grad()

        for t in range(train_data.shape[0] - K):
            x = train_data[t].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                loss = 0
                for k in range(K):
                    delta = forward(model, x, graph, domain, edge_index, edge_features)
                    pred  = x + delta
                    loss += loss_fn(pred, train_data[t + k + 1].to(device))
                    x     = pred

            (loss / accum_steps).backward()
            total_loss += loss.item()

            if (t + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (train_data.shape[0] - K) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for t in range(len(val_data) - K):
                x = val_data[t].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    for k in range(K):
                        delta      = forward(model, x, graph, domain, edge_index, edge_features)
                        pred       = x + delta
                        val_total += loss_fn(pred, val_data[t + k + 1].to(device)).item()
                        x          = pred

        val_loss = val_total / ((len(val_data) - K) * K)
        avg_loss = total_loss / (train_data.shape[0] - K)
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
    train(device=device, resume=False)