import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn import GNN

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(data_dir):
    node_features = torch.load(f'{data_dir}/node_features.pt')  # (T, N, node_dim)
    edge_index    = torch.load(f'{data_dir}/edge_index.pt')     # (2, E)
    edge_features = torch.load(f'{data_dir}/edge_features.pt')  # (E, edge_dim)
    return node_features, edge_index, edge_features

def train(device='cpu', resume=False):
    config  = load_config()
    domain  = config['domain']
    data_dir = f'data/{domain}'

    node_features, edge_index, edge_features = load_data(data_dir)

    edge_index    = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    train_data = node_features[:2688].to(device)
    val_data   = node_features[2688:4032].to(device)

    node_dim = config['model']['node_dim']
    edge_dim = edge_features.shape[1]

    model = GNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
    if resume and os.path.exists(f'{data_dir}/model.pt'):
        model.load_state_dict(torch.load(f'{data_dir}/model.pt', map_location=device))
        print("Resumed from checkpoint")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn   = nn.MSELoss()
    best_val  = float('inf')
    K = 4

    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0.0

        for t in range(train_data.shape[0] - K):
            x    = train_data[t]
            loss = 0
            for k in range(K):
                delta = model(x, edge_index, edge_features)
                pred  = x + delta
                loss += loss_fn(pred, train_data[t + k + 1])
                x = pred

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for t in range(len(val_data) - K):
                x = val_data[t]
                for k in range(K):
                    delta = model(x, edge_index, edge_features)
                    pred  = x + delta
                    val_total += loss_fn(pred, val_data[t + k + 1]).item()
                    x = pred

        val_loss  = val_total / ((len(val_data) - K) * K)
        avg_loss  = total_loss / (train_data.shape[0] - 1)
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} — train: {avg_loss:.6f} val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'{data_dir}/tmodel.pt')
            print(f"  saved new best val: {best_val:.6f}")

        scheduler.step(val_loss)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(device=device, resume=False)