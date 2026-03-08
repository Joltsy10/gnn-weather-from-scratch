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

def load_data():
    node_features = torch.tensor(
        np.load('data/node_features.npy'), dtype=torch.float32
    ) # (T, N, 7)
    edge_index = torch.tensor(
        np.load('data/edge_index.npy'), dtype=torch.long
    ) # (2, E)
    edge_features = torch.tensor(
        np.load('data/edge_features.npy'), dtype=torch.float32
    )# (E, 3)
    return node_features, edge_index, edge_features

def train(device = 'cpu', resume=False):
    config = load_config()
    node_features, edge_index, edge_features = load_data()
    T = node_features.shape[0]

    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    train = node_features[:2688].to(device)   # 2019 + 2020
    val   = node_features[2688:4032].to(device)  # 2021
    test  = node_features[4032:].to(device)               # 2022

    model = GNN(node_dim=7, edge_dim=3).to(device)
    if resume and os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt', map_location=device))
    print("Resumed from checkpoint")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    K = 4

    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0.0

        for t in range(train.shape[0] - K):
            x = train[t] # (N, 7)
            loss = 0
    
            for k in range(K):
                delta = model(x, edge_index, edge_features)
                pred = x + delta
                loss += loss_fn(pred, train[t + k + 1])
                x = pred

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for t in range(len(val)-K):
                x = val[t]
                for k in range(K):
                    pred = model(x, edge_index, edge_features)
                    val_total += loss_fn(pred, val[t + k + 1]).item()
                    x = pred
        val_loss = val_total / ((len(val)-K) * K)
        avg_loss = total_loss/(train.shape[0] - 1)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} — train: {avg_loss:.6f} val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'model.pt')
            print(f"  saved new best val: {best_val:.6f}")
        scheduler.step(val_loss)
    

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(device=device, resume=False)