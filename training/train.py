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

def train(device = 'cpu'):
    config = load_config()
    node_features, edge_index, edge_features = load_data()
    T = node_features.shape[0]

    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    node_features, edge_index, edge_features = load_data()

    train = node_features[:2920]   # 2019 + 2020
    val   = node_features[2920:4380]  # 2021
    test  = node_features[4380:]      # 2022

    model = GNN(node_dim=7, edge_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    loss_fn = nn.MSELoss()

    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0.0

        for t in range(train.shape[0] -1):
            x = train[t] # (N, 7)
            y = train[t + 1] # (N, 7)

            pred = model(x, edge_index, edge_features) # (N, 7)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(model(val[t], edge_index, edge_features), val[t+1])
                for t in range(len(val)-1)
            ) / (len(val)-1)
        avg_loss = total_loss/(train.shape[0] - 1)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} — train: {avg_loss:.6f} val: {val_loss:.6f}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(device=device)