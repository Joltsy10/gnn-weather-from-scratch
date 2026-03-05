import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gnn import GNN

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

def train(num_epochs = 10, lr = 1e-3, device = 'cpu'):
    node_features, edge_index, edge_features = load_data()
    T = node_features.shape[0]

    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    model = GNN(node_dim=7, edge_dim=3, hidden_dim=64, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for t in range(T -1):
            x = node_features[t] # (N, 7)
            y = node_features[t + 1] # (N, 7)

            pred = model(x, edge_index, edge_features) # (N, 7)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_loss = total_loss/(T - 1)
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.6f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(device=device)