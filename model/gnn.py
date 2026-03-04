import torch
import torch.nn as nn
from model.message_passing import MessagePassingLayer

class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Processor
        self.processor = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim) 
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, node_features, edge_index, edge_features):
            # Encoder
            x = self.encoder(node_features) # (N, hidden_dim)

            # Processor
            for layer in self.processor:
                 x = layer(x, edge_index, edge_features) # (N, hidden_dim)

            # Decoder
            out = self.decoder(x) # (N, node_dim)

            return out
    
if __name__ == "__main__":
    from model.message_passing import MessagePassingLayer
    import numpy as np

    edge_index = torch.tensor(np.load('data/edge_index.npy'), dtype=torch.long)
    edge_features = torch.tensor(np.load('data/edge_features.npy'), dtype=torch.float32)
    node_features = torch.tensor(np.load('data/node_features.npy')[0], dtype=torch.float32)

    model = GNN(node_dim=7, edge_dim=3, hidden_dim=64, num_layers=6)
    out = model(node_features, edge_index, edge_features)
    print(out.shape)  # should be (15609, 7)
