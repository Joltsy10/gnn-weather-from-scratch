import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, node_features, edge_index, edge_features):
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)

        src_features = node_features[src]  # (E, node_dim)
        msg_input = torch.cat([src_features, edge_features], dim=-1)  # (E, node_dim+edge_dim)
        messages = self.message_mlp(msg_input)  # (E, hidden_dim)

        aggregated = torch.zeros(node_features.shape[0], messages.shape[1], 
                                 device=node_features.device)  # (N, hidden_dim)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        update_input = torch.cat([node_features, aggregated], dim=-1)  # (N, node_dim+hidden_dim)
        new_node_features = self.update_mlp(update_input)  # (N, node_dim)

        return new_node_features
    
if __name__ == "__main__":
    layer = MessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)
    nodes = torch.randn(15609, 7)
    edge_index = torch.randint(0, 15609, (2, 124872))
    edges = torch.randn(124872, 3)
    out = layer(nodes, edge_index, edges)
    print(out.shape)  # should be (15609, 7)