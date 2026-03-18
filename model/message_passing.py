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

    def forward(self, src_features, dst_features, edge_index, 
                edge_features, n_dst_nodes=None):
        """
        Args:
            src_features: (N_src, node_dim) source node features
            dst_features: (N_dst, node_dim) destination node features
            edge_index:   (2, E) row 0 = src indices, row 1 = dst indices
            edge_features: (E, edge_dim)
            n_dst_nodes:  int, optional override for dst node count
        """
        if n_dst_nodes is None:
            n_dst_nodes = dst_features.shape[0]

        src = edge_index[0]
        dst = edge_index[1]

        src_node_features = src_features[src]  # (E, node_dim)
        msg_input = torch.cat([src_node_features, edge_features], dim=-1)
        messages = self.message_mlp(msg_input)  # (E, hidden_dim)

        aggregated = torch.zeros(n_dst_nodes, messages.shape[1],
                                 device=src_features.device,
                                 dtype=src_features.dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), 
                                messages)

        update_input = torch.cat([dst_features, aggregated], dim=-1)
        new_dst_features = self.update_mlp(update_input)  # (N_dst, node_dim)

        return new_dst_features
    
if __name__ == "__main__":
    layer = MessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)
    nodes = torch.randn(15609, 7)
    edge_index = torch.randint(0, 15609, (2, 124872))
    edges = torch.randn(124872, 3)
    out = layer(nodes, nodes, edge_index, edges)
    print(out.shape)  # should be (15609, 7)