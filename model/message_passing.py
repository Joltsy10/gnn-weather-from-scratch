import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MessagePassingLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')

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

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, src_features, dst_features, edge_index,
                edge_features, n_dst_nodes=None):
        if n_dst_nodes is None:
            n_dst_nodes = dst_features.shape[0]

        self._dst = dst_features
    
        return self.propagate(
            edge_index,
            x=(src_features, dst_features),
            edge_attr=edge_features,
            size=(src_features.shape[0], n_dst_nodes)
        )

    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        dst = self._dst
        new_dst = self.update_mlp(torch.cat([dst, aggr_out], dim=-1))
        return self.norm(dst + new_dst)


if __name__ == "__main__":
    layer = MessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)
    nodes = torch.randn(15609, 7)
    edge_index = torch.randint(0, 15609, (2, 124872))
    edges = torch.randn(124872, 3)
    out = layer(nodes, nodes, edge_index, edges)
    print(out.shape)  # should be (15609, 7)