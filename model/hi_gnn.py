import torch
import torch.nn as nn
from model.message_passing import MessagePassingLayer


class HiGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.finest     = num_levels - 1

        self.grid_encoder = nn.Linear(node_dim, hidden_dim)
        self.grid_decoder = nn.Linear(hidden_dim, node_dim)

        self.g2m_gnn = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
        self.m2g_gnn = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)

        self.mesh_encoders = nn.ModuleList([
            nn.Linear(4, hidden_dim)
            for _ in range(num_levels)
        ])

        self.same_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels)
        ])

        self.up_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

        self.down_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

    def forward(self, grid_features, graph):
        grid_rep = self.grid_encoder(grid_features)

        mesh_rep = [
            encoder(graph[f'mesh_{i}'].x)
            for i, encoder in enumerate(self.mesh_encoders)
        ]

        finest = self.finest

        mesh_rep[finest] = self.g2m_gnn(
            grid_rep,
            mesh_rep[finest],
            graph['grid', 'g2m', f'mesh_{finest}'].edge_index,
            graph['grid', 'g2m', f'mesh_{finest}'].edge_attr,
            n_dst_nodes=mesh_rep[finest].shape[0]
        )

        for i in reversed(range(len(self.up_gnns))):
            mesh_rep[i+1] = self.same_gnns[i+1](
                mesh_rep[i+1],
                mesh_rep[i+1],
                graph[f'mesh_{i+1}', 'm2m', f'mesh_{i+1}'].edge_index,
                graph[f'mesh_{i+1}', 'm2m', f'mesh_{i+1}'].edge_attr,
            )

            mesh_rep[i] = self.up_gnns[i](
                mesh_rep[i+1],
                mesh_rep[i],
                graph[f'mesh_{i+1}', 'up', f'mesh_{i}'].edge_index,
                graph[f'mesh_{i+1}', 'up', f'mesh_{i}'].edge_attr,
                n_dst_nodes=mesh_rep[i].shape[0]
            )

        mesh_rep[0] = self.same_gnns[0](
            mesh_rep[0],
            mesh_rep[0],
            graph['mesh_0', 'm2m', 'mesh_0'].edge_index,
            graph['mesh_0', 'm2m', 'mesh_0'].edge_attr,
        )

        for i in range(len(self.down_gnns)):
            mesh_rep[i+1] = self.down_gnns[i](
                mesh_rep[i],
                mesh_rep[i+1],
                graph[f'mesh_{i}', 'down', f'mesh_{i+1}'].edge_index,
                graph[f'mesh_{i}', 'down', f'mesh_{i+1}'].edge_attr,
                n_dst_nodes=mesh_rep[i+1].shape[0]
            )

            mesh_rep[i+1] = self.same_gnns[i+1](
                mesh_rep[i+1],
                mesh_rep[i+1],
                graph[f'mesh_{i+1}', 'm2m', f'mesh_{i+1}'].edge_index,
                graph[f'mesh_{i+1}', 'm2m', f'mesh_{i+1}'].edge_attr,
            )

        grid_rep_out = self.m2g_gnn(
            mesh_rep[finest],
            grid_rep,
            graph[f'mesh_{finest}', 'm2g', 'grid'].edge_index,
            graph[f'mesh_{finest}', 'm2g', 'grid'].edge_attr,
            n_dst_nodes=grid_rep.shape[0]
        )

        return self.grid_decoder(grid_rep_out)