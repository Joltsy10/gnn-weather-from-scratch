import torch
import torch.nn as nn
from model.message_passing import MessagePassingLayer
import yaml

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)
    
class HiGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, num_levels):
        super().__init__()
        config = load_config()
        hidden_dim = config['model']['hidden_dim']

        self.grid_encoder = nn.Linear(node_dim, hidden_dim)

        self.g2m_gnn = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)

        self.m2g_gnn = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)

        self.grid_decoder = nn.Linear(hidden_dim, node_dim)

        # One same-level GNN per mesh level
        self.same_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels)
        ])

        # One up GNN per level transition (num_levels - 1 transitions)
        self.up_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

        # One down GNN per level transition
        self.down_gnns = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

        # Mesh node encoders: one per level
        # Each mesh level needs its own encoder since they start with
        # no features — we initialize them to zeros and let the G2M
        # pass populate the finest level, then up/down populate the rest
        self.mesh_encoders = nn.ModuleList([
            nn.Linear(2, hidden_dim)  # mesh node features are [lat, lon]
            for _ in range(num_levels)
        ])

    def forward(self, grid_features, graph):
        """
        Args:
            'g2m_edge_index':   (2, E_g2m)
            'g2m_features':     (E_g2m, edge_dim)
            'm2g_edge_index':   (2, E_m2g)
            'm2g_features':     (E_m2g, edge_dim)
            'm2m_edge_index':   list of (2, E_m2m) per level
            'm2m_features':     list of (E_m2m, edge_dim) per level
            'up_edge_index':    list of (2, E_up) per transition
            'up_features':      list of (E_up, edge_dim) per transition
            'down_edge_index':  list of (2, E_down) per transition
            'down_features':    list of (E_down, edge_dim) per transition
            'mesh_features':    list of (N_mesh, 2) per level
        """

        grid_rep = self.grid_encoder(grid_features) # (N_grid, hidden_dim)

        # Encoder
        mesh_rep = [
            encoder(graph['mesh_features'][i])
            for i, encoder in enumerate(self.mesh_encoders)
        ]
        # mesh_rep is a list of (N_mesh[i], hidden_dim), one per level

        # G2M: grid to finest mesh level
        finest_mesh_rep = self.g2m_gnn(
            grid_rep, mesh_rep[-1], graph['g2m_edge_index'], graph['g2m_features'],
            n_dst_nodes=mesh_rep[-1].shape[0]
        )

        mesh_rep[-1] = finest_mesh_rep # (N_mesh_finest, hidden_dim)

        # Processor: up sweep (finest to coarsest)
        for i in reversed(range(len(self.up_gnns))):
            mesh_rep[i+1] = self.same_gnns[i+1](
            mesh_rep[i+1], mesh_rep[i+1], graph['m2m_edge_index'][i+1],
            graph['m2m_features'][i+1]
            )   

            mesh_rep[i] = self.up_gnns[i](
                mesh_rep[i+1], mesh_rep[i], graph['up_edge_index'][i],
                graph['up_features'][i],
                n_dst_nodes=mesh_rep[i].shape[0]
            )   

        # Same-level pass on coarsest level
        mesh_rep[0] = self.same_gnns[0](
            mesh_rep[0], mesh_rep[0], graph['m2m_edge_index'][0],
            graph['m2m_features'][0]
        )

        # Processor: down sweep (coarsest to finest)
        for i in range(len(self.up_gnns)):
            mesh_rep[i+1] = self.down_gnns[i](
                mesh_rep[i], mesh_rep[i+1], graph['down_edge_index'][i],
                graph['down_features'][i],
                n_dst_nodes=mesh_rep[i+1].shape[0]
            )

            mesh_rep[i+1] = self.same_gnns[i+1](
                mesh_rep[i+1], mesh_rep[i+1], graph['m2m_edge_index'][i+1],
                graph['m2m_features'][i+1]
            )

        # M2G: finest mesh to grid
        grid_rep_out = self.m2g_gnn(
            mesh_rep[-1], grid_rep, graph['m2g_edge_index'], graph['m2g_features'],
            n_dst_nodes=grid_rep.shape[0]
        ) # (N_grid, hidden_dim)

        # Decoder
        delta = self.grid_decoder(grid_rep_out)

        return delta
