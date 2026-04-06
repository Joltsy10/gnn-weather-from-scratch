# test_train.py
import numpy as np
import torch
import os
from torch_geometric.data import HeteroData

# create tiny fake graph
N_grid  = 100
N_mesh0 = 12
N_mesh1 = 42
N_mesh2 = 162
edge_dim = 3
node_dim = 7
hidden_dim = 16

graph = HeteroData()
graph['mesh_0'].x = torch.randn(N_mesh0, 4)
graph['mesh_1'].x = torch.randn(N_mesh1, 4)
graph['mesh_2'].x = torch.randn(N_mesh2, 4)

def random_edges(n_src, n_dst, n_edges):
    return torch.stack([
        torch.randint(0, n_src, (n_edges,)),
        torch.randint(0, n_dst, (n_edges,))
    ])

graph['grid', 'g2m', 'mesh_2'].edge_index = random_edges(N_grid, N_mesh2, 300)
graph['grid', 'g2m', 'mesh_2'].edge_attr  = torch.randn(300, edge_dim)
graph['mesh_2', 'm2g', 'grid'].edge_index = random_edges(N_mesh2, N_grid, 300)
graph['mesh_2', 'm2g', 'grid'].edge_attr  = torch.randn(300, edge_dim)

for i, n in enumerate([N_mesh0, N_mesh1, N_mesh2]):
    graph[f'mesh_{i}', 'm2m', f'mesh_{i}'].edge_index = random_edges(n, n, n*4)
    graph[f'mesh_{i}', 'm2m', f'mesh_{i}'].edge_attr  = torch.randn(n*4, edge_dim)

graph['mesh_1', 'up', 'mesh_0'].edge_index = random_edges(N_mesh1, N_mesh0, N_mesh1)
graph['mesh_1', 'up', 'mesh_0'].edge_attr  = torch.randn(N_mesh1, edge_dim)
graph['mesh_2', 'up', 'mesh_1'].edge_index = random_edges(N_mesh2, N_mesh1, N_mesh2)
graph['mesh_2', 'up', 'mesh_1'].edge_attr  = torch.randn(N_mesh2, edge_dim)
graph['mesh_0', 'down', 'mesh_1'].edge_index = random_edges(N_mesh0, N_mesh1, N_mesh1)
graph['mesh_0', 'down', 'mesh_1'].edge_attr  = torch.randn(N_mesh1, edge_dim)
graph['mesh_1', 'down', 'mesh_2'].edge_index = random_edges(N_mesh1, N_mesh2, N_mesh2)
graph['mesh_1', 'down', 'mesh_2'].edge_attr  = torch.randn(N_mesh2, edge_dim)

# save fake graph
os.makedirs('data/global', exist_ok=True)
torch.save(graph, 'data/global/graph.pt')

# create fake node features
T = 50
npy_path = 'data/global/node_features.npy'
fake_data = np.random.randn(T, N_grid, node_dim).astype(np.float32)
np.save(npy_path, fake_data)

# patch config
import yaml
config = {
    'domain': 'global',
    'model': {'node_dim': node_dim, 'hidden_dim': hidden_dim},
    'data': {'train_end': 40, 'val_end': 45},
    'training': {
        'num_epochs': 2,
        'lr': 0.001,
        'gcs_checkpoint': None
    }
}
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

# run training
from training.train import train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
train(device=device, resume=False)

# cleanup
os.remove('data/global/graph.pt')
os.remove('data/global/node_features.npy')
os.remove('config.yaml')
print("Train test passed")