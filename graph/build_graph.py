import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mesh')))

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
import yaml
import torch
from bridge import build_graph as build_icosahedral_graph


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(data_dir, years=['2019', '2020', '2021', '2022']):
    """
    Load ERA5 files and flatten from (time, lat, lon) to (time, N, 7)
    Works for both LAM and global
    """

    surface_list = []
    pressure_list = []
    for year in years:
        surface_list.append(xr.open_dataset(f'{data_dir}/era5_surface_{year}.nc'))
        pressure_list.append(xr.open_dataset(f'{data_dir}/era5_pressure_{year}.nc'))

    surface = xr.concat(surface_list, dim='valid_time')
    pressure = xr.concat(pressure_list, dim='valid_time')

    T = len(surface.valid_time)  # 2688

    u10 = surface['u10'].values.reshape(T, -1)   # (T, N)
    v10 = surface['v10'].values.reshape(T, -1)   # (T, N)
    sp  = surface['sp'].values.reshape(T, -1)    # (T, N)

    t_850 = pressure['t'].values[:, 0, :, :].reshape(T, -1)  # (T, N)
    t_500 = pressure['t'].values[:, 1, :, :].reshape(T, -1)  # (T, N)
    z_850 = pressure['z'].values[:, 0, :, :].reshape(T, -1)  # (T, N)
    z_500 = pressure['z'].values[:, 1, :, :].reshape(T, -1)  # (T, N)

    node_features = np.stack([u10, v10, sp, t_850, t_500, z_850, z_500], axis=-1)
    # shape: (T, N, 7)

    lat_grid, lon_grid = np.meshgrid(
        surface.latitude.values,
        surface.longitude.values,
        indexing='ij'
    )
    lat_flat = lat_grid.reshape(-1)  # (N,)
    lon_flat = lon_grid.reshape(-1)  # (N,)

    return node_features, lat_flat, lon_flat

def normalize(node_features):
    mean = node_features.mean(axis=(0, 1), keepdims=True)  # (1, 1, 7)
    std  = node_features.std(axis=(0, 1), keepdims=True)   # (1, 1, 7)
    return (node_features - mean) / std, mean.squeeze(), std.squeeze()

def build_lam_edges(lat_flat, lon_flat, k=16):
    """
    Connect each node to its k nearest neighbours.
    Returns edge_index of shape (2, E) and edge_features of shape (E, 3)
    """
    coords = np.stack([lat_flat, lon_flat], axis=-1)  # (N, 2)
    
    tree = cKDTree(coords)
    
    distances, indices = tree.query(coords, k=k+1)
    # distances: (N, k+1), indices: (N, k+1)
    indices = indices[:, 1:]    # (N, k)
    distances = distances[:, 1:]  # (N, k)

    n_nodes = len(lat_flat)
    source_nodes = np.repeat(np.arange(n_nodes), k)  # (N*k,)
    dest_nodes = indices.reshape(-1)                   # (N*k,)

    edge_index = np.stack([source_nodes, dest_nodes], axis=0)  # (2, E)

    delta_lat = lat_flat[dest_nodes] - lat_flat[source_nodes]  # (E,)
    delta_lon = lon_flat[dest_nodes] - lon_flat[source_nodes]  # (E,)
    dist      = distances.reshape(-1)                           # (E,)

    edge_features = np.stack([delta_lat, delta_lon, dist], axis=-1)  # (E, 3)

    return edge_index, edge_features

def build_and_save(config_path='config.yaml'):
    config = load_config(config_path)
    domain = config['domain']

    if domain == 'lam':
        data_dir = 'data/lam'
        graph_dir = 'data/lam'
    else:
        data_dir = 'data/global'
        graph_dir = 'data/global'

    print(f"Domain: {domain}")
    print("Loading data...")
    node_features, lat_flat, lon_flat = load_data(data_dir)

    print("Normalizing...")
    node_features, mean, std = normalize(node_features)
    torch.save(torch.tensor(mean, dtype=torch.float32), f'{graph_dir}/mean.pt')
    torch.save(torch.tensor(std, dtype=torch.float32), f'{graph_dir}/std.pt')
    
    print("Building graph...")
    if domain == 'lam':
        edge_index, edge_features = build_lam_edges(
            lat_flat, lon_flat, k=config['graph']['k']
        )
        torch.save(torch.tensor(node_features, dtype=torch.float32),
                   f'{graph_dir}/node_features.pt')
        torch.save(torch.tensor(edge_index,    dtype=torch.long),
                   f'{graph_dir}/edge_index.pt')
        torch.save(torch.tensor(edge_features, dtype=torch.float32),
                   f'{graph_dir}/edge_features.pt')
        torch.save(torch.tensor(lat_flat,      dtype=torch.float32),
                   f'{graph_dir}/lat.pt')
        torch.save(torch.tensor(lon_flat,      dtype=torch.float32),
                   f'{graph_dir}/lon.pt')
        
        print(f"Nodes: {len(lat_flat)}")
        print(f"Edges: {edge_index.shape[1]}")

    else:
        torch.save(torch.tensor(node_features, dtype=torch.float32),
                   f'{graph_dir}/node_features.pt')
        torch.save(torch.tensor(lat_flat,      dtype=torch.float32),
                   f'{graph_dir}/lat.pt')
        torch.save(torch.tensor(lon_flat,      dtype=torch.float32),
                   f'{graph_dir}/lon.pt')

        build_icosahedral_graph(
            mesh_level    = config['graph']['mesh_level'],
            grid_lat      = lat_flat,
            grid_lon      = lon_flat,
            output_dir    = graph_dir,
            g2m_angle_deg = config['graph']['g2m_angle_deg']
        )

    print(f"Node features shape: {node_features.shape}")   
    print(f"Saved to {graph_dir}")

if __name__ == "__main__":
    build_and_save()

