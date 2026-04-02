import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'mesh', 'neural-lam-global-mesh'))

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
import yaml
import torch
from bridge import build_graph as build_icosahedral_graph


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def get_years(data_dir):
    files = os.listdir(data_dir)
    years = sorted(set(
        f.split('_')[-1].replace('.nc', '')
        for f in files if f.startswith('era5_surface_')
    ))
    return years

def load_year(data_dir, year):
    surface  = xr.open_dataset(f'{data_dir}/era5_surface_{year}.nc')
    pressure = xr.open_dataset(f'{data_dir}/era5_pressure_{year}.nc')

    T = len(surface.valid_time)
    N = surface['u10'].shape[1] * surface['u10'].shape[2]

    features = np.empty((T, N, 7), dtype=np.float32)
    features[:, :, 0] = surface['u10'].values.reshape(T, -1)
    features[:, :, 1] = surface['v10'].values.reshape(T, -1)
    features[:, :, 2] = surface['sp'].values.reshape(T, -1)
    features[:, :, 3] = pressure['t'].values[:, 0, :, :].reshape(T, -1)
    features[:, :, 4] = pressure['t'].values[:, 1, :, :].reshape(T, -1)
    features[:, :, 5] = pressure['z'].values[:, 0, :, :].reshape(T, -1)
    features[:, :, 6] = pressure['z'].values[:, 1, :, :].reshape(T, -1)

    lat_grid, lon_grid = np.meshgrid(
        surface.latitude.values,
        surface.longitude.values,
        indexing='ij'
    )
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)

    surface.close()
    pressure.close()

    return features, lat_flat, lon_flat
def compute_mean_std(data_dir, years):
    total_n = 0
    combined_mean = np.zeros(7, dtype=np.float64)
    combined_var  = np.zeros(7, dtype=np.float64)

    for year in years:
        print(f"  Computing stats for {year}...")
        features, _, _ = load_year(data_dir, year)
        flat = features.reshape(-1, 7).astype(np.float64)
        n    = flat.shape[0]
        mean = flat.mean(axis=0)
        var  = flat.var(axis=0)

        delta         = mean - combined_mean
        new_n         = total_n + n
        combined_mean = (combined_mean * total_n + mean * n) / new_n
        combined_var  = (combined_var * total_n + var * n + delta**2 * total_n * n / new_n) / new_n
        total_n       = new_n

    return combined_mean.astype(np.float32), np.sqrt(combined_var).astype(np.float32)

def build_lam_edges(lat_flat, lon_flat, k=16):
    coords = np.stack([lat_flat, lon_flat], axis=-1)
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k+1)
    indices   = indices[:, 1:]
    distances = distances[:, 1:]

    n_nodes      = len(lat_flat)
    source_nodes = np.repeat(np.arange(n_nodes), k)
    dest_nodes   = indices.reshape(-1)
    edge_index   = np.stack([source_nodes, dest_nodes], axis=0)

    delta_lat     = lat_flat[dest_nodes] - lat_flat[source_nodes]
    delta_lon     = lon_flat[dest_nodes] - lon_flat[source_nodes]
    dist          = distances.reshape(-1)
    edge_features = np.stack([delta_lat, delta_lon, dist], axis=-1)

    return edge_index, edge_features

def build_and_save(config_path='config.yaml'):
    config   = load_config(config_path)
    domain   = config['domain']
    data_dir = f'data/{domain}'
    graph_dir = f'data/{domain}'

    print(f"Domain: {domain}")

    years = get_years(data_dir)
    print(f"Found years: {years}")

    print("Computing mean and std incrementally...")
    mean, std = compute_mean_std(data_dir, years)
    torch.save(torch.tensor(mean, dtype=torch.float32), f'{graph_dir}/mean.pt')
    torch.save(torch.tensor(std,  dtype=torch.float32), f'{graph_dir}/std.pt')
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    print("Normalizing and saving node features year by year...")
    first_year_features, lat_flat, lon_flat = load_year(data_dir, years[0])
    T0, N, V = first_year_features.shape
    total_T = T0 * len(years)

    node_features_mmap = np.lib.format.open_memmap(
        f'{graph_dir}/node_features_mmap.npy',
        mode='w+',
        dtype=np.float32,
        shape=(total_T, N, V)
    )

    offset = 0
    for year in years:
        print(f"  Normalizing {year}...")
        features, _, _ = load_year(data_dir, year)
        normalized = (features - mean) / std
        T = normalized.shape[0]
        node_features_mmap[offset:offset + T] = normalized.astype(np.float32)
        offset += T

    print("Converting memmap to tensor and saving...")
    node_features_tensor = torch.from_numpy(np.array(node_features_mmap))
    torch.save(node_features_tensor, f'{graph_dir}/node_features.pt')
    os.remove(f'{graph_dir}/node_features_mmap.npy')

    torch.save(torch.tensor(lat_flat, dtype=torch.float32), f'{graph_dir}/lat.pt')
    torch.save(torch.tensor(lon_flat, dtype=torch.float32), f'{graph_dir}/lon.pt')

    print("Building graph...")
    if domain == 'lam':
        edge_index, edge_features = build_lam_edges(
            lat_flat, lon_flat, k=config['graph']['k']
        )
        torch.save(torch.tensor(edge_index,    dtype=torch.long),    f'{graph_dir}/edge_index.pt')
        torch.save(torch.tensor(edge_features, dtype=torch.float32), f'{graph_dir}/edge_features.pt')
        print(f"Nodes: {len(lat_flat)}")
        print(f"Edges: {edge_index.shape[1]}")
    else:
        build_icosahedral_graph(
            mesh_level    = config['graph']['mesh_level'],
            grid_lat      = lat_flat,
            grid_lon      = lon_flat,
            output_dir    = graph_dir,
            g2m_angle_deg = config['graph']['g2m_angle_deg']
        )

    print(f"Node features shape: {node_features_tensor.shape}")
    print(f"Saved to {graph_dir}")

if __name__ == "__main__":
    build_and_save()