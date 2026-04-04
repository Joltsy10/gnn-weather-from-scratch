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
    config    = load_config(config_path)
    domain    = config['domain']
    data_dir  = f'data/{domain}'
    graph_dir = f'data/{domain}'

    years = get_years(data_dir)
    print(f"Found years: {years}")

    print("Pass 1: computing mean...")
    total_n = 0
    mean = np.zeros(7, dtype=np.float64)
    lat_flat, lon_flat = None, None
    for year in years:
        print(f"  {year}")
        features, lf, lonf = load_year(data_dir, year)
        if lat_flat is None:
            lat_flat, lon_flat = lf, lonf
        flat = features.reshape(-1, 7)
        n    = flat.shape[0]
        mean = (mean * total_n + flat.astype(np.float64).sum(axis=0)) / (total_n + n)
        total_n += n
        del features, flat

    mean_f32 = mean.astype(np.float32)

    print("Pass 2: computing std...")
    var = np.zeros(7, dtype=np.float32)
    for year in years:
        print(f"  {year}")
        features, _, _ = load_year(data_dir, year)
        flat = features.reshape(-1, 7)
        flat -= mean_f32
        flat **= 2
        var += flat.sum(axis=0)
        del features, flat
    std = np.sqrt(var / total_n)

    std_f32  = np.maximum(std.astype(np.float32), 1e-6)

    torch.save(torch.tensor(mean_f32), f'{graph_dir}/mean.pt')
    torch.save(torch.tensor(std_f32),  f'{graph_dir}/std.pt')
    print(f"Mean: {mean_f32}")
    print(f"Std:  {std_f32}")

    print("Pass 3: normalizing and saving...")
    year_lengths = []
    for year in years:
        surface = xr.open_dataset(f'{data_dir}/era5_surface_{year}.nc')
        year_lengths.append(len(surface.valid_time))
        surface.close()
    total_T = sum(year_lengths)
    N, V = 65160, 7

    mmap_path = f'{graph_dir}/node_features.npy'
    node_features_mmap = np.lib.format.open_memmap(
        mmap_path, mode='w+', dtype=np.float32, shape=(total_T, N, V)
    )

    offset = 0
    for year, T in zip(years, year_lengths):
        print(f"  {year} ({T} timesteps)")
        features, _, _ = load_year(data_dir, year)
        features -= mean_f32
        features /= std_f32
        node_features_mmap[offset:offset + T] = features
        node_features_mmap.flush()
        offset += T
        del features

    print(f"Node features shape: {node_features_mmap.shape}")

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

    print(f"Saved to {graph_dir}")

if __name__ == "__main__":
    build_and_save()