import numpy as np
from scipy.spatial import cKDTree
import xarray as xr

def load_data(surface_path, pressure_2021_path, pressure_2022_path):
    """
    Load ERA5 files and flatten from (time, lat, lon) to (time, N, 7)
    where N = lat * lon = 15609 grid nodes.
    """
    surface = xr.open_dataset(surface_path)
    pressure = xr.concat([
        xr.open_dataset(pressure_2021_path),
        xr.open_dataset(pressure_2022_path)
    ], dim='valid_time')

    # Number of grid points
    n_lat = len(surface.latitude)
    n_lon = len(surface.longitude)
    N = n_lat * n_lon  # 15609

    # Time steps
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

def build_edges(lat_flat, lon_flat, k=8):
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

def build_and_save(surface_path, pressure_2021_path, pressure_2022_path, 
                   output_dir, k=8):
    print("Loading data...")
    node_features, lat_flat, lon_flat = load_data(
        surface_path, pressure_2021_path, pressure_2022_path
    )

    print("Normalizing...")
    node_features, mean, std = normalize(node_features)
    np.save(f'{output_dir}/mean.npy', mean)
    np.save(f'{output_dir}/std.npy', std)
    
    print("Building edges...")
    edge_index, edge_features = build_edges(lat_flat, lon_flat, k=k)
    
    print(f"Nodes: {len(lat_flat)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge features shape: {edge_features.shape}")
    
    np.save(f'{output_dir}/node_features.npy', node_features)
    np.save(f'{output_dir}/edge_index.npy', edge_index)
    np.save(f'{output_dir}/edge_features.npy', edge_features)
    np.save(f'{output_dir}/lat.npy', lat_flat)
    np.save(f'{output_dir}/lon.npy', lon_flat)
    
    print("Saved.")

if __name__ == "__main__":
    build_and_save(
        surface_path='data/era5_surface.nc',
        pressure_2021_path='data/era5_pressure_2021.nc',
        pressure_2022_path='data/era5_pressure_2022.nc',
        output_dir='data',
        k=16
    )

