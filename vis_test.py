import numpy as np
from vis_main import WeatherVisualizer
import os

def generate_dummy_global_data(num_nodes=10242):
    """Generates synthetic icosahedral data for testing."""
    # Fibonacci sphere for roughly even distribution
    indices = np.arange(0, num_nodes, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_nodes)
    theta = np.pi * (1 + 5**0.5) * indices
    
    lat = 90 - np.rad2deg(phi)
    lon = np.rad2deg(theta) % 360 - 180
    
    # Create a synthetic weather pattern (e.g., a Rossby wave)
    actual = 280 + 20 * np.sin(3 * np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    # Create prediction with a targeted error cluster over the equator
    pred = actual + 5 * np.exp(-(lat**2 + (lon-50)**2)/400) 
    
    return lat, lon, actual, pred

def main():
    print("Loading test data...")
    lat, lon, actual, pred = generate_dummy_global_data()
    
    # Initialize visualizer without normalization for dummy data
    vis = WeatherVisualizer(
        mean=np.zeros(7), 
        std=np.ones(7), 
        var_names=['u10', 'v10', 'sp', 't850 (K)', 't500', 'z850', 'z500']
    )
    
    print("Generating 3D Global Comparison...")
    fig = vis.plot_global_comparison(
        actual=actual, 
        pred=pred, 
        lat=lat, 
        lon=lon, 
        var_idx=3,
        warp_scale=0.2
    )
    
    # Save to a temporary HTML file and open in the default web browser
    output_file = "render_test.html"
    print(f"Opening visualization in browser: {output_file}")
    
    # Auto_play disables the default plotly logo and modebar for a cleaner look
    fig.write_html(output_file, auto_open=True, config={'displayModeBar': False})

if __name__ == "__main__":
    main()