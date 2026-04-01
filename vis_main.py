import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class WeatherVisualizer:
    def __init__(self, mean=None, std=None, var_names=None):
        """
        Initializes the visualizer with normalization statistics.
        """
        self.mean = mean
        self.std = std
        self.var_names = var_names or ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']
        
        # Professional dark theme styling
        self.layout_template = "plotly_dark"
        self.lighting_props = dict(
            ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.1, fresnel=0.2
        )

    def _denormalize(self, data, var_idx):
        if self.mean is not None and self.std is not None:
            return data * self.std[var_idx] + self.mean[var_idx]
        return data

    def _create_ghost_globe(self, resolution=50):
        """Creates the translucent R=1 reference sphere."""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        return go.Surface(
            x=x, y=y, z=z,
            surfacecolor=np.zeros_like(z),
            colorscale=[[0, 'rgba(40, 50, 70, 0.3)'], [1, 'rgba(40, 50, 70, 0.3)']],
            showscale=False, hoverinfo='skip',
            lighting=dict(ambient=0.8, diffuse=0.2, specular=0.1)
        )

    def _mesh_to_surface(self, values, lat, lon, warp_scale, resolution=100):
        """Interpolates unstructured icosahedral nodes to a structured grid for surface rendering."""
        # Create regular lat/lon grid
        grid_lat, grid_lon = np.mgrid[-90:90:complex(resolution), -180:180:complex(resolution)]
        
        # Interpolate values
        grid_val = griddata((lat, lon), values, (grid_lat, grid_lon), method='linear', fill_value=np.nanmin(values))
        
        # Spherical coordinates mapping
        phi = np.deg2rad(90 - grid_lat)
        theta = np.deg2rad(grid_lon)
        
        # Calculate warp displacement
        norm_val = (grid_val - np.nanmin(grid_val)) / (np.nanmax(grid_val) - np.nanmin(grid_val) + 1e-8)
        r_warp = 1 + (norm_val * warp_scale)
        
        x = r_warp * np.sin(phi) * np.cos(theta)
        y = r_warp * np.sin(phi) * np.sin(theta)
        z = r_warp * np.cos(phi)
        
        return x, y, z, grid_val

    def plot_global_comparison(self, actual, pred, lat, lon, var_idx=3, warp_scale=0.15):
        """
        Creates a dual-viewport synchronized comparison of Actual vs Predicted on the 3D globe.
        """
        var_name = self.var_names[var_idx]
        actual_denorm = self._denormalize(actual, var_idx)
        pred_denorm = self._denormalize(pred, var_idx)

        # Global min/max to ensure colorscales match perfectly across both plots
        cmin = min(np.nanmin(actual_denorm), np.nanmin(pred_denorm))
        cmax = max(np.nanmax(actual_denorm), np.nanmax(pred_denorm))

        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=(f"Actual {var_name}", f"Predicted {var_name}")
        )

        for i, data in enumerate([actual_denorm, pred_denorm]):
            col = i + 1
            x, y, z, grid_val = self._mesh_to_surface(data, lat, lon, warp_scale)
            
            # 1. Add Ghost Globe
            fig.add_trace(self._create_ghost_globe(), row=1, col=col)
            
            # 2. Add Warped Data Surface
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=grid_val,
                cmin=cmin, cmax=cmax,
                colorscale='Inferno', # High contrast scientific scale
                colorbar=dict(title=var_name, x=1.05) if i == 1 else None,
                showscale=(i == 1),
                lighting=self.lighting_props,
                hovertemplate="Val: %{surfacecolor:.2f}<extra></extra>"
            ), row=1, col=col)

        # Link camera controls for synchronized rotation
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5))
        fig.update_layout(
            template=self.layout_template,
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, camera=camera),
            scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, camera=camera),
            margin=dict(l=0, r=0, b=0, t=40),
            title_text=f"Global Autoregressive Rollout Comparison: {var_name}"
        )
        return fig

    def plot_lam_grid(self, actual_grid, pred_grid, lat_grid, lon_grid, var_idx=3, extent=[68, 98, 6, 38]):
        """Standard 2D Cartopy plot for regional (LAM) data."""
        var_name = self.var_names[var_idx]
        actual_denorm = self._denormalize(actual_grid, var_idx)
        pred_denorm = self._denormalize(pred_grid, var_idx)
        
        cmin = min(np.nanmin(actual_denorm), np.nanmin(pred_denorm))
        cmax = max(np.nanmax(actual_denorm), np.nanmax(pred_denorm))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        for i, (data, title) in enumerate(zip([actual_denorm, pred_denorm], ['Actual', 'Predicted'])):
            ax = axes[i]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7)
            
            im = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(), 
                               cmap='RdBu_r', vmin=cmin, vmax=cmax)
            ax.set_title(f"{title} {var_name}")
            
        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=var_name)
        return fig