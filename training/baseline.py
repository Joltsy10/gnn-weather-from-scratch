import torch
import numpy as np
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

config   = load_config()
domain   = config['domain']
data_dir = f'data/{domain}'

node_features = torch.load(f'{data_dir}/node_features.pt')
mean = torch.load(f'{data_dir}/mean.pt').numpy()
std  = torch.load(f'{data_dir}/std.pt').numpy()

test = node_features[4032:]
K = 1

var_names = ['u10', 'v10', 'sp', 't850', 't500', 'z850', 'z500']

# Normalized persistence MAE per step
mae_per_step = np.zeros(K)
for t in range(len(test) - K):
    for k in range(K):
        mae_per_step[k] += torch.mean(torch.abs(test[t] - test[t + k + 1])).item()
mae_per_step /= (len(test) - K)

print(f"Domain: {domain}")
print(f"Test timesteps: {len(test)}")
print(f"\nPersistence baseline (normalized MAE):")
for k in range(K):
    print(f"  T+{k+1} ({(k+1)*6}h): {mae_per_step[k]:.6f}")

# Per-variable persistence MAE in physical units at T+1
print(f"\nPersistence per-variable MAE at T+1 (physical units):")
units = ['m/s', 'm/s', 'Pa', 'K', 'K', 'm²/s²', 'm²/s²']
for i, (name, unit) in enumerate(zip(var_names, units)):
    mae_norm = torch.mean(torch.abs(test[:-1, :, i] - test[1:, :, i])).item()
    mae_phys = mae_norm * std[i]
    print(f"  {name}: {mae_phys:.4f} {unit} (normalized: {mae_norm:.4f})")