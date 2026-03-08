import torch
import numpy as np

node_features = torch.tensor(
    np.load('data/node_features.npy'), dtype=torch.float32
)

test = node_features[4032:]
K = 4

mae_per_step = np.zeros(K)

for t in range(len(test) - K):
    for k in range(K):
        mae_per_step[k] += torch.mean(torch.abs(test[t] - test[t + k + 1])).item()

mae_per_step /= (len(test) - K)

for k in range(K):
    print(f"T+{k+1} ({(k+1)*6}h): {mae_per_step[k]:.6f}")

print(len(test))