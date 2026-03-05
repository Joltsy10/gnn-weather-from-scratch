import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

node_features = torch.tensor(
        np.load('data/node_features.npy'), dtype=torch.float32
    ) # (T, N, 7)

node_features = node_features[:2688]
T = node_features.shape[0]

loss_fn = nn.MSELoss()
total_loss = 0

for t in range(T-1):
    loss = loss_fn(node_features[t], node_features[t+1])

    total_loss += loss

avg_loss = total_loss/(T - 1)
print(avg_loss)
