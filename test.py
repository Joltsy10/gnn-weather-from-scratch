import torch
from model.message_passing import MessagePassingLayer

torch.manual_seed(42)
nodes = torch.randn(100, 7, requires_grad=True)
edge_index = torch.randint(0, 100, (2, 400))
edges = torch.randn(400, 3)

layer = MessagePassingLayer(node_dim=7, edge_dim=3, hidden_dim=64)
out = layer(nodes, nodes, edge_index, edges)

print("Output shape:", out.shape)
print("Output has nan:", torch.isnan(out).any().item())
print("Gradients flow:", out.sum().backward() is None)
print("Node grad exists:", nodes.grad is not None)

# verify different inputs give different outputs
nodes2 = torch.randn(100, 7)
out2 = layer(nodes2, nodes2, edge_index, edges)
print("Different inputs give different outputs:", not torch.allclose(out, out2))