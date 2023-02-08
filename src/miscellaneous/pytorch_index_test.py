import torch

sim = torch.rand(4,5,4)
a = sim[[0, 1], [3]]

print(sim.size())
print(a.size())

#print(sim)
#print(a)