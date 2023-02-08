import torch

A1 = torch.rand((10, 3))
A2 = torch.rand((10, 3))
A3 = torch.rand((10, 3))
A4 = torch.rand((10, 3))
A = torch.stack([A1, A2, A3, A4], 1)
print(A.shape)
indices = torch.randint(0, 4, (10, ))

#print(A)
#print(indices)
print(indices.shape)
#B = A[indices]
B = A[torch.arange(A.size(0)), indices]
print(B)
#C = torch.index_select(A, 1, indices)
#D = torch.gather(A, 1, indices)

print(A.shape)
print(indices.shape)
print(B.shape)