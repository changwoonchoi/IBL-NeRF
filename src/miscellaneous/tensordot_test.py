import torch

cos_theta = torch.zeros((512, 256))
f0 = torch.zeros((512, 3))

cos_theta = torch.stack([cos_theta] * 3, dim=-1)
print(cos_theta.shape)
f0 = f0[:,None,:]


f = f0 + (1 - f0) * (1 - cos_theta) ** 5
print(f.shape)

#c = torch.tensordot(a, b, dims=[[0], [0]])
#print(c.shape)
