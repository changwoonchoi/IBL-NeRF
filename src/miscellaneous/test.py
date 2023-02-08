# import torch
#
# a = torch.rand((4, 4, 2))
# b = torch.rand((5, 2))
#
# a2 = torch.unsqueeze(a, -2)
# print(a2.shape)
# a3 = torch.cat(5*[a2], dim=-2)
# print(a3.shape)
#
# diff = a3 - b
# print(diff.shape)
#
# diff_norm = torch.linalg.norm(diff, dim=-1)
# print(diff_norm.shape)
#
# index = torch.argmin(diff_norm, dim=-1)
# print(index.shape)

a = {}
new_a = {}
a["k"] = 1

for k, v in a.items():
	new_a[k+"_0"] = a[k]
print(new_a)