import torch
import numpy as np
import torch.nn.functional as F


def direction_to_canonical(direction):
	direction = F.normalize(direction, dim=-1)
	cos_theta = direction[:, 2]
	phi = torch.atan2(direction[:, 1], direction[:, 0])
	phi += np.pi
	u = (cos_theta + 1) * 0.5
	v = phi / (2*np.pi)
	uv = torch.stack([u, v], dim=1)
	return uv


class EnvironmentMap:
	def __init__(self, n=16):
		self.emission = torch.rand((3, 2 * n, n), requires_grad=True)

	def get_radiance(self, position, direction):
		uv = direction_to_canonical(direction)
		uv = 2 * uv - 1
		env_radiance = F.grid_sample(self.emission[None, ...], uv[None, :, None, ...], align_corners=True)
		env_radiance = env_radiance.permute((0, 2, 3, 1))
		env_radiance = env_radiance.squeeze()
		return env_radiance

# a = EnvironmentMap(n=2)
#
# optimizer = torch.optim.Adam(params=[{'params': a.emission}], lr=0.01, betas=(0.9, 0.999))
# ckpt = torch.load("save_target.tar")
# print(a.emission)
# a.emission.data = ckpt['emission']
# # a.emission.requires_grad = True
# print(a.emission)
#
# for i in range(100):
# 	directions = torch.rand((100, 3))
# 	positions = torch.rand((100, 3))
# 	radiance = a.get_radiance(positions, directions)
# 	loss = torch.norm(radiance, p=1)
#
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
# 	# print(loss, i)
# print(a.emission)
# save_target = {"emission": a.emission}
# torch.save(save_target, "save_target.tar")
