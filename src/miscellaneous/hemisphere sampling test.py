import sys
sys.path.append("../")
import numpy as np
from utils.math_utils import get_direction_from
import matplotlib.pyplot as plt
from nerf_models.microfacet import Microfacet
import torch
import torch.nn.functional as F


def get_hemisphere_samples(N):
	input_array = np.zeros((N * N, 3), dtype=np.float32)
	for i in range(N * N):
		v = get_direction_from(i, (0.5, 0.5), (N, N))
		input_array[i][0] = v[0]
		input_array[i][1] = v[2]
		input_array[i][2] = v[1]

	return input_array


if __name__ == "__main__":
	a = torch.Tensor([0, 0, 0])

	# print(sys.path)
	result = get_hemisphere_samples(16)
	microfacet = Microfacet(f0=0.04)

	for i in range(10):
		pts2l = torch.Tensor(result)
		pts2c = torch.Tensor([0, 1, 0])
		normal = torch.Tensor([0, 1, 0])
		albedo = torch.Tensor([1, 1, 1])
		roughness = torch.Tensor([0.1 * (i + 1)])

		pts2l = pts2l[None, ...]
		pts2c = pts2c[None, ...]
		normal = normal[None, ...]
		albedo = albedo[None, ...]
		roughness = roughness[None, ...]
		l_dot_n = torch.sum(pts2l * normal, dim=-1, keepdim=True)
		#l_dot_n = torch.clip(l_dot_n, 0, 1)

		brdf_d, brdf_s, l_dot_n = microfacet(pts2l, pts2c, normal, albedo, roughness)
		brdf = brdf_d + brdf_s
		brdf = brdf.clone()

		brdf *= l_dot_n
		brdf = brdf[0]

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.scatter(result[:,0], result[:,2], result[:,1], c=brdf[:,0])
		ax.set_xlim3d(-1, 1)
		ax.set_ylim3d(-1, 1)
		ax.set_zlim3d(0, 1)
		plt.show()