import torch
import mcubes
import trimesh


@torch.no_grad()
def query(grid_num, bound, chunk, net_query_fn, net_fn):
	t = torch.linspace(-bound, bound, grid_num + 1)
	query_pts = torch.stack(torch.meshgrid(t, t, t), dim=-1).type('torch.cuda.FloatTensor')
	sh = query_pts.shape
	flat = query_pts.reshape([-1, 3])
	sigma = []
	for i in range(0, flat.shape[0], chunk):
		sigma_chunk = net_query_fn(
			flat[i:i + chunk][..., None, :],
			viewdirs=torch.zeros_like(flat[i:i + chunk]),
			network_fn=net_fn
		)[..., 3]
		sigma.append(sigma_chunk.reshape(-1,))
	sigma = torch.cat(sigma, dim=0)
	sigma = sigma.reshape([*sh[:-1]])
	return sigma


def march_cubes(sigma, grid_num, th):
	vertices, triangles = mcubes.marching_cubes(sigma, th)
	mesh = trimesh.Trimesh(vertices / grid_num, triangles)
	return mesh
