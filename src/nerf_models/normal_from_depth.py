import torch
import torch.nn.functional as F


def raw2depth(raw, dists, z_vals):
	"""
	Raw output to depth value
	"""
	raw2sigma = lambda raw_, dists_, act_fn=F.relu: 1. - torch.exp(-act_fn(raw_) * dists_)
	sigma = raw2sigma(raw, dists)
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:,:-1]
	depth_map = torch.sum(weights * z_vals, -1)
	return depth_map


def get_normal_from_depth_gradient_direction(rays_o, rays_d, network_query_fn, network_fn, z_vals):
	"""
	Calculate normal from depth gradient w.r.t direction.
	"""
	up = torch.Tensor([0, 1, 0])
	up = up.repeat(*rays_d.shape[:-1], 1)

	right = torch.cross(rays_d, up, dim=-1)
	up = torch.cross(right, rays_d)

	a = torch.zeros(*rays_d.shape[:-1], 1)
	b = torch.zeros(*rays_d.shape[:-1], 1)
	a.requires_grad = True
	b.requires_grad = True

	new_d = a * right + b * up + torch.sqrt(1- a*a - b*b) * rays_d
	# new_d = torch.stack([a, b, torch.sqrt(1- a*a - b*b)], dim=-1)
	# new_x = rays_o + right * a + up * b
	pts = rays_o[..., None, :] + new_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

	raw = network_query_fn(pts, None, network_fn)
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
	sigma = raw2sigma(raw[..., 0], dists)
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	depth_map = torch.sum(weights * z_vals, -1)

	depth_map.backward(torch.ones_like(depth_map))

	dx = a.grad
	dy = b.grad

	grad = right * dx + up * dy
	normal = F.normalize(grad - rays_d, dim=-1)
	return normal


def get_normal_from_depth_gradient_direction_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=0.01):
	"""
	Calculate normal from numerical depth gradient w.r.t direction.
	"""
	up = torch.Tensor([0, 1, 0])
	up = up.repeat(*rays_d.shape[:-1], 1)

	right = torch.cross(rays_d, up, dim=-1)
	up = torch.cross(right, rays_d)

	new_d_right = F.normalize(rays_d + epsilon * right, dim=-1)
	new_d_left = F.normalize(rays_d - epsilon * right, dim=-1)
	new_d_up = F.normalize(rays_d + epsilon * up, dim=-1)
	new_d_down = F.normalize(rays_d - epsilon * up, dim=-1)

	new_d = torch.cat([new_d_right, new_d_left, new_d_up, new_d_down], dim=0)
	new_o = torch.cat([rays_o] * 4, dim=0)
	new_z_vals = torch.cat([z_vals] * 4, dim=0)

	pts = new_o[..., None, :] + new_d[..., None, :] * new_z_vals[..., :, None]  # [N_rays, N_samples, 3]

	raw = network_query_fn(pts, None, network_fn)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	raw_right, raw_left, raw_up, raw_down = torch.split(raw, [rays_o.shape[0]] * 4, dim=0)

	depth_right = raw2depth(raw_right[..., 0], dists, z_vals)
	depth_left = raw2depth(raw_left[..., 0], dists, z_vals)
	depth_up = raw2depth(raw_up[..., 0], dists, z_vals)
	depth_down = raw2depth(raw_down[..., 0], dists, z_vals)

	pos_right = rays_o + depth_right[...,None] * new_d_right
	pos_left = rays_o + depth_left[...,None] * new_d_left
	pos_up = rays_o + depth_up[...,None] * new_d_up
	pos_down = rays_o + depth_down[...,None] * new_d_down

	dx = pos_right - pos_left
	dy = pos_up - pos_down

	normal = torch.cross(dx, dy, dim=-1)
	normal = F.normalize(normal, dim=-1)
	return normal


def get_normal_from_depth_gradient(rays_o, rays_d, network_query_fn, network_fn, z_vals):
	"""
	Calculate normal from depth gradient w.r.t position.
	"""
	up = torch.Tensor([0, 1, 0])
	up = up.repeat(*rays_d.shape[:-1], 1)

	right = torch.cross(rays_d, up, dim=-1)
	up = torch.cross(right, rays_d)

	a = torch.zeros(*rays_d.shape[:-1], 1)
	b = torch.zeros(*rays_d.shape[:-1], 1)
	a.requires_grad = True
	b.requires_grad = True

	new_x = rays_o + right * a + up * b
	pts = new_x[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

	raw = network_query_fn(pts, None, network_fn)
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
	sigma = raw2sigma(raw[..., 0], dists)
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	depth_map = torch.sum(weights * z_vals, -1)

	depth_map.backward(torch.ones_like(depth_map))

	dx = a.grad
	dy = b.grad

	grad = right * dx + up * dy
	normal = F.normalize(grad - rays_d, dim=-1)
	return normal


def get_normal_from_depth_gradient_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=0.01):
	"""
	Calculate normal from numerical depth gradient w.r.t position.
	"""
	up = torch.Tensor([0, 1, 0])
	up = up.repeat(*rays_d.shape[:-1], 1)

	right = torch.cross(rays_d, up, dim=-1)
	up = torch.cross(right, rays_d)

	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

	new_x_right = pts + epsilon * right[..., None, :]
	new_x_left = pts - epsilon * right[..., None, :]
	new_x_up = pts + epsilon * up[..., None, :]
	new_x_down = pts - epsilon * up[..., None, :]

	new_pts = torch.cat([new_x_right, new_x_left, new_x_up, new_x_down], dim=0)

	raw = network_query_fn(new_pts, None, network_fn)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	def raw2depth(raw):
		raw2sigma = lambda raw_, dists_, act_fn=F.relu: 1. - torch.exp(-act_fn(raw_) * dists_)
		sigma = raw2sigma(raw, dists)
		weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
		depth_map = torch.sum(weights * z_vals, -1)
		return depth_map

	raw_right, raw_left, raw_up, raw_down = torch.split(raw, [rays_o.shape[0]] * 4, dim=0)

	depth_right = raw2depth(raw_right[..., 0])
	depth_left = raw2depth(raw_left[..., 0])
	depth_up = raw2depth(raw_up[..., 0])
	depth_down = raw2depth(raw_down[..., 0])

	dx = 2 * epsilon * right + (depth_right[...,None] - depth_left[...,None]) * rays_d
	dy = 2 * epsilon * up + (depth_up[...,None] - depth_down[...,None]) * rays_d

	normal = torch.cross(dx, dy, dim=-1)
	normal = F.normalize(normal, dim=-1)
	return normal

# def get_normal_from_depth_gradient_simple(rays_o, rays_d, network_query_fn, network_fn, z_vals):
# 	rays_o.requires_grad=True
# 	raw = network_query_fn(rays_o[..., None, :], None, network_fn)
# 	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
# 	dists = z_vals[..., 1:] - z_vals[..., :-1]
# 	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
# 	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
# 	sigma = raw2sigma(raw[..., 0], dists)
# 	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
# 	depth_map = torch.sum(weights * z_vals, -1)
#
# 	depth_map.backward(torch.ones_like(depth_map))
#
# 	normal = F.normalize(rays_o.grad, dim=-1)
# 	rays_o.requires_grad = False
# 	return normal