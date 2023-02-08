import torch
import numpy as np
import math


trans_t = lambda t: torch.Tensor([
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, t],
	[0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
	[1, 0, 0, 0],
	[0, np.cos(phi), -np.sin(phi), 0],
	[0, np.sin(phi), np.cos(phi), 0],
	[0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
	[np.cos(th), 0, -np.sin(th), 0],
	[0, 1, 0, 0],
	[np.sin(th), 0, np.cos(th), 0],
	[0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
	c2w = trans_t(radius)
	c2w = rot_phi(phi / 180. * np.pi) @ c2w
	c2w = rot_theta(theta / 180. * np.pi) @ c2w
	c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
	return c2w


def mapUVToDirection(uv, flipy=False):
	x = 2 * uv[0] - 1
	y = 2 * uv[1] - 1
	if (y > -x):
		if (y < x):
			xx = x
			if (y > 0):
				offset = 0
				yy = y
			else:
				offset = 7
				yy = x + y
		else:
			xx = y
			if (x > 0):
				offset = 1
				yy = y - x
			else:
				offset = 2
				yy = -x
	else:
		if (y > x):
			xx = -x
			if (y > 0):
				offset = 3
				yy = -x - y
			else:
				offset = 4
				yy = -y
		else:
			xx = -y
			if (x > 0):
				offset = 6
				yy = x
			else:
				if y != 0:
					offset = 5
					yy = x - y
				else:
					return (0, 1, 0)
	assert xx >= 0
	theta = math.acos(max(min(1 - xx * xx, 1), -1))
	phi = (math.pi / 4) * (offset + (yy / xx))
	if flipy:
		ay = - math.cos(theta)
	else:
		ay = math.cos(theta)
	return (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), ay)


def mapDirectionToUV(direction):
	M_PIf = math.pi
	Q_PIf = M_PIf / 4
	x = direction[0]
	y = -direction[2]
	theta = math.acos(abs(direction[1]))
	phi = math.atan2(y, x)
	phi += (2 * M_PIf)
	phi = phi % (2 * M_PIf)

	xx = math.sqrt(1 - math.cos(theta))
	offset = int(phi / Q_PIf)
	yy = phi / Q_PIf - float(offset)

	assert yy >= 0
	yy = yy * xx

	if (y > -x):
		if (y < x):
			u = xx
			if (y > 0):
				v = yy
			else:
				v = yy - u
		else:
			v = xx
			if (x > 0):
				u = v - yy
			else:
				u = -yy
	else:
		if (y > x):
			u = -xx
			if (y > 0):
				v = -u - yy
			else:
				v = -yy
		else:
			v = -xx
			if (x > 0):
				u = yy
			else:
				u = yy + v

	u = 0.5 * u + 0.5
	v = 0.5 * v + 0.5
	return (u, v)


def get_direction_from(index, offset, size):
	sx, sy = size
	u_index = (index // sy)
	v_index = (index % sy)
	inverted = False
	if u_index > sx:
		u_index -= sx
		inverted = True

	u_index_r = (float(u_index) + offset[0]) / (float(sx))
	v_index_r = (float(v_index) + offset[1]) / (float(sy))
	rx, ry, rz = mapUVToDirection((u_index_r, v_index_r))
	if inverted:
		return rx, ry, -rz
	else:
		return rx, ry, rz


def get_hemisphere_samples(N):
	input_array = np.zeros((N * N, 3), dtype=np.float32)
	for i in range(N * N):
		v = get_direction_from(i, (0.5, 0.5), (N, N))
		input_array[i][0] = v[0]
		input_array[i][1] = v[1]
		input_array[i][2] = v[2]

	return input_array


import random
def get_hemisphere_samples_random(N):
	input_array = np.zeros((N * N, 3), dtype=np.float32)
	for i in range(N * N):
		v = get_direction_from(i, (random.random(), random.random()), (N, N))
		input_array[i][0] = v[0]
		input_array[i][1] = v[1]
		input_array[i][2] = v[2]

	return input_array


def get_uniform_hemisphere_samples(N):

	random_us = torch.rand(N, 2)

	z = random_us[..., 0]
	r = torch.sqrt(torch.clip(1 - z * z, 0, 1))
	phi = 2 * np.pi * random_us[..., 1]
	hemisphere_samples = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)

	return hemisphere_samples

import torch.nn.functional as F


def get_TBN(normal):
	binormal = torch.zeros_like(normal).float()
	#binormal[..., 0] = -normal[..., 1]
	#binormal[..., 1] = normal[..., 0]
	condition = normal[..., 0] > normal[..., 2]

	binormal[..., 0] = torch.where(condition, -normal[..., 1], binormal[..., 0])
	binormal[..., 1] = torch.where(condition, normal[..., 0], -normal[..., 2])
	binormal[..., 2] = torch.where(condition, binormal[..., 2], normal[..., 1])

	binormal = F.normalize(binormal, dim=-1)
	tangent = torch.cross(binormal, normal, dim=-1)
	return binormal, tangent
