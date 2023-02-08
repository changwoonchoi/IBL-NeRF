import torch
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt


def normalize(x):
	return x / np.linalg.norm(x, axis=-1, keepdims=True)


def depth_to_position(H, W, K, c2w, d):
	i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
	i = i.t()
	j = j.t()
	dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
	dirs = F.normalize(dirs, dim=-1)
	# Rotate ray directions from camera frame to the world frame
	rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
	# Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_p = c2w[:3,-1] + rays_d.numpy() * d
	return rays_p


def position_to_projection(K, c2w, position):
	cam_pos = c2w[:3, -1]
	view_matrix = c2w[:3, :3].transpose()
	position_camera = np.sum((position - cam_pos)[..., None, :] * view_matrix[:3, :3], -1)
	proj_u = -K[0][0] * position_camera[..., 0] / position_camera[..., 2]
	proj_v = -K[1][1] * position_camera[..., 1] / position_camera[..., 2]


def position_to_normal(position):
	position_padded = np.pad(position, ((1, 1),(1, 1),(0, 0)), 'edge')

	left = position_padded[1:-1,:-2,:]
	right = position_padded[1:-1, 2:, :]
	up = position_padded[:-2, 1:-1, :]
	bottom = position_padded[2:, 1:-1, :]

	va = right - left
	vb = bottom - up

	va = normalize(va)
	vb = normalize(vb)

	vc = np.cross(vb, va, axis=-1)
	vc = normalize(vc)

	return vc


def position_to_normal_torch(position):
	position = torch.Tensor(position)
	print(position.shape, "BEFORE PADDING")
	position_padded = F.pad(position, (0, 0, 1, 1, 1, 1), 'constant')
	print(position_padded.shape, "PADDING")

	left = position_padded[1:-1,:-2,:]
	right = position_padded[1:-1, 2:, :]
	up = position_padded[:-2, 1:-1, :]
	bottom = position_padded[2:, 1:-1, :]

	va = right - left
	vb = bottom - up

	va = F.normalize(va, dim=-1)
	vb = F.normalize(vb, dim=-1)

	vc = torch.cross(vb, va, dim=-1)
	vc = F.normalize(vc, dim=-1)
	vc = vc.numpy()
	return vc


def get_TBN(normal):
	binormal = np.zeros_like(normal)
	binormal[..., 0] = np.where(normal[...,0] > normal[...,2], -normal[..., 1], 0)
	binormal[..., 1] = np.where(normal[...,0] > normal[...,2], normal[..., 0], -normal[...,2])
	binormal[..., 2] = np.where(normal[..., 0] > normal[..., 2], 0, normal[..., 1])
	binormal = binormal / np.linalg.norm(binormal, axis=-1, keepdims=True)
	tangent = np.cross(binormal, normal, axis=-1)

	return binormal, tangent

from scipy.special import comb
def smoothstep(edge0, edge1, x):
	t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)

import random
def cosine_sample_hemisphere():
	u1 = random.random()
	u2 = random.random()
	r = np.sqrt(u1)
	phi = 2 * np.pi * u2
	x = r * np.cos(phi)
	y = r * np.sin(phi)
	z = np.sqrt(max(0, 1-x*x-y*y))
	return np.array([x, y, z], dtype=np.float32)


def depth_to_ssao(depth_path, pose, camera_angle_x, is_disp=False, irradiance_path = None):
	# disp = cv2.imread(depth_path)
	# disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
	# disp = np.asarray(disp, dtype=np.float32) / 255.0
	# height, width, channel = disp.shape
	# normal = np.zeros_like(disp)

	if is_disp:
		disp = cv2.imread(depth_path)
		disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
		disp = np.asarray(disp, dtype=np.float32) / 255.0
		depth = 1.0 / (disp[:, :, 0:1] + 1e-10)
	else:
		depth = np.load(depth_path)
		depth = depth[..., None]

	height, width, _ = depth.shape
	focal = .5 * width / np.tan(0.5 * camera_angle_x)
	K = np.array([
		[focal, 0, 0.5 * width],
		[0, focal, 0.5 * height],
		[0, 0, 1]
	]).astype(np.float32)

	print(depth.shape)
	position = depth_to_position(height, width, K, pose, depth)
	#position = np.load("../../data_temp/result_20211223/seg_nerf_dataset_scale_4/kitchen/train/1_position.npy")

	# position_to_projection(K, pose, position)


	normal = position_to_normal_torch(position)
	binormal, tangent = get_TBN(normal)
	TBNs = np.stack([tangent, binormal, normal], axis=-1)
	#TBNs = np.transpose(TBNs, [0, 1, 3, 2])
	kernel_size = 64
	#samples = []
	#for i in range(kernel_size):
	#	samples.append(cosine_sample_hemisphere())
	samples = get_ssao_samples(kernel_size)

	print(samples.shape)
	print(TBNs.shape)

	radius = 0.5
	dirs = np.tensordot(TBNs, samples, axes=([3, 1]))

	samplePos = position[..., None] + radius * dirs
	samplePos = np.transpose(samplePos, (0, 1, 3, 2))
	#samplePos = position

	print(samplePos.shape, "SamplePos")


	cam_pos = pose[:3, -1]
	view_matrix = pose[:3, :3].transpose()
	position_camera = np.sum((samplePos - cam_pos)[..., None, :] * view_matrix[:3, :3], -1)
	print(position_camera.shape, "position_camera")
	proj_u = -K[0][0] * position_camera[..., 0] / (K[0][2] * position_camera[..., 2])
	proj_v = K[1][1] * position_camera[..., 1] / (K[1][2] * position_camera[..., 2])

	print(np.max(proj_u), "x MAX")
	print(np.min(proj_u), "x MIN")
	print(np.max(proj_v), "y MAX")
	print(np.min(proj_v), "y MIN")

	#plt.imshow(proj_u)
	#plt.show()

	proj_uv = np.stack([proj_u, proj_v], axis=-1)
	proj_uv = np.transpose(proj_uv, [2, 0, 1, 3])
	print(proj_uv.shape)

	input = np.transpose(position, [2, 0, 1])
	print(input.shape)

	input = torch.Tensor(input)
	input = input.expand((proj_uv.shape[0], *input.shape))

	proj_uv = torch.Tensor(proj_uv)

	print(input.shape, "INPUT")
	print(proj_uv.shape, "proj_uv")
	result = F.grid_sample(input, proj_uv, align_corners=True, padding_mode="border")
	result = result.numpy()
	result = result.transpose([2, 3, 0, 1])
	sampleDepth = -result[...,2]
	samplePos_z = -samplePos[...,2]
	print(sampleDepth.shape)
	print(samplePos_z.shape)
	print("SSAO")
	rangeCheck = smoothstep(0.0, 1.0, radius / (abs(sampleDepth - samplePos_z) + 1e-10))

	occlusion = (samplePos_z >= sampleDepth + 0.025) * rangeCheck
	occlusion = occlusion.astype(np.int32)
	occlusion = np.mean(occlusion, axis=-1)
	occlusion = 1 - occlusion
	ssao = np.stack((occlusion,) * 3, axis=-1)
	depth = np.stack((depth[...,0],) * 3, axis=-1)

	plt.imshow(1 / (depth + 1))
	plt.figure()
	plt.imshow((normal + 1) * 0.5)
	plt.figure()
	plt.imshow(ssao)


	if irradiance_path is not None:
		irradiance = cv2.imread(irradiance_path)
		irradiance = cv2.cvtColor(irradiance, cv2.COLOR_BGR2RGB)
		irradiance = np.asarray(irradiance, dtype=np.float32) / 255.0
		plt.figure()
		plt.imshow(irradiance)
	plt.show()
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.scatter(samples[:,0], samples[:,1], samples[:,2])
	# plt.show()


def lerp(a, b, f):
	return a + f * (b-a)


def get_ssao_samples(n=64):
	samples = []
	for i in range(n):
		sample = np.array([
			random.random() * 2 - 1,
			random.random() * 2 - 1,
			random.random()
		])
		sample = normalize(sample)
		scale = i / n
		scale = lerp(0.1, 1, scale * scale)
		sample *= scale
		# sample = np.array([0, 0, 1])
		samples.append(sample)
	return np.asarray(samples)

import os
import json
if __name__ == "__main__":
	target = "kitchen"
	basedir = '../../data/mitsuba/%s' % target
	with open(os.path.join(basedir, 'transforms_test.json'), 'r') as fp:
		meta = json.load(fp)
	skip = 10
	poses = []
	print(len(meta['frames']))
	for frame in meta['frames']:
		# (3) load pose information
		pose = np.array(frame['transform']).astype(np.float32)
		#print(pose)
		# Mitsuba --> camera forward is +Z !!
		pose[:3, 0] *= -1
		pose[:3, 2] *= -1
		poses.append(pose)
	camera_angle_x = float(meta['frames'][0]['fov_degree']) / 180.0 * np.pi

	for i in range(10):
		#path = "../../data/mitsuba/%s/train/%d_depth.npy" % (target, i)
		#irradiance_path = "../../data/mitsuba/%s/train/%d_irradiance.png" % (target, i+1)
		#depth_to_ssao(path, poses[i], camera_angle_x, False, irradiance_path)

		# path = "../../logs/specular_ibl_no_normalize/%s/infer_normal/testset_100000/disp_00%d.png" % (target, i)
		#path = "../../logs/specular_ibl/%s/not_infer_normal/testset_015000/disp_00%d.png" % (
		# target, i)
		path = "../../logs/specular_ibl_from_gt_albedo/%s/not_infer_normal/testset_200000/disp_00%d.png" % (target, i)
		irradiance_path = "../../logs/specular_ibl_from_gt_albedo/%s/not_infer_normal/testset_200000/irradiance_00%d.png" % (target, i)
		depth_to_ssao(path, poses[i], camera_angle_x, True, irradiance_path)
