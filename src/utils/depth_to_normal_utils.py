import sys
sys.path.append('../')

import torch
import numpy as np
import torch.nn.functional as F

# Ray helpers
def depth_to_position(H, W, K, c2w, d):
	i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
	i = i.t()
	j = j.t()
	dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
	dirs = F.normalize(dirs, dim=-1)
	# Rotate ray directions from camera frame to the world frame
	rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
	# Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_p = c2w[:3,-1] + rays_d * d[...,None]
	return rays_p


def normalize(x):
	return x / np.linalg.norm(x, axis=-1, keepdims=True)


def depth_to_normal_image_space(depth_map, pose, K):
	height, width = depth_map.shape
	position = depth_to_position(height, width, K, pose, depth_map)

	position = position.cpu().detach().numpy()
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
	return torch.Tensor(vc)


from dataset.dataset_interface import load_dataset
import imageio
import os

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def depth_to_normal_replica(scene, split):
	# basedir = os.path.join("../../data/replica", scene, split)
	# depth_image_path = os.path.join(basedir, ('depth{:06d}.png').format(i))
	# depth_image = cv2.imread(depth_image_path, -1)
	# with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
	# 	meta = json.load(fp)
	#
	# camera_angle_x = float(meta['camera_angle_x']) / 180.0 * math.pi
	# height = depth_image.shape[0]
	# width = depth_image.shape[1]
	# focal = .5 * width / np.tan(0.5 * camera_angle_x)
	#
	# K = np.array([
	# 	[focal, 0, 0.5 * width],
	# 	[0, focal, 0.5 * height],
	# 	[0, 0, 1]
	# ]).astype(np.float32)

	scene_dataset = load_dataset("replica", "../../data/replica/%s" % scene, split=split, load_depth=True, load_image=False)
	scene_dataset.load_all_data(1)
	scene_dataset.to_tensor("cpu")
	for i in range(100):
		K = scene_dataset.get_focal_matrix()
		normal = depth_to_normal_image_space(scene_dataset.depths[i], scene_dataset.poses[i], K)
		normal = (normal + 1) * 0.5
		normal = normal.cpu().numpy()
		result_image_8bit = to8b(normal)
		filename = os.path.join("../../data/replica", scene, split, ('normal{:06d}.png').format(i))
		imageio.imwrite(filename, result_image_8bit)


if __name__ == "__main__":
	path = os.path.join("../../data/replica")
	exp_names = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
	exp_names.sort()
	splits = ["train", "val", "test"]
	for exp in exp_names:
		for split in splits:
			print(exp, split)
			depth_to_normal_replica(exp, split)
#depth_to_normal_replica("office_0", "train")


# def depth_to_normal_image_space(depth_map, pose, K):
# 	height, width = depth_map.shape
# 	position = depth_to_position(height, width, K, pose, depth_map)
# 	normal = torch.zeros((*depth_map.shape, 3))
#
# 	def get_value(x, y):
# 		n_i = np.clip(x, 0, width - 1)
# 		n_j = np.clip(y, 0, height - 1)
# 		return position[n_j, n_i, :]
#
# 	for i in range(width):
# 		for j in range(height):
# 			s01 = get_value(i - 1, j)
# 			s21 = get_value(i + 1, j)
# 			s10 = get_value(i, j - 1)
# 			s12 = get_value(i, j + 1)
#
# 			va = s21 - s01
# 			vb = s12 - s10
#
# 			va = F.normalize(va, dim=-1)
# 			vb = F.normalize(vb, dim=-1)
#
# 			vc = torch.cross(vb, va)
# 			vc = F.normalize(vc, dim=-1)
# 			normal[j,i,:] = vc
# 	return normal
