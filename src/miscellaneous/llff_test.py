import numpy as np
import os
from utils.image_utils import *
import imageio

def normalize(x):
	return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
	vec2 = normalize(z)
	vec1_avg = up
	vec0 = normalize(np.cross(vec1_avg, vec2))
	vec1 = normalize(np.cross(vec2, vec0))
	m = np.stack([vec0, vec1, vec2, pos], 1)
	return m

def poses_avg(poses):
	hwf = poses[0, :3, -1:]

	center = poses[:, :3, 3].mean(0)
	vec2 = normalize(poses[:, :3, 2].sum(0))
	up = poses[:, :3, 1].sum(0)
	c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

	return c2w

def recenter_poses(poses):
	poses_ = poses+0
	bottom = np.reshape([0,0,0,1.], [1,4])
	c2w = poses_avg(poses)
	c2w = np.concatenate([c2w[:3,:4], bottom], -2)
	bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
	poses = np.concatenate([poses[:,:3,:4], bottom], -2)

	poses = np.linalg.inv(c2w) @ poses
	poses_[:,:3,:4] = poses[:,:3,:4]
	poses = poses_
	return poses


path = "../../data/nerf_llff_data/fern/"
image_path = path + "images_8/image000.png"
sh = imageio.imread(image_path).shape
poses_arr = np.load(path+"poses_bounds.npy")
poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
print(poses_arr.shape)
print(poses.shape)

poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
poses[2, 4, :] = poses[2, 4, :]
print(poses.shape)
print(sh)
print(poses)