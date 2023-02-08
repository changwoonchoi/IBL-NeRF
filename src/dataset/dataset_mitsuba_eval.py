from abc import ABC

from torch.utils.data import Dataset
import os
import numpy as np
import json
import imageio
import torch
from utils.label_utils import colored_mask_to_label_map_np
from utils.math_utils import pose_spherical

import matplotlib.pyplot as plt
from dataset.dataset_interface import NerfDataset
from torchvision import transforms
import cv2
import math
from utils.image_utils import *
from glob import glob


class MitsubaEvalDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("mitsuba_eval", **kwargs)
		self.scene_name = basedir.split("/")[-1]
		self.basedir = basedir
		self.load_diffuse_specular = True
		self.file_n = len(list(glob(os.path.join(self.basedir, "specular_*.png"))))

	def __len__(self):
		return self.file_n

	def __getitem__(self, index):
		sample = {}

		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		rgb_file_path = os.path.join(self.basedir, "rgb_{:03d}.png".format(index))
		diffuse_file_path = os.path.join(self.basedir, "diffuse_{:03d}.png".format(index))
		specular_file_path = os.path.join(self.basedir, "specular_{:03d}.png".format(index))
		irradiance_file_path = os.path.join(self.basedir, "irradiance_{:03d}.png".format(index))
		roughness_file_path = os.path.join(self.basedir, "roughness_{:03d}.png".format(index))
		albedo_file_path = os.path.join(self.basedir, "albedo_{:03d}.png".format(index))

		# (1) load RGB Image
		sample["image"] = load_image_from_path(rgb_file_path, scale=1)
		sample["diffuse"] = load_image_from_path(diffuse_file_path, scale=1)
		sample["specular"] = load_image_from_path(specular_file_path, scale=1)
		sample["irradiance"] = load_image_from_path(irradiance_file_path, scale=1)
		sample["roughness"] = load_image_from_path(roughness_file_path, scale=1)
		sample["albedo"] = load_image_from_path(albedo_file_path, scale=1)
		if "monte_carlo" in self.basedir:
			sample["albedo"] = np.power(sample["albedo"], 1/2.2)
		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
