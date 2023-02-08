import sys
sys.path.append('../')

import os
from utils.image_utils import *
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import natsort
import cv2
from PyPDF2 import PdfFileMerger
from pathlib import Path
import re
import numpy as np
import natsort
import pprint

def load_image_target(folder, target, index, count=-1):
	if count == -1:
		target_count = "testset_099999"
	else:
		target_count = "testset_{:06d}".format(count)
	file_path = os.path.join(folder, target_count, target + "_{:03d}.png".format(index))
	return load_image_from_path(file_path, scale=1)


def load_image_target_gt_mitsuba(scene, target, index, scale):
	if target=="rgb":
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d.png" % index)
	else:
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d_%s.png" % (index, target))
	return load_image_from_path(file_path, scale=scale)

def find_all_images(basedir):
	image_name_list = []
	for path in Path(basedir).rglob("*.png"):
		image_name_list.append(str(path))
	image_name_list = natsort.natsorted(image_name_list)
	#print(len(image_name_list))
	#pprint.pprint(image_name_list)
	return image_name_list


def load_image_target_gt(scene, target, index, scale):
	file_path = os.path.join("../../data/falcor", scene)
	image_name_list = find_all_images(file_path)
	file_path = image_name_list[index-1]
	return load_image_from_path(file_path, scale=scale)


def load_image_target_gt_replica(scene, target, index, scale):
	if target=="disp":
		depth_file_path = os.path.join("../../data/replica", scene, "test", 'depth{:06d}.png'.format(index))
		depth = cv2.imread(depth_file_path, -1)
		depth = cv2.resize(depth, None, fx=scale, fy=scale)
		depth_scale = 65535.0 * 0.1
		depth = depth.astype(np.float32)
		depth = depth / depth_scale
		depth = 1 / (depth + 1e-10)
		depth = np.clip(depth, 0, 1)
		depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
		return depth
	elif target=="radiance" or target =="rgb":
		file_path = os.path.join("../../data/replica", scene, "test", 'frame{:06d}.jpg'.format(index))
		return load_image_from_path(file_path, scale=scale)
	else:
		file_path = os.path.join("../../data/replica", scene, "test", 'frame{:06d}.jpg'.format(index))
		image = load_image_from_path(file_path, scale=scale)
		return np.ones_like(image)


def visualize_comparison(basedir, scene_name, index=1, exp_names=None, compare_targets=None, skip=1, scale=1, target_iter=-1):
	exp_names_dict = {
		"ours": "Ours",
		"ours_gt_normal": "Ours(GT)",
		"ours_hdr": "Ours",
		"monte_carlo_nerf_surface": "MC",
		"monte_carlo_env_map": "MC + Env",
		"gt": "GT"
	}

	if exp_names is None:
		# exp_names = ["monte_carlo_nerf_surface", "monte_carlo_env_map", "ours", "gt"]
		path = os.path.join(basedir, scene_name)
		#exp_names = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI)) and "smooth" not in dI and "hdr" not in dI]
		exp_names = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
		exp_names = natsort.natsorted(exp_names)
		exp_names = ["gt"] + exp_names
		print(exp_names)

	if compare_targets is None:
		# compare_targets = ["diffuse", "specular", "rgb"]
		compare_targets = ["disp", "albedo", "irradiance", "roughness", "diffuse", "specular", "rgb", "radiance"]

	n_row = len(exp_names)
	n_col = len(compare_targets)
	fig = plt.figure(figsize=(2 * n_col + 2, 2 * n_row))
	fig_index = 1

	for i_exp, exp_name in enumerate(exp_names):
		path = os.path.join(basedir, scene_name, exp_name)

		for i_target, compare_target in enumerate(compare_targets):
			try:
				if exp_name == "gt":
					# image = load_image_target_gt_replica(scene_name, compare_target, skip * index, 1 / scale)
					scene_name_temp = scene_name.replace("_depth", "")
					scene_name_temp = scene_name_temp.replace("_separate", "")

					image = load_image_target_gt(scene_name_temp, compare_target, skip * index + 1, 1 / scale)

				else:
					image = load_image_target(path, compare_target, index, target_iter)
				#if compare_target == 'albedo' and exp_name != 'gt':
				#	image = np.power(image, 1 / 2.2)

				ax = fig.add_subplot(n_row, n_col, fig_index)
				# plt.axis('off')
				plt.xticks([])
				plt.yticks([])
				if i_exp == 0:
					ax.set_xlabel(compare_target)
					ax.xaxis.set_label_position('top')
				if i_target == 0:
					# ax.set_ylabel(exp_names_dict.get(exp_name, exp_name))
					ax.set_ylabel(exp_name)

				ax.imshow(image)
				fig_index += 1
			except:
				fig_index += 1

	plt.suptitle("Scene: %s, Index: %d" % (scene_name, index))
	fig.tight_layout()
	#plt.show()
	directory = '../../images_merged/%s' % (basedir.split("/")[-1])
	if not os.path.exists(directory):
		os.makedirs(directory)

	pdf_name = '%s/%s.pdf' % (directory, scene_name)
	plt.savefig(pdf_name)
	return pdf_name


#for i in range(100):
#	visualize_comparison("../../logs_eval/final_config_ours_only/", "veach-ajar", index=i+1, exp_names=["ours", "ours_gt_normal"])
#for i in range(100):
#	visualize_comparison("../../logs_eval/final_config_lindisp_equal_sample", "bedroom", index=i+1)

# visualize_comparison("../../logs/final_config_ours_only", "veach-ajar", index=3, skip=10, target_iter=200000)

# visualize_comparison("../../logs/final_config_ours_only_object", "chair", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "ficus", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "hotdog", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "lego", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "materials", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "mic", index=3, skip=10, target_iter=200000)
# visualize_comparison("../../logs/final_config_ours_only_object", "ship", index=3, skip=10, target_iter=200000)

#visualize_comparison("../../logs/final_config_ours_only", "dining-room", index=1, skip=10, target_iter=200000)
#visualize_comparison("../../logs/final_config_ours_only", "classroom", index=1, skip=10, target_iter=200000)
#visualize_comparison("../../logs/final_config_ours_only", "bathroom", index=1, skip=10, target_iter=200000)


# path = os.path.join("../../logs/final_config_ours_only_replica")
# scene_names = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
# print(scene_names)
# for scene_name in scene_names:
# 	visualize_comparison("../../logs/final_config_ours_only_replica", scene_name, index=1, skip=10, target_iter=100000)

#basedir = "../../logs/final_config_ours_only_roughness_exp_coeff"
#basedir = "../../logs/final_config_ours_only_compare_gt_fixed"
#basedir = "../../logs/final_config_ours_only_mutiplied_v"
basedir = "../../logs/falcor"

#scene_names = [dI for dI in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, dI))]
#scene_names = natsort.natsorted(scene_names)
#scene_names = ['bedroom', 'bedroom_depth', 'classroom', 'classroom_depth', 'kitchen', 'kitchen_depth', 'living-room', 'living-room_depth', 'living-room-2', 'living-room-2_depth', 'veach_door_simple', 'veach_door_simple_depth']
#scene_names = ["kitchen", "bedroom", "living-room-2", "veach_door_simple"]
scene_names = ["kitchen", "kitchen_separate"]
print(scene_names)
pdf_list = []

for scene_name in scene_names:
	pdf = visualize_comparison(basedir, scene_name, index=2, skip=10, target_iter=500000)
	pdf_list.append(pdf)

pdf_merger = PdfFileMerger(strict=False)
for file in pdf_list:
	pdf_merger.append(file)

pdf_merger.write("../../images_merged/%s/merged.pdf"% basedir.split("/")[-1])
pdf_merger.close()

#visualize_comparison(basedir, "kitchen", index=2, skip=10, target_iter=100000)

#visualize_comparison("../../logs/final_config_ours_only_roughness_exp_coeff", "kitchen", index=2, skip=10, target_iter=100000)
#visualize_comparison("../../logs/final_config_ours_only_replica", "office_1", index=1, skip=10, target_iter=100000)
#visualize_comparison("../../logs/final_config_ours_only_replica", "office_2", index=1, skip=10, target_iter=100000)
#visualize_comparison("../../logs/final_config_ours_only_replica", "office_3", index=1, skip=10, target_iter=100000)
