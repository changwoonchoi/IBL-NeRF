import sys
sys.path.append('../')

import os
from utils.image_utils import *
import matplotlib.pyplot as plt
import numpy as np
from evaluation.image_cropper import *

def load_image_target(folder, target, index):
	file_path = os.path.join(folder, "testset_099999", target + "_{:03d}.png".format(index))
	return load_image_from_path(file_path, scale=1)

def load_image_target_gt(scene, target, index, scale):
	if target=="rgb":
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d.png" % index)
	else:
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d_%s.png" % (index, target))
	return load_image_from_path(file_path, scale=scale)

def visualize_comparison(basedir, scene_name, index=1, exp_names=None, compare_targets=None, skip=1, scale=1):

	fig_index = 1
	exp_name = "ours_gt_normal"
	path = os.path.join(basedir, scene_name, exp_name)
	out_path = os.path.join(basedir, scene_name, exp_name, "outputs")

	compare_targets = ["reflected_radiance",
						"reflected_coarse_radiance_1",
					   "reflected_coarse_radiance_2",
					   "reflected_coarse_radiance_3",
					   "roughness",
					   "prefiltered_reflected",
					   "rgb"]

	n_row = len(compare_targets) // 2
	n_col = 2
	fig = plt.figure(figsize=(2 * n_col + 2, 2 * n_row))

	def crop_image_temp(_image, name):
		# cropped_image = image[226:277, 50:91, :]
		target_crops_s = [(45, 210), (451, 262), (360, 100)]
		sizes = [(64, 64), (64,64), (128, 128)]
		target_crops = []
		for i_crop, target_crop in enumerate(target_crops_s):
			s, e = target_crop
			target_crops.append((s, e, s + sizes[i_crop][0], e + sizes[i_crop][1]))

		for i_crop, target_crop in enumerate(target_crops):
			cropped_image = crop(_image, target_crop)
			save_image_pil(cropped_image, "%s/%s_cropped_%d.png" % (out_path, name, i_crop))

		rectangle_image = draw_image(_image, target_crops)
		save_image_pil(rectangle_image, "%s/%s_cropped.png" % (out_path, name))

	for i_target, compare_target in enumerate(compare_targets):
		image = load_image_target(path, compare_target, index)
		crop_image_temp(image, compare_target)

		#ax = fig.add_subplot(n_row, n_col, fig_index)
		#ax.imshow(image)
		fig_index += 1

	rgb_gt = load_image_target_gt(scene_name, "rgb", index + 1, scale=1)
	crop_image_temp(rgb_gt, "rgb_gt")
	rgb_gt = load_image_target_gt(scene_name, "specular", index + 1, scale=1)
	crop_image_temp(rgb_gt, "specular_gt")

	roughness = load_image_target(path, "roughness", index)
	roughness = roughness[..., 0]
	N = 4
	roughness_index = (roughness * N).astype(np.int32)
	roughness_index = np.clip(roughness_index, 0, N-1)
	roughness_index_remainder = (roughness * N) - roughness_index
	colors = [
		[1, 1, 1],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	]
	colors = np.asarray(colors)
	roughness_colored1 = colors[roughness_index]
	roughness_colored2 = colors[np.clip(roughness_index + 1, 0, N-1)]
	roughness_colored = roughness_index_remainder[...,None] * roughness_colored2 + (1-roughness_index_remainder[...,None])* roughness_colored1
	crop_image_temp(roughness_colored, "roughness_colored")

	# print(roughness_colored.shape)
	plt.imshow(roughness_colored)
	save_pred_images(roughness_colored, "%s/roughness_colored_%d.png" % (out_path, index))
	# plt.suptitle("Index: %d"% index)
	#fig.tight_layout()
	plt.suptitle("Index: %d" % index)
	plt.show()
	#plt.savefig('line_plot.pdf')
visualize_comparison("../../logs_eval/final_config_ours_only/", "kitchen", index=23)
#for i in range(23,24,1):
#	visualize_comparison("../../logs_eval/final_config_ours_only/", "kitchen", index=i+1)