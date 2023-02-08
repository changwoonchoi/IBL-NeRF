import sys
sys.path.append('../')

from utils.image_utils import *
import os
from dataset.dataset_interface import load_dataset
from piq import ssim, psnr
import torch
import pandas as pd


def load_eval_images(basedir, scene_name, exp_name, target_n=-1, **kwargs):
	path = os.path.join(basedir, scene_name, exp_name)
	if target_n == -1:
		load_folder = sorted(next(os.walk(path))[1])[-1]
	else:
		load_folder = 'testset_{:06d}'.format(target_n)
	load_folder = os.path.join(path, load_folder)

	mitsuba_eval = load_dataset("mitsuba_eval", load_folder, **kwargs)
	mitsuba_eval.load_all_data(4)
	mitsuba_eval.to_tensor("cpu")
	return mitsuba_eval


def eval_error(ground_truth, pred, target="diffuse", metric=None, visualize=False):
	if target == "diffuse":
		dataset_gt = ground_truth.diffuses
		dataset_prd = pred.diffuses
	elif target == "specular":
		dataset_gt = ground_truth.speculars
		dataset_prd = pred.speculars
	elif target =="albedo":
		dataset_gt = ground_truth.albedos
		dataset_prd = pred.albedos
	elif target == "roughness":
		dataset_gt = ground_truth.roughness
		dataset_prd = pred.roughness
	elif target == "irradiance":
		dataset_gt = ground_truth.irradiances
		dataset_prd = pred.irradiances
	else:
		dataset_gt = ground_truth.images
		dataset_prd = pred.images

	if len(dataset_gt.shape) == 3:
		dataset_gt = dataset_gt[..., None]
		dataset_prd = dataset_prd[..., None]
	if dataset_gt.shape[-1] == 1:
		dataset_prd = dataset_prd[..., 0:1]

	# if visualize:
	# 	import matplotlib.pyplot as plt
	# 	plt.imshow(dataset_gt[0])
	# 	plt.figure()
	# 	plt.imshow(dataset_prd[0])
	# 	plt.show()
	dataset_gt = torch.permute(dataset_gt, (0, 3, 1, 2))
	dataset_prd = torch.permute(dataset_prd, (0, 3, 1, 2))

	# dataset_gt = dataset_gt[0:1, ...]
	# dataset_prd = dataset_prd[0:1, ...]

	# dataset_gt = torch.pow(dataset_gt, 2.2)
	# dataset_gt = dataset_gt / (1-dataset_gt + 1e-7)
	# dataset_prd = dataset_prd / (1 - dataset_prd + 1e-7)

	if metric == "ssim":
		metric_f = ssim
	elif metric == "psnr":
		metric_f = psnr
	else:
		metric_f = torch.nn.MSELoss()
	value = metric_f(dataset_gt, dataset_prd)

	return value


def calculate_error_whole(basedir, scene_names=None, exp_names=None, scale=4, skip=10):
	if scene_names is None:
		scene_names = sorted(next(os.walk(basedir))[1])
	if exp_names is None:
		exp_names = sorted(next(os.walk(os.path.join(basedir, scene_names[0])))[1])
	# scene_names = ["bathroom", "beroom", "kitchen", "living-room-2", "living-room-3", "staircase", "veach-ajar", "veach_door_simple"]
	# scene_names = ["kitchen", "bathroom2"]
	# exp_names = ["monte_carlo_env_map", "monte_carlo_nerf_surface", "ours", "ours_hdr", "ours_smooth", "ours_smooth_hdr"]
	# exp_names = ["ours", "ours_with_gt_depth"]
	compare_targets = ["image", "diffuse", "specular", "albedo", "roughness", "irradiance"]
	metrics = ["ssim", "psnr", "mse"]
	df = pd.DataFrame()

	for scene in scene_names:
		load_params = {
			"load_diffuse_specular": True,
			"image_scale": 1/scale,
			"skip": skip,
			"split": "test",
			"load_albedo": "albedo" in compare_targets,
			"load_roughness": "roughness" in compare_targets,
			"load_irradiance": "irradiance" in compare_targets
		}
		load_params_target = {
			"load_diffuse_specular": True,
			"load_albedo": "albedo" in compare_targets,
			"load_roughness": "roughness" in compare_targets,
			"load_irradiance": "irradiance" in compare_targets
		}
		scene_gt_dataset = load_dataset("mitsuba", "../../data/mitsuba/%s" % scene, **load_params)
		scene_gt_dataset.load_all_data(4)
		scene_gt_dataset.to_tensor("cpu")
		print(scene)
		for exp_name in exp_names:
			print(exp_name)
			exp_dataset = load_eval_images(basedir, scene, exp_name, **load_params_target)
			for compare_target in compare_targets:
				print(compare_target)
				metric_errors = {}
				for metric in metrics:
					error = eval_error(scene_gt_dataset, exp_dataset, compare_target, metric, visualize=exp_name=="ours")
					metric_errors[metric] = float(error)

				df = df.append({"scene": scene, "exp_name": exp_name, "compare_target": compare_target, **metric_errors}, ignore_index=True)

		# from utils.visualize_pyplot_utils import visualize_images_vertical
		# import matplotlib.pyplot as plt
		# visualize_images_vertical([diffuse_gt] + diffuses, clim_val=1.0)
		# plt.show()

	average = df.groupby(['compare_target', 'exp_name']).mean()
	average.to_csv(os.path.join(basedir, "error.csv"))
	df.to_csv(os.path.join(basedir, "error_total.csv"), index=False)
	print(df)


# calculate_error_whole("../../logs/final_config/", scene_names=["kitchen"], exp_names=["ours", "monte_carlo_env_map"])
# calculate_error_whole("../../logs/final_config_equal_time/")
# calculate_error_whole("../../logs_eval/final_config_lindisp_equal_sample_scale_4/", scale=4, skip=1)
calculate_error_whole("../../logs_eval/final_config_lindisp_equal_sample/", exp_names=["ours", "monte_carlo_env_map", "monte_carlo_nerf_surface"], scale=1, skip=1)
# calculate_error_whole("../../logs_eval/final_config_ours_only/", scale=1, skip=1)
