from utils.image_utils import *
import os
from dataset.dataset_interface import load_dataset
from piq import ssim, psnr
import torch
import pandas as pd

import json
def calculate_time_whole(basedir, scene_names=None, exp_names=None, export_base_dir = None, load_from_tar=False):
	if export_base_dir is None:
		export_base_dir = basedir.replace('logs', 'logs_eval')

	if scene_names is None:
		scene_names = sorted(next(os.walk(basedir))[1])
	if exp_names is None:
		exp_names = sorted(next(os.walk(os.path.join(basedir, scene_names[0])))[1])

	df = pd.DataFrame()

	for scene_name in scene_names:
		print(scene_name)

		for exp_name in exp_names:
			path = os.path.join(basedir, scene_name, exp_name)
			if load_from_tar:
				ckpt_path = [f for f in sorted(os.listdir(path)) if 'tar' in f][-1]
				ckpt_path = os.path.join(path, ckpt_path)
				ckpt = torch.load(ckpt_path)
				step = ckpt['global_step']
				time = ckpt['elapsed_time']
			else:
				try:
					json_path = os.path.join(path, "train_info_step_time.json")
					with open(json_path, 'r') as fp:
						f = json.load(fp)
						step = f["global_step"]
						time = f["training_time"]
				except Exception:
					pass
			time_per_step = time / step
			df = df.append({"scene": scene_name, "exp_name": exp_name, "step": step, "time": time, "time_per_step": time_per_step}, ignore_index=True)

	average = df.groupby(['exp_name']).mean()
	average.to_csv(os.path.join(basedir, "time.csv"))
	df.to_csv(os.path.join(basedir, "time_total.csv"), index=False)



# calculate_error_whole("../logs/final_config/", scene_names=["kitchen"], exp_names=["ours", "monte_carlo_env_map"])
# calculate_error_whole("../logs/final_config_equal_time/")
# calculate_time_whole("../../logs/final_config_lindisp_equal_sample/")
calculate_time_whole("../../logs/final_config_ours_only/")
