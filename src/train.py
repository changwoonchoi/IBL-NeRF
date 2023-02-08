import random
import numpy as np
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
DEBUG = False

from config_parser import export_config
from nerf_models.ibl_nerf_renderer import render_decomp, render_decomp_path
from nerf_models.ibl_nerf import create_IBLNeRF

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.label_utils import *
from config_parser import recursive_config_parser
from dataset.dataset_interface import load_dataset
from miscellaneous.test_dataset_speed import *

from utils.generator_utils import *
from utils.timing_utils import *
import cv2
from torch.nn.functional import normalize
from utils.math_utils import *


def test(args):
	# (0) Print train phase overview
	logger_dataset = load_logger("Dataset Info")
	logger_export = load_logger("Export Logger")
	use_instance_mask = args.instance_mask
	logger_dataset.info("Instance mask: " + str(use_instance_mask))
	logger_dataset.info("Instance mask encoding: " + str(args.instance_label_encoding))
	logger_dataset.info("Infer normal: " + str(args.infer_normal))
	logger_dataset.info("Learn normal from oracle: " + str(args.learn_normal_from_oracle))
	logger_dataset.info("Learn albedo from oracle: " + str(args.learn_albedo_from_oracle))

	# (1) Load dataset
	with time_measure("[1] Data load"):
		def load_dataset_split(split="train", do_logging=True, **kwargs):
			# create dataset config
			target_dataset = load_dataset(args.dataset_type, args.datadir, split=split, **kwargs)
			target_dataset.load_instance_label_mask = use_instance_mask

			# real data load using multiprocessing(torch DataLoader) --> load all at once
			# TODO : if dataset is too large, it may not be loaded at once.
			target_dataset.load_all_data(num_of_workers=1)
			if do_logging:
				logger_dataset.info(target_dataset)
			return target_dataset

		# load train and validation dataset
		load_normal = args.learn_normal_from_oracle or args.calculating_normal_type == "ground_truth"
		load_params = {
			"image_scale": args.image_scale,
			"load_image": False,
			"load_normal": load_normal,
			"load_depth": False,
			"load_roughness": False,
			"load_albedo": False,
			"sample_length": args.sample_length,
			"coarse_radiance_number": args.coarse_radiance_number,
			"load_instance_label_mask": args.instance_mask,
			"near_plane": args.near_plane,
			"far_plane": args.far_plane,
			"load_depth_range_from_file": args.load_depth_range_from_file,
			"gamma_correct": args.gamma_correct
		}

		dataset = load_dataset_split("test", skip=1, load_priors=False, **load_params)

		hwf = [dataset.height, dataset.width, dataset.focal]

		# move data to GPU side
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		dataset.to_tensor(args.device)

		# Load BRDF LUT
		brdf_lut_path = "../data/ibl_brdf_lut.png"
		brdf_lut = cv2.imread(brdf_lut_path)
		brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)

		brdf_lut = brdf_lut.astype(np.float32)
		brdf_lut /= 255.0
		brdf_lut = torch.tensor(brdf_lut).to(args.device)
		brdf_lut = brdf_lut.permute((2, 0, 1))

	# (2) Create log file / folder
	with time_measure("[2] Log file create"):
		# Create log dir and copy the config file
		expname = args.expname
		export_config(args, args.export_basedir)

	# (3) Create nerf model
	with time_measure("[3] NeRFDecomp load"):
		# set instance label dimension
		label_encoder = None
		if use_instance_mask:
			label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding,
											  args.instance_label_dimension)
			args.instance_label_dimension = label_encoder.get_dimension()
		else:
			args.instance_label_dimension = 0

		# create nerf model
		args.num_cluster = dataset.num_cluster
		render_kwargs_train, render_kwargs_test, start, elapsed_time, grad_vars, optimizer = create_IBLNeRF(args)
		global_step = start

		# update near / far plane
		bds_dict = dataset.get_near_far_plane()
		render_kwargs_train.update(bds_dict)
		render_kwargs_test.update(bds_dict)

		render_kwargs_train['brdf_lut'] = brdf_lut
		render_kwargs_test['brdf_lut'] = brdf_lut
		is_instance_label_logit = False
		if use_instance_mask:
			is_instance_label_logit = isinstance(label_encoder, OneHotLabelEncoder) and (args.CE_weight_type != "mse")
			render_kwargs_train["is_instance_label_logit"] = is_instance_label_logit
			render_kwargs_test["is_instance_label_logit"] = is_instance_label_logit
		logger_render_options = load_logger("Render Kwargs")
		logs = ["[Render Kwargs (simple only)]"]
		for k, v in render_kwargs_train.items():
			if isinstance(v, (str, float, int, bool)):
				logs += ["\t-%s : %s" % (k, str(v))]
		logger_render_options.info("\n".join(logs))

	# (5) Main eval loop
	K = dataset.get_focal_matrix()

	hemisphere_samples = get_hemisphere_samples(args.N_hemisphere_sample_sqrt)
	hemisphere_samples = torch.Tensor(hemisphere_samples).to(args.device)

	def run_test_dataset(_i, render_factor=4):
		testsavedir = os.path.join(args.export_basedir, expname, 'testset_{:06d}'.format(_i))
		os.makedirs(testsavedir, exist_ok=True)

		render_decomp_path(
			dataset, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
			render_factor=render_factor, init_basecolor=dataset.init_basecolor,
			calculate_normal_from_depth_map=args.calculate_all_analytic_normals,
			use_instance=use_instance_mask, label_encoder=label_encoder,
			hemisphere_samples=hemisphere_samples,
			approximate_radiance=True
		)

	with torch.no_grad():
		run_test_dataset(global_step, render_factor=1)


def train(args):
	# (0) Print train phase overview
	logger_dataset = load_logger("Dataset Info")
	logger_export = load_logger("Export Logger")

	# (1) Load dataset
	with time_measure("[1] Data load"):
		def load_dataset_split(split="train", do_logging=True, **kwargs):
			# create dataset config
			target_dataset = load_dataset(args.dataset_type, args.datadir, split=split, **kwargs)

			target_dataset.load_all_data(num_of_workers=1)
			if do_logging:
				logger_dataset.info(target_dataset)
			return target_dataset

		load_params = {
			"image_scale": args.image_scale,
			"sample_length": args.sample_length,
			"coarse_radiance_number": args.coarse_radiance_number,
			"near_plane": args.near_plane,
			"far_plane": args.far_plane,
			"load_depth_range_from_file": args.load_depth_range_from_file,
			"gamma_correct": args.gamma_correct,
			"load_priors": args.load_priors,
			"prior_type": args.prior_type
		}
		dataset = load_dataset_split("train", **load_params)

		# force load albedo, normal, and irradiance for test set
		load_params["load_albedo"] = True
		load_params["load_normal"] = True
		load_params["load_irradiance"] = True

		# force not to load albedo & irradiance prior images for test set
		load_params["load_priors"] = False
		if args.dataset_type == "mitsuba":
			dataset_val = load_dataset_split("test", skip=10, **load_params)
		else:
			raise ValueError("We provide only IBL-NeRF dataset for current version.")

		hwf = [dataset.height, dataset.width, dataset.focal]

		# move data to GPU side
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		dataset.to_tensor(args.device)
		dataset_val.to_tensor(args.device)

		# Load BRDF LUT
		brdf_lut_path = "../data/ibl_brdf_lut.png"
		brdf_lut = cv2.imread(brdf_lut_path)
		brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)

		brdf_lut = brdf_lut.astype(np.float32)
		brdf_lut /= 255.0
		brdf_lut = torch.tensor(brdf_lut).to(args.device)
		brdf_lut = brdf_lut.permute((2, 0, 1))

	# (2) Create log file / folder
	with time_measure("[2] Log file create"):
		# Create log dir and copy the config file
		basedir = args.basedir
		expname = args.expname
		export_config(args, basedir)

		# Create Tensorboard writer
		writer = SummaryWriter(log_dir=os.path.join(basedir, expname))

	# (3) Create nerf model
	with time_measure("[3] IBL-NeRF load"):
		# create nerf model
		render_kwargs_train, render_kwargs_test, start, elapsed_time, grad_vars, optimizer = create_IBLNeRF(args)
		global_step = start

		# update near / far plane
		bds_dict = dataset.get_near_far_plane()
		render_kwargs_train.update(bds_dict)
		render_kwargs_test.update(bds_dict)

		render_kwargs_train['brdf_lut'] = brdf_lut
		render_kwargs_test['brdf_lut'] = brdf_lut

		logger_render_options = load_logger("Render Kwargs")
		logs = ["[Render Kwargs (simple only)]"]
		for k, v in render_kwargs_train.items():
			if isinstance(v, (str, float, int, bool)):
				logs += ["\t-%s : %s" % (k, str(v))]
		logger_render_options.info("\n".join(logs))

	# (4) Create the sample generator
	with time_measure("[4] Sample generator create"):
		batch_size = args.N_rand
		use_batching = not args.no_batching
		start = start + 1

		if use_batching:
			sample_generator = sample_generator_all_image_merged(dataset, batch_size=batch_size)
		else:
			sample_generator = sample_generator_single_image(dataset, batch_size=batch_size,
															 precrop_iters=args.precrop_iters,
															 precrop_frac=args.precrop_frac, initial_iters=start,
															 ray_sample=args.ray_sample)

	# (5) Main train loop
	K = dataset.get_focal_matrix()
	N_iters = args.N_iter + 1

	# export ground truth image
	img_gt = dataset_val.images.permute((0, 3, 1, 2))
	writer.add_images('test/gt_rgb', img_gt, 0)

	for k in range(args.coarse_radiance_number):
		img_gt_k = dataset_val.prefiltered_images[k].permute((0, 3, 1, 2))#.get_coarse_images(k+1)
		writer.add_images('test/gt_rgb_coarse_%d' % (k+1), img_gt_k, 0)

	normal_gt = dataset_val.normals.permute((0, 3, 1, 2))
	albedo_gt = dataset_val.albedos.permute((0, 3, 1, 2))
	irradiance_gt = dataset_val.irradiances.permute((0, 3, 1, 2))

	writer.add_images('test/gt_irradiance', irradiance_gt, 0)
	writer.add_images('test/gt_normal', normal_gt, 0)
	writer.add_images('test/gt_albedo', albedo_gt, 0)


	mse_loss = torch.nn.MSELoss()
	l1_loss = torch.nn.L1Loss()

	normal_target_keys = [
		"normal_map_from_sigma_gradient",
		"normal_map_from_sigma_gradient_surface",
		"normal_map_from_depth_gradient",
		"normal_map_from_depth_gradient_direction",
		"normal_map_from_depth_gradient_epsilon",
		"normal_map_from_depth_gradient_direction_epsilon"
	]

	assert args.calculating_normal_type in normal_target_keys + ["ground_truth"]


	render_kwargs_train["N_hemisphere_sample_sqrt"] = args.N_hemisphere_sample_sqrt
	render_kwargs_test["N_hemisphere_sample_sqrt"] = args.N_hemisphere_sample_sqrt

	time_limit_in_sec = -1
	if args.time_limit_in_minute > 0:
		time_limit_in_sec = args.time_limit_in_minute * 60
		N_iters = 1000000

	def save_file(_i):
		path = os.path.join(basedir, expname, '{:06d}.tar'.format(_i))
		save_target = {
			'global_step': global_step,
			'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
			'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'elapsed_time': elapsed_time
		}

		torch.save(save_target, path)
		print('Saved checkpoints at', path)

	def run_test_dataset(_i, render_factor=4):
		testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(_i))
		os.makedirs(testsavedir, exist_ok=True)

		for param_group in optimizer.param_groups:
			for var in param_group['params']:
				var.requires_grad = False

		render_decomp_path_results = render_decomp_path(
			dataset_val, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
			render_factor=render_factor, approximate_radiance=True
		)

		for key_name in render_decomp_path_results.keys():
			stacked_images = render_decomp_path_results[key_name]
			if len(stacked_images.shape) != 4:
				stacked_images = np.expand_dims(stacked_images, -1)
			writer.add_images('test/inferred/%s' % key_name, stacked_images.transpose((0, 3, 1, 2)), _i)

		for param_group in optimizer.param_groups:
			for var in param_group['params']:
				var.requires_grad = True
		logger_export.info('Saved test set')

	original_lr_rates = {}
	for param_group in optimizer.param_groups:
		original_lr_rates[param_group['name']] = param_group['lr']
		print(param_group['name'], 'learning rate : ', param_group['lr'])


	for i in trange(start, N_iters):
		ith_train_start_time = time.time()

		# sample rgb and rays from sample generator
		# target_info, rays_o, rays_d = next(sample_generator)  # 3 x (N_rand, 3)
		target_info, rays_o, rays_d, neigh_info, rays_o_neigh, rays_d_neigh = next(sample_generator)  # (N_rand, 3), (N_rand, 8, 3)

		target_rgb = target_info["rgb"]
		if args.learn_albedo_from_oracle:
			target_chromaticity = target_info["albedo"]
		else:
			target_chromaticity = target_rgb / (torch.linalg.norm(target_rgb, dim=-1, keepdim=True) + 1e-10)
		if args.load_priors:
			target_prior_albedo = target_info["prior_albedo"]
			target_prior_albedo_chrom = target_prior_albedo / (torch.linalg.norm(target_prior_albedo, dim=-1, keepdim=True) + 1e-10)
			target_prior_irradiance = target_info["prior_irradiance"]
		batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
		batch_rays_neigh = None
		if args.ray_sample == "patch":
			batch_rays_neigh = torch.stack([rays_o_neigh.reshape((-1, 3)), rays_d_neigh.reshape((-1, 3))],
										   0)  # (2, N_rand * 8, 3)

		#####  Core optimization loop  #####
		if args.calculate_all_analytic_normals:
			calculate_normal_from_depth_gradient = (i % args.summary_step == 0)
			calculate_normal_from_sigma_gradient = (i % args.summary_step == 0)
			calculate_normal_from_sigma_gradient_surface = (i % args.summary_step == 0)
			calculate_normal_from_depth_gradient_direction = (i % args.summary_step == 0)
			calculate_normal_from_depth_gradient_epsilon = (i % args.summary_step == 0)
			calculate_normal_from_depth_gradient_direction_epsilon = (i % args.summary_step == 0)
		else:
			calculate_normal_from_depth_gradient = False
			calculate_normal_from_sigma_gradient = False
			calculate_normal_from_sigma_gradient_surface = False
			calculate_normal_from_depth_gradient_direction = False
			calculate_normal_from_depth_gradient_epsilon = False
			calculate_normal_from_depth_gradient_direction_epsilon = False

		if i >= args.N_iter_ignore_normal:
			if args.infer_normal_target == "normal_map_from_sigma_gradient":
				calculate_normal_from_sigma_gradient = True
			elif args.infer_normal_target == "normal_map_from_sigma_gradient_surface":
				calculate_normal_from_sigma_gradient_surface = True
			elif args.infer_normal_target == "normal_map_from_depth_gradient":
				calculate_normal_from_depth_gradient = True
			elif args.infer_normal_target == "normal_map_from_depth_gradient_direction":
				calculate_normal_from_depth_gradient_direction = True
			elif args.infer_normal_target == "normal_map_from_depth_gradient_epsilon":
				calculate_normal_from_depth_gradient_epsilon = True
			elif args.infer_normal_target == "normal_map_from_depth_gradient_direction_epsilon":
				calculate_normal_from_depth_gradient_direction_epsilon = True

		if (i >= args.N_iter_ignore_approximated_radiance) and args.freeze_radiance:
			render_kwargs_train["network_fn"].freeze_radiance = True
			render_kwargs_train["network_fine"].freeze_radiance = True

		if args.load_priors and i >= args.N_iter_ignore_prior and args.freeze_roughness:
			render_kwargs_train["network_fn"].freeze_roughness = True
			render_kwargs_train["network_fine"].freeze_roughness = True
			render_kwargs_train["network_fn"].freeze_radiance = True
			render_kwargs_train["network_fine"].freeze_radiance = True

		# 1. render sample
		result = render_decomp(
			dataset.height, dataset.width, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
			calculate_normal_from_sigma_gradient=calculate_normal_from_sigma_gradient,
			calculate_normal_from_sigma_gradient_surface=calculate_normal_from_sigma_gradient_surface,
			calculate_normal_from_depth_gradient=calculate_normal_from_depth_gradient,
			calculate_normal_from_depth_gradient_direction=calculate_normal_from_depth_gradient_direction,
			calculate_normal_from_depth_gradient_epsilon=calculate_normal_from_depth_gradient_epsilon,
			calculate_normal_from_depth_gradient_direction_epsilon=calculate_normal_from_depth_gradient_direction_epsilon,
			gt_values=target_info,
			approximate_radiance=i >= args.N_iter_ignore_approximated_radiance,
			**render_kwargs_train
		)

		def calculate_loss(key_name, target="ground_truth_normal", loss_fn=mse_loss):
			if key_name not in result:
				# print("Key %s not in result" % key_name)
				return 0
			if isinstance(target, float):
				loss_from_target = torch.mean((result[key_name]-target) ** 2)

				if key_name + '0' in result:
					loss_from_target += torch.mean((result[key_name + '0']-target) ** 2)
			elif not isinstance(target, str):
				loss_from_target = loss_fn(result[key_name], target)

				if key_name + '0' in result:
					loss_from_target += loss_fn(result[key_name + '0'], target)
			else:
				loss_from_target = loss_fn(result[key_name], result[target])
				if key_name + '0' in result:
					if target + '0' in result:
						loss_from_target += loss_fn(result[key_name + '0'], result[target + '0'])
					else:
						loss_from_target += loss_fn(result[key_name + '0'], result[target])
			return loss_from_target

		# normal from gt
		if "normal" in target_info:
			result["ground_truth_normal"] = normalize(2 * target_info["normal"] - 1, dim=-1)

		# 2. calculate loss

		# 0) approximated radiance loss
		loss_render = calculate_loss("color_map", target_rgb)

		# 1) radiance loss
		loss_render_radiance = calculate_loss("radiance_map", target_rgb)

		# 1-A) coarse radiance loss (for prefiltered env-map)
		loss_render_coarse_radiance = []
		for k in range(args.coarse_radiance_number):
			# print("Result", k, result["radiance_map_%d" % (k + 1)])
			loss_render_radiance_i = calculate_loss("radiance_map_%d" % (k + 1),
													target_info["rgb_%d" % (k + 1)])
			loss_render_coarse_radiance.append(loss_render_radiance_i)

		# 2) albedo render loss
		loss_albedo_render = calculate_loss("albedo_map", target_chromaticity)

		# 3) Depth map if required
		loss_depth = 0
		if args.infer_depth and i >= args.N_iter_ignore_depth:
			start_time = time.time()
			# according to NeRV paper
			# 3-1) point from camera
			loss_depth = mse_loss(result['inferred_depth_map'], result['depth_map'].detach())

			# 3-2) point in volume
			# select first hit point & random direction align with normal
			normal_map = result["ground_truth_normal"]

			expected_points = rays_o + rays_d * result['depth_map'][..., None]
			expected_points = expected_points.detach()

			random_direction = 2 * torch.rand(*rays_d.shape) - 1
			normal_dot = torch.sum(random_direction * normal_map, dim=-1)
			random_direction = torch.sign(normal_dot)[..., None] * random_direction
			random_direction = F.normalize(random_direction, dim=-1)

			random_points = torch.stack([expected_points.reshape((-1, 3)), random_direction.reshape((-1, 3))],
										0)

			random_points = random_points[:, 0:args.N_depth_random_volume, :]
			# with torch.no_grad():
			result_random_volume = render_decomp(
				dataset.height, dataset.width, K, chunk=args.chunk, rays=random_points, verbose=i < 10,
				retraw=True,
				init_basecolor=dataset.init_basecolor,
				is_depth_only=True,
				approximate_radiance=False,
				**render_kwargs_train
			)

			loss_depth_random = mse_loss(result_random_volume['inferred_depth_map'],
										 result_random_volume['depth_map'].detach())
			loss_depth += loss_depth_random
		# if 'depth_map0' in result:
		#    loss_depth += mse_loss(result['inferred_depth_map'], result['depth_map0'].detach())

		loss_sigma_depth = 0
		if args.depth_map_from_ground_truth and args.train_depth_from_ground_truth:
			loss_sigma_depth = calculate_loss("depth_map", target_info["depth"][..., 0])
			loss_sigma_depth /= (dataset.far * dataset.far * 0.1)

		# 3) Normal render loss

		# inferred & gt / sigma
		loss_inferred_normal = 0
		if args.infer_normal and i >= args.N_iter_ignore_normal:
			loss_inferred_normal = calculate_loss("inferred_normal_map", args.infer_normal_target)


		# 7) prior loss
		if args.load_priors:
			if args.albedo_prior_type == "chrom":
				result["albedo_chrom_map"] = result["albedo_map"] / (torch.linalg.norm(result["albedo_map"], dim=-1, keepdim=True) + 1e-10)
				loss_prior_albedo = calculate_loss("albedo_chrom_map", target_prior_albedo_chrom)
			elif args.albedo_prior_type == "rgb":
				loss_prior_albedo = calculate_loss("albedo_map", target_prior_albedo)
			else:
				raise ValueError
			loss_prior_irradiance = calculate_loss("irradiance_map", target_prior_irradiance)

		# 8) irradiance regularize
		loss_irradiance_reg = 0
		if args.load_priors and i >= args.N_iter_ignore_prior:
			loss_irradiance_reg = mse_loss(result["irradiance_map"], torch.ones_like(result["irradiance_map"]) * dataset.prior_irradiance_mean)

		# Final loss
		# (a) radiance loss
		total_loss = args.beta_radiance_render * loss_render_radiance
		total_loss += args.beta_sigma_depth * loss_sigma_depth
		for k in range(args.coarse_radiance_number):
			total_loss += args.beta_radiance_render * loss_render_coarse_radiance[k]

		if args.initialize_roughness and i < args.N_iter_ignore_approximated_radiance:
			total_loss += args.beta_roughness_render * calculate_loss("roughness_map", args.roughness_init)

		# (b) normal loss
		if i >= args.N_iter_ignore_normal:
			total_loss += args.beta_inferred_normal * loss_inferred_normal
		# args.beta_render * loss_render + args.beta_albedo_render * loss_albedo_render + args.beta_inferred_depth * loss_depth
		# + args.beta_inferred_normal * loss_inferred_normal + args.beta_radiance_render * loss_render_radiance \
		if i >= args.N_iter_ignore_approximated_radiance:
			total_loss += args.beta_render * loss_render


		# (f) depth loss
		if i >= args.N_iter_ignore_depth:
			total_loss += args.beta_inferred_depth * loss_depth

		# (g) prior loss
		if i >= args.N_iter_ignore_prior and args.load_priors:
			total_loss += args.beta_prior_albedo * loss_prior_albedo
			total_loss += args.beta_prior_irradiance * loss_prior_irradiance
			total_loss += args.beta_irradiance_reg * loss_irradiance_reg
		# print(loss_albedo_render, "Loss_albedo_render!!")

		if i % args.summary_step == 0:
			writer.add_scalar("elapsed_time", elapsed_time, i)

			writer.add_scalar('Loss/Total_Loss', total_loss, i)
			writer.add_scalar('Loss/Loss_render', loss_render, i)

			writer.add_scalar('Loss/Loss_albedo_render', loss_albedo_render, i)
			# writer.add_scalar('Loss/Loss_roughness_render', loss_roughness_render, i)

			writer.add_scalar('Loss/Loss_radiance_render', loss_render_radiance, i)

			for k in range(args.coarse_radiance_number):
				writer.add_scalar('Loss/Loss_radiance_render_coarse_%d' % (k + 1),
								  loss_render_coarse_radiance[k], i)

			if args.calculate_all_analytic_normals:
				for normal_key in normal_target_keys:
					loss_from_gt = calculate_loss(normal_key)
					writer.add_scalar('Loss_normal/%s' % normal_key, loss_from_gt, i)

			if args.infer_depth:
				writer.add_scalar('Loss/Loss_depth', loss_depth, i)

			if args.infer_normal:
				writer.add_scalar('Loss_normal/inferred_normal', loss_inferred_normal, i)
				loss_from_gt = calculate_loss("inferred_normal_map", "ground_truth_normal")
				writer.add_scalar('Loss_normal/inferred_normal_from_gt', loss_from_gt, i)
			if args.depth_map_from_ground_truth and args.train_depth_from_ground_truth:
				writer.add_scalar('Loss/loss_sigma_depth_from_gt', loss_sigma_depth, i)

			if args.load_priors:
				writer.add_scalar('Loss/Loss_prior_albedo', loss_prior_albedo, i)
				writer.add_scalar('Loss/Loss_prior_irradiance', loss_prior_irradiance, i)
				writer.add_scalar('Loss/Loss_irradiance_reg', loss_irradiance_reg, i)

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		decay_rate = 0.1
		decay_steps = args.lrate_decay * 1000

		def set_lr(name, start_count):
			for param_group in optimizer.param_groups:
				if param_group['name'] == name and global_step > start_count:
					new_lrate = original_lr_rates[name] * (decay_rate ** ((global_step - start_count) / decay_steps))
					param_group['lr'] = new_lrate

		set_lr("coarse", 0)
		set_lr("fine", 0)
		set_lr("depth", args.N_iter_ignore_depth)
		set_lr("normal", args.N_iter_ignore_normal)
		set_lr("albedo_mlp", args.N_iter_ignore_approximated_radiance)
		set_lr("roughness_mlp", args.N_iter_ignore_approximated_radiance)
		set_lr("irradiance_mlp", args.N_iter_ignore_approximated_radiance)

		ith_train_time = time.time() - ith_train_start_time
		elapsed_time += ith_train_time

		if time_limit_in_sec > 0 and elapsed_time > time_limit_in_sec:
			print("%f sec is over" % time_limit_in_sec, elapsed_time)
			run_test_dataset(i, render_factor=4)
			save_file(i)
			break

		# new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
		# for param_group in optimizer.param_groups:
		#     param_group['lr'] = new_lrate

		# Export weight
		if i % args.i_weights == 0:
			save_file(i)

		# export images
		if i % args.i_testset == 0 and i > 0:
			run_test_dataset(i, render_factor=4)

		global_step += 1

	path = os.path.join(basedir, expname, 'train_info_step_time.json')
	with open(str(path), "w") as f:
		data = {
			"training_time": elapsed_time,
			"global_step": global_step
		}
		json.dump(data, f, indent=4)

	#f.write("elapsed_time : %f\n" % training_time)
	#f.write("global_step : %d\n" % global_step)

	#except TimeoutError:


def render_only(args):
	# (0) Print train phase overview
	logger_dataset = load_logger("Dataset Info")
	logger_export = load_logger("Export Logger")
	use_instance_mask = args.instance_mask
	logger_dataset.info("Instance mask: " + str(use_instance_mask))
	logger_dataset.info("Instance mask encoding: " + str(args.instance_label_encoding))
	logger_dataset.info("Infer normal: " + str(args.infer_normal))
	logger_dataset.info("Learn normal from oracle: " + str(args.learn_normal_from_oracle))
	logger_dataset.info("Learn albedo from oracle: " + str(args.learn_albedo_from_oracle))

	add_object_mode = True

	# Load BRDF LUT
	brdf_lut_path = "../data/ibl_brdf_lut.png"
	brdf_lut = cv2.imread(brdf_lut_path)
	brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)

	brdf_lut = brdf_lut.astype(np.float32)
	brdf_lut /= 255.0
	brdf_lut = torch.tensor(brdf_lut).to(args.device)
	brdf_lut = brdf_lut.permute((2, 0, 1))

	# (1) Load dataset
	with time_measure("[1] Data load"):
		def load_dataset_split(split="train", do_logging=True, **kwargs):
			# create dataset config
			target_dataset = load_dataset(args.dataset_type, args.datadir, split=split, **kwargs)
			target_dataset.load_instance_label_mask = use_instance_mask

			# real data load using multiprocessing(torch DataLoader) --> load all at once
			# TODO : if dataset is too large, it may not be loaded at once.
			target_dataset.load_all_data(num_of_workers=1)
			if do_logging:
				logger_dataset.info(target_dataset)
			return target_dataset

		load_params = {
			"image_scale": args.image_scale,
			"load_image": False,
			"load_normal": add_object_mode,
			"load_depth": add_object_mode,
			"sample_length": args.sample_length,
			"coarse_radiance_number": args.coarse_radiance_number,
			"load_instance_label_mask": args.instance_mask,
			"near_plane": args.near_plane,
			"far_plane": args.far_plane,
			"load_depth_range_from_file": args.load_depth_range_from_file,
			"gamma_correct": args.gamma_correct
		}
		dataset = load_dataset_split("test", skip=1, **load_params)

		hwf = [dataset.height, dataset.width, dataset.focal]

		# move data to GPU side
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		dataset.to_tensor(args.device)

	# (2) Create log file / folder
	with time_measure("[2] Log file create"):
		# Create log dir and copy the config file
		basedir = args.basedir
		expname = args.expname
		export_config(args, basedir)

	# (3) Create nerf model
	with time_measure("[3] NeRFDecomp load"):
		# set instance label dimension
		label_encoder = None
		if use_instance_mask:
			label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding, args.instance_label_dimension)
			args.instance_label_dimension = label_encoder.get_dimension()
		else:
			args.instance_label_dimension = 0

		# create nerf model
		args.num_cluster = dataset.num_cluster
		render_kwargs_train, render_kwargs_test, start, elapsed_time, grad_vars, optimizer = create_IBLNeRF(args)
		render_kwargs_test['brdf_lut'] = brdf_lut
		global_step = start

		# update near / far plane
		bds_dict = dataset.get_near_far_plane()
		render_kwargs_train.update(bds_dict)
		render_kwargs_test.update(bds_dict)

		logger_render_options = load_logger("Render Kwargs")
		logs = ["[Render Kwargs (simple only)]"]
		for k, v in render_kwargs_train.items():
			if isinstance(v, (str, float, int, bool)):
				logs += ["\t-%s : %s" % (k, str(v))]
		logger_render_options.info("\n".join(logs))


	# (5) Main train loop
	K = dataset.get_focal_matrix()

	testsavedir = os.path.join(args.export_basedir, expname, 'testset_final')
	os.makedirs(testsavedir, exist_ok=True)

	print("Render additional mode!!")
	with torch.no_grad():
		render_decomp_path(
			dataset, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
			render_factor=2, init_basecolor=dataset.init_basecolor,
			calculate_normal_from_depth_map=args.calculate_all_analytic_normals,
			use_instance=use_instance_mask, label_encoder=label_encoder,
			approximate_radiance=True,
			add_object_mode=add_object_mode
		)
	from utils.video_export import export_as_video_all
	export_as_video_all(testsavedir)


if __name__ == '__main__':
	parser = recursive_config_parser()
	args = parser.parse_args()
	args.device = device

	if args.expname is None:
		expname = args.config.split("/")[-1]
		expname = expname.split(".")[0]
		args.expname = expname

	if args.export_basedir is None:
		args.export_basedir = args.basedir.replace("logs", "logs_eval")

	if args.render_only:
		render_only(args)
	else:
		train(args)
