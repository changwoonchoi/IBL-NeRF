import configargparse
import os
from pathlib import Path


def load_all_include(config_file):
	parser = config_parser()
	args = parser.parse_args("--config %s" % config_file)
	path = Path(config_file)

	include = []
	if args.include:
		include.append(os.path.join(path.parent, args.include))
		return include + load_all_include(os.path.join(path.parent, args.include))
	else:
		return include


def recursive_config_parser():
	parser = config_parser()
	args = parser.parse_args()
	include_files = load_all_include(args.config)

	include_files = list(reversed(include_files))
	parser = config_parser(default_files=include_files)
	return parser


def config_parser(default_files=None):
	if default_files is not None:
		parser = configargparse.ArgumentParser(default_config_files=default_files)
	else:
		parser = configargparse.ArgumentParser()

	parser.add_argument('--config', is_config_file=True, help='config file path')
	parser.add_argument('--include', type=str, default=None, help='config file path')

	parser.add_argument("--expname", type=str, default=None, help='experiment name')
	parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
	parser.add_argument("--export_basedir", type=str, default=None, help='where to export images')
	parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

	# training options

	parser.add_argument("--calculate_in_linear_rgb", action="store_true", help="calculate in rgb linear space")

	parser.add_argument("--image_scale", type=float, default=1.0, help="image scale ex) 0.5 = half")
	parser.add_argument("--instance_mask", action="store_true", help='NeRF with instance mask')
	parser.add_argument("--instance_loss_weight", type=float, default=0.01, help='Instance loss weight')
	parser.add_argument("--instance_label_encoding", type=str, default="one_hot",
	                    help="how to encode instance label. one of single, one_hot, label_color")
	parser.add_argument("--instance_label_dimension", type=int, default=0, help="instance mask dimension")
	parser.add_argument("--use_instance_feature_layer", action="store_true", help='NeRF with instance_feature_layer(Zhi, 2021)')
	parser.add_argument("--use_basecolor_score_feature_layer", action="store_true", help='NeRF with basecolor score feature layer')
	parser.add_argument("--use_illumination_feature_layer", action="store_true", help='NeRF with illumination feature_layer(Zhi, 2021)')
	parser.add_argument("--load_depth_range_from_file", action="store_true", help='load_depth_range_from_file')

	parser.add_argument("--N_iter", type=int, default=200000, help="Total iteration num")
	parser.add_argument("--target_load_N_iter", type=int, default=-1, help="target_load_N_iter")

	parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
	parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
	parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
	parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
	parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
	                    help='batch size (number of random rays per gradient step)')
	parser.add_argument("--ray_sample", type=str, default="pixel")
	parser.add_argument("--N_depth_random_volume", type=int, default=256, help='N_depth_random_volume')

	parser.add_argument("--CE_weight_type", type=str, default=None, help='weight type in CE Loss, bg_weakened/adaptive/equal or mse')
	parser.add_argument("--N_iter_ignore_normal", type=int, default=15000, help="Ignore normal loss")
	parser.add_argument("--N_iter_ignore_depth", type=int, default=15000, help="Ignore depth loss")
	parser.add_argument("--N_iter_ignore_approximated_radiance", type=int, default=5000, help="Ignore normal loss")
	parser.add_argument("--N_iter_ignore_smooth", type=int, default=15000, help="Ignore smoothness loss")
	parser.add_argument("--N_iter_ignore_instancewise_constant", type=int, default=15000, help="Ignore instancewise constant loss")
	parser.add_argument("--N_iter_ignore_prior", type=int, default=10000, help="Ignore prior loss")


	parser.add_argument("--coarse_radiance_number", type=int, default=0, help='coarse_radiance_number')

	parser.add_argument("--beta_sparse_base", type=float, default=1., help="")
	parser.add_argument("--beta_res", type=float, default=1., help="")
	parser.add_argument("--beta_mod", type=float, default=1., help="")
	parser.add_argument("--beta_indirect", type=float, default=1., help="")
	parser.add_argument("--beta_render", type=float, default=1.)
	parser.add_argument("--beta_inferred_normal", type=float, default=0.1)
	parser.add_argument("--beta_albedo_render", type=float, default=1.)
	parser.add_argument("--beta_radiance_render", type=float, default=1.)
	parser.add_argument("--beta_inferred_depth", type=float, default=1.)
	parser.add_argument("--beta_instance", type=float, default=1.)
	parser.add_argument("--beta_instancewise_constant", type=float, default=0.1)
	parser.add_argument("--beta_sigma_depth", type=float, default=1)
	parser.add_argument("--beta_roughness_render", type=float, default=1)
	parser.add_argument("--beta_prior_albedo", type=float, default=0.01)
	parser.add_argument("--beta_prior_irradiance", type=float, default=0)
	parser.add_argument("--beta_irradiance_reg", type=float, default=0)

	parser.add_argument("--albedo_instance_constant", action='store_true')
	parser.add_argument("--irradiance_instance_constant", action='store_true')
	parser.add_argument("--color_independent_to_direction", action='store_true')

	parser.add_argument("--initialize_roughness", action='store_true')
	parser.add_argument("--freeze_roughness", action='store_true')
	parser.add_argument("--correct_depth_for_prefiltered_radiance_infer", action='store_true')

	parser.add_argument("--roughness_init", type=float, default=0.5)

	parser.add_argument("--infer_albedo_separate", action='store_true')
	parser.add_argument("--infer_roughness_separate", action='store_true')
	parser.add_argument("--infer_irradiance_separate", action='store_true')

	parser.add_argument("--roughness_smooth", action='store_true')
	parser.add_argument("--albedo_smooth", action='store_true')
	parser.add_argument("--irradiance_smooth", action='store_true')
	parser.add_argument("--gamma_correct", action='store_true')
	parser.add_argument("--freeze_radiance", action='store_true')

	parser.add_argument("--albedo_multiplier", type=float, default=1.0, help='fraction of img taken for central crops')
	parser.add_argument("--load_priors", action='store_true')
	parser.add_argument("--prior_type", type=str, default='bell', help='bell or ting')
	parser.add_argument("--albedo_prior_type", type=str, default='rgb', help='rgb or chrom')

	parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
	parser.add_argument("--lrate_env_map", type=float, default=5e-4, help='learning rate for env_map')

	parser.add_argument("--lrate_decay", type=int, default=250,
	                    help='exponential learning rate decay (in 1000 steps)')
	parser.add_argument("--chunk", type=int, default=1024*16,
	                    help='number of rays processed in parallel, decrease if running out of memory')
	parser.add_argument("--netchunk", type=int, default=1024 * 64,
	                    help='number of pts sent through network in parallel, decrease if running out of memory')
	parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
	parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
	parser.add_argument("--ft_path", type=str, default=None,
	                    help='specific weights npy file to reload for coarse network')

	# rendering options
	parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
	parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
	parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
	parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
	parser.add_argument("--i_embed", type=int, default=0,
	                    help='set 0 for default positional encoding, -1 for none')
	parser.add_argument("--multires", type=int, default=10,
	                    help='log2 of max freq for positional encoding (3D location)')
	parser.add_argument("--multires_views", type=int, default=4,
	                    help='log2 of max freq for positional encoding (2D direction)')
	parser.add_argument("--raw_noise_std", type=float, default=0.,
	                    help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

	parser.add_argument("--render_only", action='store_true',
	                    help='do not optimize, reload weights and render out render_poses path')
	parser.add_argument("--render_test", action='store_true',
	                    help='render the test set instead of render_poses path')
	parser.add_argument("--render_factor", type=int, default=1,
	                    help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	parser.add_argument("--render_decompose", action='store_true', help="render decomposed instance in test phase")
	parser.add_argument("--alpha_th", type=float, default=.0, help='decompose alpha thredhold')
	parser.add_argument("--instance_th", type=float, default=.0, help='decompose instance thredhold')

	parser.add_argument("--decompose_target", type=str, default="0", help='decompose target instance ids')
	parser.add_argument("--decompose_mode", type=str, default="binary", help='decompose mode one of all or binary')

	parser.add_argument("--infer_normal", action='store_true', help='infer normal from NeRF')
	parser.add_argument("--infer_normal_at_surface", action='store_true', help='infer normal from surface')
	parser.add_argument("--infer_normal_target", type=str, default="normal_map_from_sigma_gradient", help='decompose mode one of all or binary')

	parser.add_argument("--infer_depth", action='store_true', help='infer depth using additional MLP')
	parser.add_argument("--use_radiance_linear", action='store_true', help='is_radiance_linear')
	parser.add_argument("--infer_visibility", action='store_true', help='infer_visibility')

	parser.add_argument("--use_gradient_for_incident_radiance", action='store_true', help='stop_gradient_for_incident_radiance')
	parser.add_argument("--monte_carlo_integration_method", type=str, default="surface", help='decompose mode one of all or binary')
	parser.add_argument("--use_environment_map", action='store_true', help='use_environment_map')

	parser.add_argument("--learn_normal_from_oracle", action='store_true', help='learn_normal_from_oracle')
	parser.add_argument("--learn_albedo_from_oracle", action='store_true', help='learn_albedo_from_oracle')

	parser.add_argument("--calculate_irradiance_from_gt", action='store_true', help='calculate_irradiance_from_gt')
	parser.add_argument("--calculate_roughness_from_gt", action='store_true', help='calculate_roughness_from_gt')
	parser.add_argument("--calculate_albedo_from_gt", action='store_true', help='calculate_albedo_from_gt')
	parser.add_argument("--roughness_exp_coefficient", type=float, default=1.0, help='roughness_exp_coefficient of img taken for central crops')

	parser.add_argument("--calculate_all_analytic_normals", action='store_true', help='calculate_analytic_normals')
	parser.add_argument("--calculating_normal_type", type=str, default='ground_truth', help='types of analytic normal, one of [normal_map_from_sigma_gradient,normal_map_from_sigma_gradient_surface, normal_map_from_depth_gradient, normal_map_from_depth_gradient_direction, normal_map_from_depth_gradient_epsilon, normal_map_from_depth_gradient_direction_epsilon, ground_truth]')

	parser.add_argument("--N_hemisphere_sample_sqrt", type=int, default=16, help='N_hemisphere_sample_sqrt')
	parser.add_argument("--N_envmap_size", type=int, default=16, help='N_envmap_size')
	parser.add_argument("--use_monte_carlo_integration", action='store_true', help='use_monte_carlo_integration')
	parser.add_argument("--depth_map_from_ground_truth", action='store_true', help='depth_map_from_ground_truth')

	parser.add_argument("--lut_coefficient", type=str, default="F", help='lut coefficient type, F or F0')

	# training options
	parser.add_argument("--precrop_iters", type=int, default=0, help='number of steps to train on central crops')
	parser.add_argument("--precrop_frac", type=float, default=.5, help='fraction of img taken for central crops')
	parser.add_argument("--epsilon_for_numerical_normal", type=float, default=.01, help='epsilon_for_numerical_normal')
	parser.add_argument("--epsilon_direction_for_numerical_normal", type=float, default=.005, help='epsilon_direction_for_numerical_normal')
	parser.add_argument("--time_limit_in_minute", type=float, default=-1, help='time_limit_in_hour')

	# test options
	parser.add_argument("--extract_mesh", action='store_true', help='extract mesh')

	parser.add_argument("--train_depth_from_ground_truth", action='store_true', help='train_depth_from_ground_truth')


	# dataset options
	parser.add_argument("--dataset_type", type=str, default='mitsuba', help='dataset type. we support only mitsuba for current version.')
	parser.add_argument("--testskip", type=int, default=8,
	                    help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')


	# clevr options
	parser.add_argument("--near_plane", type=float, default=1, help='near_plane')
	parser.add_argument("--far_plane", type=float, default=20, help='far_plane')


	## blender flags
	parser.add_argument("--white_bkgd", action='store_true',
	                    help='set to render synthetic data on a white bkgd (always use for dvoxels)')
	parser.add_argument("--half_res", action='store_true',
	                    help='load blender synthetic data at 400x400 instead of 800x800')

	## llff flags
	parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
	parser.add_argument("--no_ndc", action='store_true',
	                    help='do not use normalized device coordinates (set for non-forward facing scenes)')
	parser.add_argument("--lindisp", action='store_true',
	                    help='sampling linearly in disparity rather than depth')
	parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
	parser.add_argument("--llffhold", type=int, default=8,
	                    help='will take every 1/N images as LLFF test set, paper uses 8')

	# logging/saving options
	parser.add_argument("--summary_step", type=int, default=100)
	parser.add_argument("--i_print", type=int, default=100,
	                    help='frequency of console printout and metric loggin')
	parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
	parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
	parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
	parser.add_argument("--i_video", type=int, default=50000, help='frequency of render_poses video saving')

	# editing options
	parser.add_argument("--edit_intrinsic", action='store_true', help='edit_intrinsic')
	parser.add_argument("--editing_img_idx", type=int, default=0, help='index of image to edit')

	parser.add_argument("--edit_roughness", action='store_true', help='edit_roughness')
	parser.add_argument("--edit_albedo", action='store_true', help='edit_albedo')
	parser.add_argument("--edit_normal", action='store_true', help='edit_normal')

	parser.add_argument("--num_edit_objects", type=int, default=1, help='num_edit_objects')

	parser.add_argument("--edit_albedo_by_img", action='store_true', help='edit_albedo_by_img')
	parser.add_argument("--edit_normal_by_img", action='store_true', help='edit_normal_by_img')
	parser.add_argument("--edit_irradiance_by_img", action='store_true', help='edit_irradiance_by_img')

	parser.add_argument("--editing_target_roughness_list", type=float, action="append")
	parser.add_argument("--editing_target_albedo_list", action="append")
	parser.add_argument("--editing_target_irradiance_list", type=float, action="append")

	# inserting options
	parser.add_argument("--insert_object", action='store_true', help='insert_object')
	parser.add_argument("--inserting_img_idx", type=int, default=0, help='index of image to insert')

	parser.add_argument("--num_insert_objects", type=int, default=1, help='num_insert_objects')

	parser.add_argument("--inserting_target_roughness_list", type=float, action="append")
	parser.add_argument("--inserting_target_albedo_list", type=float, action="append")
	parser.add_argument("--inserting_target_irradiance_list", type=float, action="append")

	return parser


def export_config(args, basedir):
	# Create log dir and copy the config file
	expname = args.expname

	os.makedirs(os.path.join(basedir, expname), exist_ok=True)
	f = os.path.join(basedir, expname, 'args.txt')
	with open(f, 'w') as file:
		for arg in sorted(vars(args)):
			attr = getattr(args, arg)
			file.write('{} = {}\n'.format(arg, attr))
	if args.config is not None:
		f = os.path.join(basedir, expname, 'config.txt')
		with open(f, 'w') as file:
			file.write(open(args.config, 'r').read())