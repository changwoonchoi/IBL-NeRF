from nerf_models.nerf_renderer_helper import *
import time
from tqdm import tqdm, trange
import os
import imageio
from utils.label_utils import *
from torch.nn.functional import normalize
from torch.autograd import Variable
from nerf_models.microfacet import *
from torchvision import transforms
DEBUG = False
from utils.depth_to_normal_utils import depth_to_normal_image_space

from nerf_models.normal_from_depth import *
# from nerf_models.normal_from_sigma import *

import matplotlib.pyplot as plt
from utils.math_utils import get_TBN
from nerf_models.microfacet import Microfacet
from utils.math_utils import *

gamma = 2.2
epsilon_srgb = 1e-12


def rgb_to_srgb(x):
	return torch.pow(x + epsilon_srgb, 1.0/gamma)


def tonemap_reinherd(x):
	return x / (x + 1)


def high_dynamic_range_radiance_f(x):
	return F.relu(x)


def raw2outputs_simple(raw, z_vals, rays_d, coarse_radiance_number=3, detach=False, is_radiance_sigmoid=True):
	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	sigma = raw2sigma(raw[..., 0], dists)

	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	if detach:
		weights = weights.detach()

	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)
		N += 3

	return radiance_map, coarse_radiance_maps


def raw2outputs_neigh(rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, is_radiance_sigmoid):
	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, rays_d, network_fn)

	# distances
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]

	roughness = torch.sigmoid(raw[..., 4])  # (N_rand * 8, N_sample)
	albedo = torch.sigmoid(raw[..., 1:4])  # (N_rand * 8, N_sample, 3)
	irradiance = radiance_f(raw[..., 5])  # (N_rand * 8, N_sample, )

	roughness_map = torch.sum(weights * roughness, -1)  # (N_rand * 8, )
	albedo_map = torch.sum(weights[..., None] * albedo, 1)  # (N_rand * 8, 3)
	irradiance_map = torch.sum(weights * irradiance, -1)  # (N_rand * 8, )

	if is_radiance_sigmoid:
		radiance_to_ldr = lambda x:x
	else:
		radiance_to_ldr = tonemap
	#radiance_to_ldr = lambda x: None if x is None else torch.pow(radiance_to_ldr_temp(x) + epsilon_srgb, 1.0 / gamma)


	results = {}
	# don't calculate gradient for neighborhood pixels
	results["roughness_map"] = roughness_map
	results["albedo_map"] = albedo_map
	results["irradiance_map"] = radiance_to_ldr(irradiance_map)
	results["weights"] = weights
	return results


def raw2outputs_depth(rays_o, rays_d, z_vals, network_query_fn, network_fn, raw_noise_std):
	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, None, network_fn)

	# distances
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	visibility_cum = torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)
	visibility = visibility_cum[:, -1]
	weights = sigma * visibility_cum[:, :-1]

	depth_map = torch.sum(weights * z_vals, -1)

	results = {}
	results["depth_map"] = depth_map
	results["weights"] = weights
	results['visibility'] = visibility
	return results


def raw2outputs(rays_o, rays_d, z_vals, z_vals_constant,
				network_query_fn, network_fn,
				raw_noise_std=0., pytest=False,
				is_depth_only=False,
				infer_normal=False,
				infer_normal_at_surface=False,
				normal_mlp=None,
				albedo_mlp=None,
				roughness_mlp=None,
				irradiance_mlp=None,
				brdf_lut=None,
				epsilon=0.01,
				epsilon_direction=0.01,
				gt_values=None,
				target_normal_map_for_radiance_calculation="ground_truth",
				calculate_irradiance_from_gt=False,
				calculate_albedo_from_gt = False,
				calculate_roughness_from_gt=False,
				**kwargs):
	"""Transforms model's predictions to semantically meaningful values.
	Args:
		raw: [num_rays, num_samples along ray, 4]. Prediction from model.
		- instance_label_dimension==0: [num_rays, num_samples along ray, 4]. Prediction from model. (R,G,B,a)
		- instance_label_dimension >0: [num_rays, num_samples along ray, 10]. (R,G,B,a,instance_label_dimension)
		z_vals: [num_rays, num_samples along ray]. Integration time.
		rays_d: [num_rays, 3]. Direction of each ray.
		is_instance_label_logit: export instance label as logit
	Returns:
		rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
		disp_map: [num_rays]. Disparity map. Inverse of depth map.
		acc_map: [num_rays]. Sum of weights along each ray.
		weights: [num_rays, num_samples]. Weights assigned to each sampled color.
		depth_map: [num_rays]. Estimated distance to object.
	"""

	# radiance & gamma correct setting
	is_radiance_sigmoid = not kwargs.get('use_radiance_linear', False)
	gamma_correct = kwargs.get('gamma_correct', False)

	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	if is_depth_only:
		return raw2outputs_depth(rays_o, rays_d, z_vals, network_query_fn, network_fn, raw_noise_std)
	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, rays_d, network_fn)

	# distances
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

		# Overwrite randomly sampled data if pytest
		if pytest:
			np.random.seed(0)
			noise = np.random.rand(*list(raw[..., 0].shape)) * raw_noise_std
			noise = torch.Tensor(noise)

	assert (not kwargs.get("load_edit_intrinsic_mask") or not kwargs.get("insert_object")), "edit_intrinsic and insert_object cannot be True at the same time"
	# intrinsic edit
	if kwargs.get("edit_intrinsic", False):
		num_edit_objects = kwargs.get("num_edit_objects")
		assert num_edit_objects > 0, "num_edit_objects must be greater than 0"
		mask_img = gt_values["edit_intrinsic_mask"][:, 0]
		masks = []
		for object_idx in range(num_edit_objects):
			# rgb value of ith object is [10(i+1), 10(i+1), 10(i+1)]
			masks.append(torch.logical_and(11 * (object_idx + 1) / 255. > mask_img, mask_img > 9 * (object_idx + 1) / 255.))
		mask_all = mask_img > 0
	# object insert
	elif kwargs.get("insert_object", False):
		num_insert_objects = kwargs.get("num_insert_objects")
		assert num_insert_objects > 0, "num_insert_objects must be greater than 0"
		mask_img = gt_values["object_insert_mask"][:, 0]
		masks = []
		for object_idx in range(num_insert_objects):
			# rgb value of ith object is [10(i+1), 10(i+1), 10(i+1)]
			masks.append(torch.logical_and(11 * (object_idx + 1) / 255. > mask_img, mask_img > 9 * (object_idx + 1) / 255.))
		mask_all = mask_img > 0

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	weights_detached = weights.detach()

	# (2) get depth / disp / acc map
	depth_map = torch.sum(weights * z_vals, -1)
	target_depth_map = depth_map
	if kwargs.get("depth_map_from_ground_truth", False):
		target_depth_map = gt_values["depth"][..., 0]
	if kwargs.get("edit_intrinsic", False) and kwargs.get("edit_depth", False):
		target_depth_map[mask_all] = gt_values["edit_depth"][..., 0][mask_all]
	if kwargs.get("insert_object", False):
		target_depth_map[mask_all] = gt_values["object_insert_depth"][..., 0][mask_all]

	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)

	# (3) get surface point surface_x
	x_surface = rays_o + rays_d * target_depth_map[..., None]
	x_surface.detach_()

	# (4A) calculate normal from sigma gradient or read ground_truth value
	inferred_normal_map = None
	if infer_normal:
		if infer_normal_at_surface:
			inferred_normal_map = network_query_fn(x_surface[..., None, :], None, normal_mlp)
			inferred_normal_map = 2 * torch.sigmoid(inferred_normal_map) - 1
			inferred_normal_map.squeeze_(-2)
		else:
			inferred_normal_raw = network_query_fn(pts, None, normal_mlp)
			inferred_normal = 2 * torch.sigmoid(inferred_normal_raw) - 1
			inferred_normal_map = torch.sum(weights_detached[..., None] * inferred_normal, -2)

	target_normal_map = None


	# (5) other values
	albedo = torch.sigmoid(raw[..., 1:4])
	albedo_map = torch.sum(weights_detached[..., None] * albedo, -2)

	roughness = torch.sigmoid(raw[..., 4])
	roughness_map = torch.sum(weights_detached * roughness, -1)

	irradiance = radiance_f(raw[..., 5])
	irradiance_map = torch.sum(weights_detached * irradiance, -1)

	if albedo_mlp is not None:
		raw_albedo = network_query_fn(pts, None, albedo_mlp)
		albedo = torch.sigmoid(raw_albedo[..., 0:3])
		albedo_map = torch.sum(weights_detached[..., None] * albedo, -2)

	if roughness_mlp is not None:
		raw_roughness = network_query_fn(pts, None, roughness_mlp)
		roughness = torch.sigmoid(raw_roughness[..., 0])
		roughness_map = torch.sum(weights_detached * roughness, -1)

	if irradiance_mlp is not None:
		raw_irradiance = network_query_fn(pts, None, irradiance_mlp)
		irradiance = torch.sigmoid(raw_irradiance[..., 0])
		irradiance_map = torch.sum(weights_detached * irradiance, -1)

	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	instance_map = None

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(network_fn.coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights_detached[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)

		N += 3

	target_albedo_map = albedo_map
	if calculate_albedo_from_gt:
		target_albedo_map = gt_values["albedo"]

	target_roughness_map = roughness_map
	if calculate_roughness_from_gt:
		target_roughness_map = gt_values["roughness"][...,0]

	target_irradiance_map = irradiance_map[...,None]
	if calculate_irradiance_from_gt:
		target_irradiance_map = gt_values["irradiance"]


	target_binormal_map = None
	target_tangent_map = None
	approximated_radiance_map = None
	specular_map = None
	diffuse_map = None
	min_irradiance_map = None
	max_irradiance_map = None
	visibility_average_map = None
	n_dot_v = None
	reflected_radiance_map = None
	prefiltered_reflected_map = None
	reflected_coarse_radiance_map = []
	if kwargs.get('approximate_radiance', False):

		# calculate normal only approximate radiance
		if target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient":
			normal_map_from_sigma_gradient = get_normal_from_sigma_gradient(pts, weights_detached, network_query_fn, network_fn)
			target_normal_map = normal_map_from_sigma_gradient
		elif target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient_surface":
			normal_map_from_sigma_gradient_surface = get_normal_from_sigma_gradient_surface(x_surface, network_query_fn, network_fn)
			target_normal_map = normal_map_from_sigma_gradient_surface
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient":
			normal_map_from_depth_gradient = get_normal_from_depth_gradient(rays_o, rays_d, network_query_fn, network_fn, z_vals)
			normal_map_from_depth_gradient.detach_()
			target_normal_map = normal_map_from_depth_gradient
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_epsilon":
			with torch.no_grad():
				normal_map_from_depth_gradient_epsilon = get_normal_from_depth_gradient_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon)
			target_normal_map = normal_map_from_depth_gradient_epsilon
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction":
			normal_map_from_depth_gradient_direction = get_normal_from_depth_gradient_direction(rays_o, rays_d, network_query_fn, network_fn, z_vals)
			normal_map_from_depth_gradient_direction.detach_()
			target_normal_map = normal_map_from_depth_gradient_direction
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction_epsilon":
			with torch.no_grad():
				normal_map_from_depth_gradient_direction_epsilon = get_normal_from_depth_gradient_direction_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon_direction)
			target_normal_map = normal_map_from_depth_gradient_direction_epsilon
		elif target_normal_map_for_radiance_calculation == "ground_truth":
			target_normal_map = normalize(2 * gt_values["normal"] - 1, dim=-1)
		elif target_normal_map_for_radiance_calculation == "inferred_normal_map":
			target_normal_map = inferred_normal_map
		else:
			raise ValueError

		# Edit!
		if kwargs.get("edit_intrinsic", False):
			# Override normal
			if kwargs.get("edit_normal", False):
				gt_normal_map = normalize(2 * gt_values["edit_normal"] - 1, dim=-1)
				target_normal_map[mask_all] = gt_normal_map[mask_all]
			# Override albedo
			assert not kwargs.get("edit_albedo", False) or not len(kwargs.get("editing_target_albedo_list", [])) == 0, "Cannot load both edit_albedo and editing_target_albedo_list"
			if kwargs.get("edit_albedo", False):
				if kwargs.get("edit_albedo_by_img", False):
					target_albedo_map[mask_all] = gt_values["edit_albedo"][mask_all]
				else:
					for object_idx in range(num_edit_objects):
						target_albedo_map[masks[object_idx]] = torch.Tensor(kwargs.get("editing_target_albedo_list", [])[object_idx * 3:object_idx * 3 + 3])
			# Override roughness
			assert not kwargs.get("edit_roughness", False) or not len(kwargs.get("editing_target_roughness_list", [])) == 0, "Cannot load both edit_roughness and editing_target_roughness_list"
			if kwargs.get("edit_roughness", False):
				if kwargs.get("edit_roughness_by_img"):
					target_roughness_map[mask_all] = gt_values["edit_roughness"][mask_all][0]
				else:
					for object_idx, editing_roughness in enumerate(kwargs.get("editing_target_roughness_list", [])):
						target_roughness_map[masks[object_idx]] = editing_roughness
			# TODO: other components. e.g.) irradiance, ...

		elif kwargs.get("insert_object", False):
			gt_normal_map = normalize(2 * gt_values["object_insert_normal"] - 1, dim=-1)
			target_normal_map[mask_all] = gt_normal_map[mask_all]
			assert kwargs.get("num_insert_objects", 0) == len(kwargs.get("inserting_target_roughness_list", [])), "Number of inserting objects does not match number of roughness values"
			assert kwargs.get("num_insert_objects", 0) == len(kwargs.get("inserting_target_albedo_list", [])) / 3, "Number of inserting objects does not match number of albedo values"
			for object_idx in range(kwargs.get("num_insert_objects", 0)):
				target_roughness_map[masks[object_idx]] = kwargs.get("inserting_target_roughness_list", [])[object_idx]
				if kwargs.get("inserting_target_irradiance_list", [])[object_idx] > 0:
					target_irradiance_map[masks[object_idx]] = kwargs.get("inserting_target_irradiance_list", [])[object_idx]
				target_albedo_map[masks[object_idx]] = torch.Tensor(kwargs.get("inserting_target_albedo_list", [])[3 * object_idx:3 * object_idx + 3])

		n_dot_v = torch.sum(-rays_d * target_normal_map, -1)
		n_dot_v = torch.clip(n_dot_v, 0, 1)

		# (7) calculate color from split-sum approximation

		# grid_sample input is  [-1, 1] x [-1, 1]
		BRDF_2D_LUT_uv = torch.stack([2 * n_dot_v - 1, 2 * target_roughness_map - 1], -1)
		envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None, :, None, ...], align_corners=True)
		envBRDF = envBRDF.permute((0, 2, 3, 1))
		envBRDF = envBRDF.squeeze()

		# dielectric
		F0 = torch.tensor([0.04, 0.04, 0.04])
		F0 = F0.repeat(*depth_map.shape, 1)
		target_metallic_map = (1-target_roughness_map)[..., None]
		F0 = F0 * (1-target_metallic_map) + target_albedo_map * target_metallic_map

		envBRDF_coefficient1 = envBRDF[..., 0]
		envBRDF_coefficient0 = envBRDF[..., 1]
		envBRDF_coefficient1 = torch.stack(3 * [envBRDF_coefficient1], -1)
		fresnel_map = fresnel_schlick_roughness(n_dot_v, F0, target_roughness_map)
		if kwargs.get('lut_coefficient') == 'F':
			specular_map = fresnel_map * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
		elif kwargs.get('lut_coefficient') == 'F0':
			specular_map = F0 * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
		else:
			raise ValueError
		reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
		reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]

		if not kwargs.get('use_gradient_for_incident_radiance', False):

			with torch.no_grad():
				reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
				reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, is_radiance_sigmoid=is_radiance_sigmoid)

				prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)
		else:
			reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
			reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, is_radiance_sigmoid=is_radiance_sigmoid)

			prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)

		N_pref = len(reflected_coarse_radiance_map) + 1
		if kwargs.get("correct_depth_for_prefiltered_radiance_infer", False):
			depth_0 = (kwargs["far"] + kwargs["near"]) * 0.5
			depth_map_detached = depth_map.detach()
			mipmap_level = roughness_map * depth_map_detached / depth_0[..., 0]
			mipmap_level = torch.clip(mipmap_level, 0, 1)
		else:
			mipmap_level = roughness_map

		mipmap_index1 = (mipmap_level * (N_pref - 1)).long()
		mipmap_index1 = torch.clip(mipmap_index1, 0, N_pref - 1)
		mipmap_index2 = torch.clip(mipmap_index1 + 1, 0, N_pref - 1)
		mipmap_remainder = ((mipmap_level * (N_pref - 1)) - mipmap_index1)[..., None]
		prefiltered_reflected_map = \
			(1-mipmap_remainder) * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index1] +\
			mipmap_remainder * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index2]

		diffuse_map = (1 - fresnel_map) * (1-target_metallic_map) * target_albedo_map * target_irradiance_map
		specular_map = specular_map * prefiltered_reflected_map
		approximated_radiance_map = diffuse_map + specular_map


	# Organize results
	results = {}

	if is_radiance_sigmoid:
		ldr_f = lambda x: x
	else:
		ldr_f = lambda x: tonemap_reinherd(x)

	if gamma_correct:
		gamma_correct_f = lambda x: rgb_to_srgb(x)
	else:
		gamma_correct_f = lambda x: x
	#print(is_radiance_sigmoid, "RADIANCE SIGMOID", gamma_correct, "GAMMA CORRECT")

	output_f = lambda x: x if x is None else gamma_correct_f(ldr_f(x))
	albedo_f = lambda x: x if x is None else gamma_correct_f(x)

	results["color_map"] = output_f(approximated_radiance_map)
	results["radiance_map"] = output_f(radiance_map)
	for k in range(len(coarse_radiance_maps)):
		results["radiance_map_%d" % (k + 1)] = output_f(coarse_radiance_maps[k])
	for k in range(len(reflected_coarse_radiance_map)):
		results["reflected_coarse_radiance_map_%d" % (k + 1)] = output_f(reflected_coarse_radiance_map[k])

	results["irradiance_map"] = output_f(target_irradiance_map)
	results["min_irradiance_map"] = output_f(min_irradiance_map)
	results["max_irradiance_map"] = output_f(max_irradiance_map)
	results["reflected_radiance_map"] = output_f(reflected_radiance_map)
	results["prefiltered_reflected_map"] = output_f(prefiltered_reflected_map)

	results["albedo_map"] = albedo_f(target_albedo_map)
	results["roughness_map"] = target_roughness_map
	results["specular_map"] = output_f(specular_map)
	results["diffuse_map"] = output_f(diffuse_map)
	results["n_dot_v_map"] = n_dot_v
	results["instance_map"] = instance_map
	results["visibility_average_map"] = visibility_average_map

	results["inferred_normal_map"] = inferred_normal_map
	results["target_normal_map"] = target_normal_map
	results["target_binormal_map"] = target_binormal_map
	results["target_tangent_map"] = target_tangent_map

	results["disp_map"] = disp_map
	results["acc_map"] = acc_map
	results["depth_map"] = depth_map
	results["target_depth_map"] = target_depth_map

	results["weights"] = weights

	return results


def raw2outputs_additional(rays_o, rays_d, z_vals, z_vals_constant,
				network_query_fn, network_fn,
				raw_noise_std=0., pytest=False,
				gt_values=None,
				**kwargs):
	is_radiance_sigmoid = not kwargs.get('use_radiance_linear', False)

	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, rays_d, network_fn)

	# distances
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

		# Overwrite randomly sampled data if pytest
		if pytest:
			np.random.seed(0)
			noise = np.random.rand(*list(raw[..., 0].shape)) * raw_noise_std
			noise = torch.Tensor(noise)

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	#weights_detached = weights.detach()

	# (2) get depth / disp / acc map
	depth_map = torch.sum(weights * z_vals, -1)

	# Radiance & Additional radiance
	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	N = 9
	coarse_radiance_maps = []
	for i in range(network_fn.coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)

		N += 3

	# depth
	gt_depth_map = gt_values["depth"][..., 0]
	mask = (depth_map < gt_depth_map) | (gt_depth_map == 0)
	target_depth_map = torch.where(mask, depth_map, gt_depth_map)

	# (3) get surface point surface_x
	x_surface = rays_o + rays_d * target_depth_map[..., None]
	x_surface.detach_()

	# (4A) calculate normal from sigma gradient or read ground_truth value
	target_normal_map = normalize(2 * gt_values["normal"] - 1, dim=-1)

	reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
	reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]

	reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
	reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, radiance_f=radiance_f)
	prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)

	#roughness_map = kwargs["roughness"]
	roughness_map = torch.ones_like(depth_map) * kwargs["roughness"]
	#print(roughness_map, "ROUGHNESS!!")

	N_pref = len(reflected_coarse_radiance_map) + 1
	mipmap_index1 = (roughness_map * (N_pref - 1)).long()
	mipmap_index1 = torch.clip(mipmap_index1, 0, N_pref - 1)
	mipmap_index2 = torch.clip(mipmap_index1 + 1, 0, N_pref - 1)
	mipmap_remainder = ((roughness_map * (N_pref - 1)) - mipmap_index1)[..., None]
	prefiltered_reflected_map = \
		(1 - mipmap_remainder) * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index1] + \
		mipmap_remainder * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index2]

	#reflected_radiance_map = torch.ones_like(reflected_radiance_map) * roughness + (1-roughness) * reflected_radiance_map
	new_radiance_map = torch.where(mask[...,None], radiance_map, prefiltered_reflected_map)

	# Organize results
	results = {}
	results["color_map"] = new_radiance_map
	results["radiance_map"] = radiance_map
	results["weights"] = weights

	return results


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False,
				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
				pytest=False, **kwargs):
	"""Volumetric rendering.
	Args:
	  ray_batch: array of shape [batch_size, ...]. All information necessary
		for sampling along a ray, including: ray origin, ray direction, min
		dist, max dist, and unit-magnitude viewing direction.
	  network_fn: function. Model for predicting RGB and density at each point
		in space.
	  network_query_fn: function used for passing queries to network_fn.
	  N_samples: int. Number of different times to sample along each ray.
	  retraw: bool. If True, include model's raw, unprocessed predictions.
	  lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
	  perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
		random points in time.
	  N_importance: int. Number of additional times to sample along each ray.
		These samples are only passed to network_fine.
	  network_fine: "fine" network with same spec as network_fn.
	  white_bkgd: bool. If True, assume a white background.
	  raw_noise_std: ...
	  verbose: bool. If True, print more debugging info.
	Returns:
	  rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
	  disp_map: [num_rays]. Disparity map. 1 / depth.
	  acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
	  raw: [num_rays, num_samples, 4]. Raw predictions from model.
	  rgb0: See rgb_map. Output for coarse model.
	  disp0: See disp_map. Output for coarse model.
	  acc0: See acc_map. Output for coarse model.
	  z_std: [num_rays]. Standard deviation of distances along ray for each
		sample.
	"""

	# (1) Sample positions
	N_rays = ray_batch.shape[0]
	rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
	viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
	bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
	near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

	t_vals = torch.linspace(0., 1., steps=N_samples)
	if not lindisp:
		z_vals = near * (1. - t_vals) + far * (t_vals)
	else:
		z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

	z_vals = z_vals.expand([N_rays, N_samples])

	if perturb > 0.:
		# get intervals between samples
		mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		upper = torch.cat([mids, z_vals[..., -1:]], -1)
		lower = torch.cat([z_vals[..., :1], mids], -1)
		# stratified samples in those intervals
		t_rand = torch.rand(z_vals.shape)

		# Pytest, overwrite u with numpy's fixed random numbers
		if pytest:
			np.random.seed(0)
			t_rand = np.random.rand(*list(z_vals.shape))
			t_rand = torch.Tensor(t_rand)

		z_vals = lower + (upper - lower) * t_rand

	z_vals_constant = z_vals
	result = raw2outputs(
		rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, pytest, near=near, far=far, **kwargs
	)

	# (2) need importance sampling
	if N_importance > 0:
		weights = result["weights"]
		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
		z_samples = z_samples.detach()
		run_fn = network_fn if network_fine is None else network_fine

		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
		result_fine = raw2outputs(
			rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, run_fn, raw_noise_std, pytest, near=near, far=far, **kwargs
		)

		for k, v in result.items():
			result_fine[k + "0"] = result[k]

		result = result_fine

	if N_importance > 0:
		result['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

	result = {k: v for k, v in result.items() if v is not None}

	if kwargs.get("infer_depth", False):
		inferred_depth_map = network_query_fn(rays_o[..., None, :], viewdirs, kwargs["depth_mlp"])
		inferred_depth_map = F.relu(inferred_depth_map[..., 0])
		inferred_depth_map.squeeze_()
		result["inferred_depth_map"] = inferred_depth_map

	for k in result:
		if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")

	return result


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
	"""Render rays in smaller minibatches to avoid OOM.
	"""
	all_ret = {}
	gt_values = kwargs.get("gt_values", None)
	N = rays_flat.shape[0]
	for i in range(0, N, chunk):
		gt_values_ith = {}
		if gt_values is not None:
			for k in gt_values.keys():
				gt_values_ith[k] = gt_values[k][i:min(i+chunk, N)]
		kwargs["gt_values"] = gt_values_ith
		ret = render_rays(rays_flat[i:i + chunk], **kwargs)
		for k in ret:
			if k not in all_ret:
				all_ret[k] = []
			all_ret[k].append(ret[k])

	# all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
	for k in all_ret:
		all_ret[k] = torch.cat(all_ret[k], dim=0)
	return all_ret


def render_decomp(
		H, W, K, chunk=1024 * 32, rays=None, c2w=None, near=0., far=1.,
		c2w_staticcam=None, is_depth_only=False, **kwargs
):
	"""Render rays
	Args:
	  H: int. Height of image in pixels.
	  W: int. Width of image in pixels.
	  focal: float. Focal length of pinhole camera.
	  chunk: int. Maximum number of rays to process simultaneously. Used to
		control maximum memory usage. Does not affect final results.
	  rays: array of shape [2, batch_size, 3]. Ray origin and direction for
		each example in batch.
	  c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
	  near: float or array of shape [batch_size]. Nearest distance for a ray.
	  far: float or array of shape [batch_size]. Farthest distance for a ray.
	  use_viewdirs: bool. If True, use viewing direction of a point in space in model.
	  c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
	   camera while using other c2w argument for viewing directions.
	Returns:
	  rgb_map: [batch_size, 3]. Predicted RGB values for rays.
	  disp_map: [batch_size]. Disparity map. Inverse of depth.
	  acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
	  extras: dict with everything returned by render_rays().
	"""
	if c2w is not None:
		# special case to render full image
		rays_o, rays_d = get_rays(H, W, K, c2w)
	else:
		# use provided ray batch
		rays_o, rays_d = rays

	viewdirs = rays_d
	if c2w_staticcam is not None:
		# special case to visualize effect of viewdirs
		rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
	viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
	viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

	sh = rays_d.shape  # [..., 3]

	# Create ray batch
	rays_o = torch.reshape(rays_o, [-1, 3]).float()
	rays_d = torch.reshape(rays_d, [-1, 3]).float()

	near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
	rays = torch.cat([rays_o, rays_d, near, far], -1)
	rays = torch.cat([rays, viewdirs], -1)

	# Render and reshape
	all_ret = batchify_rays(rays, chunk, is_depth_only=is_depth_only, **kwargs)
	for k in all_ret:
		k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
		all_ret[k] = torch.reshape(all_ret[k], k_sh)
	return all_ret


from dataset.dataset_interface import NerfDataset


def render_decomp_path(
		dataset_test: NerfDataset, hwf, K, chunk, render_kwargs, savedir=None, render_factor=0,
		gt_values=None, **kwargs
):
	H, W, focal = hwf
	render_poses = dataset_test.poses

	if render_factor != 0:
		# Render downsampled for speed
		H = H // render_factor
		W = W // render_factor
		focal = focal / render_factor

	K = np.array([
		[focal, 0, 0.5 * W],
		[0, focal, 0.5 * H],
		[0, 0, 1]
	]).astype(np.float32)

	results = {}

	def append_result(render_decomp_results, key_name, index, out_name):
		if key_name not in render_decomp_results:
			return
		result_image = render_decomp_results[key_name]
		if result_image is None:
			return
		if out_name not in results:
			results[out_name] = []
		if "normal" in out_name or 'tangent' in out_name:
			result_image = (result_image + 1) * 0.5
		elif "depth" in key_name:
			# depth to disp
			result_image = result_image / (dataset_test.far * 0.1)
			result_image = 1. / torch.max(1e-10 * torch.ones_like(result_image), result_image)

		results[out_name].append(result_image.cpu().numpy())
		if savedir is not None:
			result_image_8bit = to8b(results[out_name][-1])

			filename = os.path.join(savedir, (out_name + '_{:03d}.png').format(index))
			imageio.imwrite(filename, result_image_8bit)

	for i, c2w in enumerate(tqdm(render_poses)):

		gt_values = dataset_test.get_resized_normal_albedo(render_factor, i)
		for k in gt_values.keys():
			gt_values[k] = torch.reshape(gt_values[k], [-1, gt_values[k].shape[-1]])
		results_i = render_decomp(
			H, W, K, chunk=chunk, c2w=c2w[:3, :4], gt_values=gt_values, **render_kwargs, **kwargs
		)
		append_result(results_i, "color_map", i, "rgb")
		append_result(results_i, "radiance_map", i, "radiance")
		for k in range(render_kwargs["coarse_radiance_number"]):
			append_result(results_i, "radiance_map_%d" % (k+1), i, "radiance_%d" % (k+1))
			append_result(results_i, "reflected_coarse_radiance_map_%d" % (k + 1), i, "reflected_coarse_radiance_%d" % (k + 1))


		append_result(results_i, "irradiance_map", i, "irradiance")
		append_result(results_i, "max_irradiance_map", i, "max_irradiance")
		append_result(results_i, "min_irradiance_map", i, "min_irradiance")

		append_result(results_i, "albedo_map", i, "albedo")
		append_result(results_i, "reflected_radiance_map", i, "reflected_radiance")
		append_result(results_i, "prefiltered_reflected_map", i, "prefiltered_reflected")

		append_result(results_i, "roughness_map", i, "roughness")
		append_result(results_i, "specular_map", i, "specular")
		append_result(results_i, "diffuse_map", i, "diffuse")
		append_result(results_i, "n_dot_v_map", i, "n_dot_v")

		append_result(results_i, "inferred_normal_map", i, "inferred_normal_map")
		append_result(results_i, "target_normal_map", i, "target_normal_map")
		append_result(results_i, "target_binormal_map", i, "target_binormal_map")
		append_result(results_i, "target_tangent_map", i, "target_tangent_map")
		append_result(results_i, "visibility_average_map", i, "visibility_average_map")

		append_result(results_i, "inferred_depth_map", i, "inferred_disp")

		append_result(results_i, "disp_map", i, "disp")
		append_result(results_i, "depth_map", i, "depth")
		append_result(results_i, "target_depth_map", i, "target_depth")


		if "depth_map" in results_i:
			depth_image = results_i["depth_map"]
			results_i["normal_map_from_depth_map"] = depth_to_normal_image_space(depth_image, c2w[:3, :4], K)
			append_result(results_i, "normal_map_from_depth_map", i, "normal_from_depth")

	for k, v in results.items():
		results[k] = np.stack(v, 0)
	return results
