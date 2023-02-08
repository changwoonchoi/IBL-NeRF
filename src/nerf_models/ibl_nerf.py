import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.logging_utils
from nerf_models.positional_embedder import get_embedder
import os
from networks.MLP import *
from nerf_models.envmap import EnvironmentMap


# Model
class IBLNeRF(nn.Module):
    def __init__(
            self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4],
            use_illumination_feature_layer=False,
            use_instance_feature_layer=False,
            coarse_radiance_number=0,
            is_color_independent_to_direction=True
    ):
        """
        NeRFDecomp Model
        params:
            D: Network Depth
            W: MLP dim
            input_ch: dim of x (position) (3 for R^3)
            input_ch_views: dim of d (direction) (3 for R^3 vector)
            output_ch:
            skips: list of layer numbers that are concatenated with x (position)
            use_instance_label: use instance label
            instance_label_dimension: dimension of instance_label
            K: number of base colors
            use_illumination_feature_layer: use additional layer following Shuaifeng Zhi et al.
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_illumination_feature_layer = use_illumination_feature_layer
        self.use_instance_feature_layer = use_instance_feature_layer

        self.positions_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.feature_linear = nn.Linear(W, W)

        # only x dependent
        self.sigma_linear = nn.Linear(W, 1)

        self.albedo_feature_linear = nn.Linear(W, W // 2)
        self.albedo_linear = nn.Linear(W // 2, 3)

        self.roughness_linear = nn.Linear(W, 1)

        self.irradiance_feature_linear = nn.Linear(W, W // 2)
        self.irradiance_linear = nn.Linear(W // 2, 1)

        # x, d dependent
        self.radiance_linear = nn.Linear(W, 3)
        self.coarse_radiance_number = coarse_radiance_number

        self.additional_radiance_feature_linear = nn.ModuleList(
            [nn.Linear(W, W // 2) for _ in range(coarse_radiance_number)]
        )
        self.additional_radiance_linear = nn.ModuleList(
            [nn.Linear(W // 2, 3) for _ in range(coarse_radiance_number)]
        )

        self.is_color_independent_to_direction = is_color_independent_to_direction
        self.freeze_radiance = False
        self.freeze_roughness = False


    def __str__(self):
        logs = ["[NeRFDecomp"]
        logs += ["\t- depth : {}".format(self.D)]
        logs += ["\t- width : {}".format(self.W)]
        logs += ["\t- input_ch : {}".format(self.input_ch)]
        logs += ["\t- use_illumination_feature_layer : {}".format(self.use_illumination_feature_layer)]
        return "\n".join(logs)

    def forward_freezed(self, x):
        with torch.no_grad():
            if x.shape[-1] == self.input_ch + self.input_ch_views:
                input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            else:
                input_pts = x
                input_views = None
            h = input_pts

            # (1) position
            for i, l in enumerate(self.positions_linears):
                # print(self.positions_linears[i].parameters(), i, "Position linears")
                h = self.positions_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], dim=-1)
                # print(h, i, "H at position!!!!!1")

            # (2) dependent only to position
            # (2-1) sigma
            sigma = self.sigma_linear(h)

            if input_views is None:
                return sigma

        # (2-2) albedo
        albedo_feature = self.albedo_feature_linear(h)
        albedo_feature = F.relu(albedo_feature)
        albedo = self.albedo_linear(albedo_feature)

        if self.freeze_roughness:
            with torch.no_grad():
                # (2-3) roughness
                roughness = self.roughness_linear(h)
        else:
            # (2-3) roughness
            roughness = self.roughness_linear(h)

        # (2-4) irradiance
        irradiance_feature = self.irradiance_feature_linear(h)
        irradiance_feature = F.relu(irradiance_feature)
        irradiance = self.irradiance_linear(irradiance_feature)

        with torch.no_grad():

            # (3) position + direction
            if not self.is_color_independent_to_direction:
                feature = self.feature_linear(h)
                h = torch.cat([feature, input_views], dim=-1)
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

            radiance = self.radiance_linear(h)
            ret = [sigma, albedo, roughness, irradiance, radiance]

            for i, l in enumerate(self.additional_radiance_feature_linear):
                radiance_i = self.additional_radiance_feature_linear[i](h)
                radiance_i = F.relu(radiance_i)
                radiance_i = self.additional_radiance_linear[i](radiance_i)
                ret.append(radiance_i)

        ret = torch.cat(ret, dim=-1)

        return ret

    def forward_not_freezed(self, x):
        if x.shape[-1] == self.input_ch + self.input_ch_views:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
            input_views = None
        h = input_pts

        # (1) position
        for i, l in enumerate(self.positions_linears):
            # print(self.positions_linears[i].parameters(), i, "Position linears")
            h = self.positions_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)
            # print(h, i, "H at position!!!!!1")

        # (2) dependent only to position
        # (2-1) sigma
        sigma = self.sigma_linear(h)

        if input_views is None:
            return sigma

        # (2-2) albedo
        albedo_feature = self.albedo_feature_linear(h)
        albedo_feature = F.relu(albedo_feature)
        albedo = self.albedo_linear(albedo_feature)

        # (2-3) roughness
        roughness = self.roughness_linear(h)

        # (2-4) irradiance
        irradiance_feature = self.irradiance_feature_linear(h)
        irradiance_feature = F.relu(irradiance_feature)
        irradiance = self.irradiance_linear(irradiance_feature)

        # (3) position + direction
        if not self.is_color_independent_to_direction:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

        radiance = self.radiance_linear(h)
        ret = [sigma, albedo, roughness, irradiance, radiance]

        for i, l in enumerate(self.additional_radiance_feature_linear):
            radiance_i = self.additional_radiance_feature_linear[i](h)
            radiance_i = F.relu(radiance_i)
            radiance_i = self.additional_radiance_linear[i](radiance_i)
            ret.append(radiance_i)

        ret = torch.cat(ret, dim=-1)

        return ret

    def forward(self, x):
        #return self.forward_not_freezed(x)
        if self.freeze_radiance:
            return self.forward_freezed(x)
        else:
            return self.forward_not_freezed(x)

def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        output = []
        for i in range(0, inputs.shape[0], chunk):
            output_chunk = fn(inputs[i:i + chunk])
            output.append(output_chunk)
        output = torch.cat(output, dim=0)
        return output
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Prepares inputs and applies network 'fn'.
    """
    #inputs.requires_grad = True
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_IBLNeRF(args):
    """
    Instantiate IBL-NeRF Model
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    logger = utils.logging_utils.load_logger("IBL-NeRF Loader")

    input_ch_views = 0
    embeddirs_fn = None
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    skips = [4]
    model = IBLNeRF(
        D=args.netdepth, W=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views, skips=skips,
        use_illumination_feature_layer=args.use_illumination_feature_layer,
        use_instance_feature_layer=args.use_instance_feature_layer,
        coarse_radiance_number=args.coarse_radiance_number,
        is_color_independent_to_direction=args.color_independent_to_direction
    ).to(args.device)
    logger.info(model)

    grad_vars = []
    grad_vars.append({'params': model.parameters(), 'name': 'coarse'})

    model_fine = None
    if args.N_importance > 0:
        model_fine = IBLNeRF(
            D=args.netdepth, W=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views, skips=skips,
            use_illumination_feature_layer=args.use_illumination_feature_layer,
            use_instance_feature_layer=args.use_instance_feature_layer,
            coarse_radiance_number=args.coarse_radiance_number,
            is_color_independent_to_direction=args.color_independent_to_direction
        ).to(args.device)
        logger.info("NeRFDecomp fine model")
        logger.info(model_fine)
        grad_vars.append({'params': model_fine.parameters(), 'name': 'fine'})

    # Depth MLP
    depth_mlp = None
    if args.infer_depth:
        depth_mlp = PositionDirectionMLP(D=args.netdepth, W=args.netwidth,
                                         input_ch=input_ch, input_ch_views=input_ch_views,
                                         out_ch=1, skips=skips)
        grad_vars.append({'params': depth_mlp.parameters(), 'name': 'depth_mlp'})

    visibility_mlp = None
    if args.infer_visibility:
        visibility_mlp = PositionDirectionMLP(D=args.netdepth, W=args.netwidth,
                                         input_ch=input_ch, input_ch_views=input_ch_views,
                                         out_ch=1, skips=skips)
        grad_vars.append({'params': depth_mlp.parameters(), 'name': 'visibility_mlp'})

    # Normal MLP
    normal_mlp = None
    if args.infer_normal:
        normal_mlp = PositionMLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, out_ch=3, skips=skips)
        grad_vars.append({'params': normal_mlp.parameters(), 'name': 'normal_mlp'})

    albedo_mlp = None
    if args.infer_albedo_separate:
        albedo_mlp = PositionMLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, out_ch=3, skips=skips)
        grad_vars.append({'params': albedo_mlp.parameters(), 'name': 'albedo_mlp'})

    roughness_mlp = None
    if args.infer_roughness_separate:
        roughness_mlp = PositionMLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, out_ch=1, skips=skips)
        grad_vars.append({'params': roughness_mlp.parameters(), 'name': 'roughness_mlp'})

    irradiance_mlp = None
    if args.infer_irradiance_separate:
        irradiance_mlp = PositionMLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, out_ch=1, skips=skips)
        grad_vars.append({'params': irradiance_mlp.parameters(), 'name': 'irradiance_mlp'})

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
    )

    env_map = None
    if args.use_environment_map:
        env_map = EnvironmentMap(n=args.N_envmap_size)
        grad_vars.append({'params': env_map.emission, 'name': 'env_map', 'lr': args.lrate_env_map})

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # optimizer_depth = torch.optim.Adam(params=depth_mlp.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    elapsed_time = 0

    # Load Checkpoint
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    elif args.target_load_N_iter > 0:
        ckpts = [os.path.join(basedir, expname, '{:06d}.tar'.format(args.target_load_N_iter))]
    else:
        ckpts = [os.path.join (basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info('Found ckpts: ' + str(ckpts))

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info('Reloading from ' + str(ckpt_path))
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        elapsed_time = ckpt.get('elapsed_time', 0)

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        model.load_state_dict(ckpt['network_fn_state_dict'])
        if args.infer_depth:
            depth_mlp.load_state_dict(ckpt['depth_mlp'])
        if args.infer_normal:
            normal_mlp.load_state_dict(ckpt['normal_mlp'])
        if args.infer_albedo_separate and 'albedo_mlp' in ckpt:
            albedo_mlp.load_state_dict(ckpt['albedo_mlp'])
        if args.infer_roughness_separate and 'roughness_mlp' in ckpt:
            roughness_mlp.load_state_dict(ckpt['roughness_mlp'])
        if args.infer_irradiance_separate and 'irradiance_mlp' in ckpt:
            irradiance_mlp.load_state_dict(ckpt['irradiance_mlp'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.use_environment_map:
            env_map.emission.data = ckpt['env_map']

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'ndc': False,
        'lindisp': args.lindisp,
        "depth_mlp": depth_mlp,
        "visibility_mlp": visibility_mlp,
        "normal_mlp": normal_mlp,
        "albedo_mlp": albedo_mlp,
        "roughness_mlp": roughness_mlp,
        "irradiance_mlp": irradiance_mlp,
        "infer_depth": args.infer_depth,
        "infer_visibility": args.infer_visibility,
        "infer_normal": args.infer_normal,
        "infer_normal_at_surface": args.infer_normal_at_surface,
        "coarse_radiance_number": args.coarse_radiance_number,
        "use_monte_carlo_integration": args.use_monte_carlo_integration,
        "use_gradient_for_incident_radiance": args.use_gradient_for_incident_radiance,
        "use_radiance_linear": args.use_radiance_linear,
        "gamma_correct": args.gamma_correct,
        "monte_carlo_integration_method": args.monte_carlo_integration_method,
        'use_environment_map': args.use_environment_map,
        "env_map": env_map,
        "lut_coefficient": args.lut_coefficient,
        "depth_map_from_ground_truth": args.depth_map_from_ground_truth,
        "target_normal_map_for_radiance_calculation": args.calculating_normal_type,
        "calculate_albedo_from_gt": args.calculate_albedo_from_gt,
        "calculate_roughness_from_gt": args.calculate_roughness_from_gt,
        "calculate_irradiance_from_gt": args.calculate_irradiance_from_gt,
        "epsilon": args.epsilon_for_numerical_normal,
        "epsilon_direction": args.epsilon_direction_for_numerical_normal,
        "N_hemisphere_sample_sqrt": args.N_hemisphere_sample_sqrt,
        "roughness_exp_coefficient": args.roughness_exp_coefficient,
        "albedo_multiplier": args.albedo_multiplier,
        "correct_depth_for_prefiltered_radiance_infer": args.correct_depth_for_prefiltered_radiance_infer
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0

    return render_kwargs_train, render_kwargs_test, start, elapsed_time, grad_vars, optimizer
