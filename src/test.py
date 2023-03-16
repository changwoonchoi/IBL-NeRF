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
    # (0) Print phase overview
    logger_dataset = load_logger("Dataset info")
    logger_export = load_logger("Export Logger")

    # (1) Load dataset
    with time_measure("[1] Data load"):
        def load_testset(do_logging=True, **kwargs):
            target_dataset = load_dataset(args.dataset_type, args.datadir, split="test", **kwargs)
            target_dataset.load_all_data(num_of_workers=1, editing_idx=kwargs.get("editing_idx", None))
            if do_logging:
                logger_dataset.info(target_dataset)
            return target_dataset

        if args.edit_intrinsic:
            editing_idx = args.editing_img_idx
        elif args.insert_object:
            editing_idx = args.inserting_img_idx
        else:
            editing_idx = None

        load_params = {
            "image_scale": args.image_scale,
            "coarse_radiance_number": args.coarse_radiance_number,
            "near_plane": args.near_plane,
            "far_plane": args.far_plane,
            "load_depth_range_from_file": args.load_depth_range_from_file,
            "gamma_correct": args.gamma_correct,
            "load_priors": False,

            "load_edit_intrinsic_mask": args.edit_intrinsic,
            "load_edit_albedo": args.edit_albedo_by_img,
            "load_edit_normal": args.edit_normal_by_img,
            "load_edit_irradiance": args.edit_irradiance_by_img,
            "load_edit_depth": args.edit_depth,

            "object_insert": args.insert_object,

            "editing_idx": editing_idx
        }

        dataset = load_testset(skip=1, **load_params)

        hwf = [dataset.height, dataset.width, dataset.focal]

        # Move data to GPU
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        dataset.to_tensor(args.device)

        # Load BRDF LUT
        brdf_lut_path = "../data/ibl_brdf_lut.png"
        brdf_lut = cv2.imread(brdf_lut_path)
        brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)

        brdf_lut = brdf_lut.astype(np.float32)
        brdf_lut /= 255.0
        brdf_lut = torch.tensor(brdf_lut).to(args.device)
        brdf_lut = brdf_lut.permute((2, 0, 1))

    # (2) Create log file / floder
    with time_measure("[2] Log file create"):
        # Creaste log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        export_config(args, args.export_basedir)

    # (3) Create nerf model
    with time_measure("[3] IBL-NeRF load"):
        _, render_kwargs_test, start, _, _, _ = create_IBLNeRF(args)
        global_step = start

        bds_dict = dataset.get_near_far_plane()
        render_kwargs_test.update(bds_dict)
        render_kwargs_test['brdf_lut'] = brdf_lut

        logger_render_options = load_logger("Render kwargs")
        logs = ["[Render Kwargs (simple only)]"]
        for k, v in render_kwargs_test.items():
            if isinstance(v, (str, float, int, bool)):
                logs += ["\t-%s : %s" % (k, str(v))]
        logger_render_options.info("\n".join(logs))

    # (4) Test
    K = dataset.get_focal_matrix()

    edit_params = {
        # editing options
        "edit_intrinsic": args.edit_intrinsic,
        "editing_img_idx": args.editing_img_idx,
        "num_edit_objects": args.num_edit_objects,
        "edit_roughness": args.edit_roughness,
        "edit_albedo": args.edit_albedo,
        "edit_normal": args.edit_normal,
        "edit_depth": args.edit_depth,
        "edit_albedo_by_img": args.edit_albedo_by_img,
        "edit_normal_by_img": args.edit_normal_by_img,
        "edit_roughness_by_img": args.edit_roughness_by_img,
        "edit_irradiance_by_img": args.edit_irradiance_by_img,
        "editing_target_roughness_list": args.editing_target_roughness_list,
        "editing_target_albedo_list": args.editing_target_albedo_list,
        "editing_target_irradiance_list": args.editing_target_irradiance_list,

        # inserting options
        "insert_object": args.insert_object,
        "inserting_img_idx": args.inserting_img_idx,
        "num_insert_objects": args.num_insert_objects,
        "inserting_target_roughness_list": args.inserting_target_roughness_list,
        "inserting_target_irradiance_list": args.inserting_target_irradiance_list,
        "inserting_target_albedo_list": args.inserting_target_albedo_list,
    }

    def run_test_dataset(_i, render_factor=1, **kwargs):
        testsavedir = os.path.join(args.export_basedir, expname, 'testset_{:06d}'.format(_i))
        os.makedirs(testsavedir, exist_ok=True)

        render_decomp_path(
            dataset, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=render_factor,
            approximate_radiance=True, **kwargs
        )

    with torch.no_grad():
        run_test_dataset(global_step, render_factor=1, **edit_params)

    logger_export.info("Done!")


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

    test(args)
