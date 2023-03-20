import os
import configargparse
import cv2
import torch
import numpy as np
import json
from piq import ssim, psnr


def calculate_metrics(gt_path, pred_path, dataset_type, target):
    metrics = {'ssim': [], 'psnr': [], 'mse': []}
    if dataset_type == "mitsuba":
        for i in range(100):
            pred_img = cv2.imread(os.path.join(pred_path, f"{args.target}_{i:03d}.png"))
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            pred_img = pred_img.astype(np.float32)
            pred_img /= 255.0

            gt_img = cv2.imread(os.path.join(gt_path, "test", f"{i+1}.png"))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = gt_img.astype(np.float32)
            gt_img /= 255.0

            pred_img = torch.Tensor(pred_img).to('cuda')[..., None]
            gt_img = torch.Tensor(gt_img).to('cuda')[..., None]

            pred_img = torch.permute(pred_img, (3, 2, 0, 1))
            gt_img = torch.permute(gt_img, (3, 2, 0, 1))

            metrics['ssim'].append(ssim(pred_img, gt_img).item())
            metrics['psnr'].append(psnr(pred_img, gt_img).item())
            metrics['mse'].append(torch.nn.MSELoss()(pred_img, gt_img).item())
        return metrics
    elif dataset_type == "bespoke":
        # SCENE = os.path.basename(gt_path)
        SCALE = 0.5
        with open(os.path.join(gt_path, "transforms.json")) as f:
            meta = json.load(f)
        total_dataset_len = len(meta['frames'])
        index_list_tmp = [i * 8 for i in range(total_dataset_len // 8 + 1)]
        index_list = [i for i in index_list_tmp if i < total_dataset_len]

        for i, index in enumerate(index_list):
            frame = meta['frames'][index]
            gt_image_file_path = os.path.join(gt_path, "images", os.path.split(frame["file_path"])[-1])
            gt_img = cv2.imread(gt_image_file_path)
            gt_img = cv2.resize(gt_img, None, fx=SCALE, fy=SCALE)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = gt_img.astype(np.float32)
            gt_img /= 255.0

            pred_img = cv2.imread(os.path.join(pred_path, f"{args.target}_{i:03d}.png"))
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            pred_img = pred_img.astype(np.float32)
            pred_img /= 255.0

            pred_img = torch.Tensor(pred_img).to('cuda')[..., None]
            gt_img = torch.Tensor(gt_img).to('cuda')[..., None]

            pred_img = torch.permute(pred_img, (3, 2, 0, 1))
            gt_img = torch.permute(gt_img, (3, 2, 0, 1))

            metrics['ssim'].append(ssim(pred_img, gt_img).item())
            metrics['psnr'].append(psnr(pred_img, gt_img).item())
            metrics['mse'].append(torch.nn.MSELoss()(pred_img, gt_img).item())
        return metrics
    else:
        raise ValueError("dataset_type must be either mitsuba or colmap")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("--gt", required=True, type=str, help="path to ground truth images")
    parser.add_argument("--pred", required=True, type=str, help="path to inferred images")
    parser.add_argument("--dataset_type", required=True, type=str, choices=["mitsuba", "bespoke"])
    parser.add_argument("--target", type=str, default="rgb", choices=["rgb", "radiance"])
    args = parser.parse_args()
    results = calculate_metrics(args.gt, args.pred, args.dataset_type, args.target)
    print(f"ssim: {np.mean(np.asarray(results['ssim']))}")
    print(f"psnr: {np.mean(np.asarray(results['psnr']))}")
    print(f"mse: {np.mean(np.asarray(results['mse']))}")
