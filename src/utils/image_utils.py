import cv2
import numpy as np
import torch
import os
from PIL import Image
import torch


def convert_image_to_uint(image):
    x = np.copy(image)
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


def save_image_pil(images, file_path):
    image_pil = Image.fromarray(images)
    image_pil.save(file_path)

def save_pred_images(images, file_path):
    x = convert_image_to_uint(images)

    new_im = Image.fromarray(x)
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if file_path.endswith(".png"):
        new_im.save(file_path)
    else:
        new_im.save("%s.png" % file_path)

def save_pred_numpy(images, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(file_path, images)

def load_image_from_path(image_file_path, scale=1):
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    image = image.astype(np.float32)
    image /= 255.0

    return image

def load_image(path):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    if image.shape[-1] == 4:
        image = image[:, :, 0:3]
    image /= 255.0
    return image


def load_numpy_from_path(image_file_path, scale=1):
    image = np.load(image_file_path)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    image = image.astype(np.float32)

    return image


def texture(image, u, v):
    width, height, channel = image.shape
    u_d = (width - 1) * u
    v_d = (height - 1) * v
    u_d = torch.clip(u_d, 0, width - 1)
    v_d = torch.clip(v_d, 0, height - 1)
    u_i = u_d.to(torch.long)
    v_i = v_d.to(torch.long)
    u_f = u_d - u_i
    v_f = v_d - v_i

    value_00 = image[u_i+0, v_i+0, :]
    value_01 = image[u_i+0, v_i+1, :]
    value_10 = image[u_i+1, v_i+0, :]
    value_11 = image[u_i+1, v_i+0, :]

    value = value_00 * (1-u_f) * (1-v_f) + value_01 * (1-u_f) * v_f + value_10 * u_f * (1-v_f) + value_11 * u_f * v_f


def rgb_to_srgb(rgb):
    """ Convert an image from linear RGB to sRGB.
    :param rgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def srgb_to_rgb(srgb):
    """ Convert an image from sRGB to linear RGB.
    :param srgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def srgb_to_rgb_torch(srgb):
    """ Convert an image from sRGB to linear RGB.
    :param srgb: numpy array in range (0.0 to 1.0)
    """
    ret = torch.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret