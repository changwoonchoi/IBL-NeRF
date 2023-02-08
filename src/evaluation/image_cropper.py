from PIL import Image





from PIL import Image, ImageDraw
import os
import glob
import numpy as np

SIZE = 256
SIZE_CROP = 64

_crop_data = {
    "kitchen": ["230_114_294_178", "469_271_533_335"],
}

target_colors = ["#FF0000", "#FFFF00", "#FFA500", "#FFFF00"]


def str_to_tuple(t):
    return tuple((map(int,t.split("_"))))


def tuple_to_str(t):
    return "_".join(map(str,t))

def crop(image, crop_data):
    image_pil = Image.fromarray((image * 255).astype('uint8'))
    cropped_image = image_pil.crop(crop_data)
    cropped_image = cropped_image.resize((256, 256))
    return np.asarray(cropped_image)

def draw_image(image, crop_sizes):
    image_pil = Image.fromarray((image * 255).astype('uint8'))
    image_pil_draw = ImageDraw.Draw(image_pil)
    for i, crop_size in enumerate(crop_sizes):
        image_pil_draw.rectangle(crop_size, width=5, outline=target_colors[i])
    return np.asarray(image_pil)


if __name__ == "__main__":
    #crop_image_in_folder_randomly("../../result_0412_2/scale_2_time_40", 6)
    merge_image("../../result_0414_compare_epsilon_opt/scale_2_time_40", _crop_data)

    #crop_image_in_folder("../../result_0412_2/scale_2_time_40", crop_data)