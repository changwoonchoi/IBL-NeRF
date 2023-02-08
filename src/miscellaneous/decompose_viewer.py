from utils.image_utils import *
import matplotlib.pyplot as plt
import numpy as np


I = load_image_from_path("../../../mitsuba/kitchen/train/1.png")
I2 = srgb_to_rgb(I)
I3 = I2 / np.clip(1-I2, 0.0001, 1)
# I = srgb_to_rgb(I)
# R = load_image_from_path("../../../mitsuba/kitchen/train/1_bell_r.png")
# R = srgb_to_rgb(R)
# S = load_image_from_path("../../../mitsuba/kitchen/train/1_bell_s.png")
# S = srgb_to_rgb(S)
# I2 = R * S
# I2 = rgb_to_srgb(I2)

plt.imshow(I)
plt.figure()
plt.imshow(I2)
plt.figure()
plt.imshow(I3)
plt.show()
