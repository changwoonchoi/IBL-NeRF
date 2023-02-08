from utils.image_utils import load_image_from_path
import numpy as np
import matplotlib.pyplot as plt

image_path = "albedo_test.PNG"
image = load_image_from_path(image_path)
image = np.power(image, 1/2.2)

plt.imshow(image)
plt.show()