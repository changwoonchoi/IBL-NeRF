import cv2
import matplotlib.pyplot as plt


# Load BRDF LUT
brdf_lut_path = "../../data/ibl_brdf_lut.png"
brdf_lut = cv2.imread(brdf_lut_path)
brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)
plt.imshow(brdf_lut)
plt.show()