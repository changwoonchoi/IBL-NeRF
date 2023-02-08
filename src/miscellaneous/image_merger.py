import cv2
import numpy as np


def merge_image(img0_path, disp0_path, img1_path, disp1_path):
	image0 = cv2.imread(img0_path)
	image1 = cv2.imread(img1_path)
	disp0 = cv2.imread(disp0_path)
	disp1 = cv2.imread(disp1_path)

	sum1 = np.sum(image1, axis=-1, keepdims=True)

	# only depth map is not enough...
	condition = np.logical_or((disp0 > disp1), (sum1 < 100))
	final_image = np.where(condition, image0, image1)
	cv2.imwrite('final_image_merged.png', final_image)


if __name__ == "__main__":
	base_dir = "../../logs/mitsuba/veach-ajar/onehot_equal_decompose/decompose/testset_000000/"
	img0_path = base_dir + "decomposed_1/rgb_1.png"
	disp0_path = base_dir + "decomposed_1/disp_1.png"
	img1_path = base_dir + "decomposed_0/rgb_0.png"
	disp1_path = base_dir + "decomposed_0/disp_0.png"
	merge_image(img0_path, disp0_path, img1_path, disp1_path)
