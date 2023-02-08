import cv2
from utils.image_utils import load_image
import os
from pathlib import Path
from pprint import pprint
import natsort

def export_as_video(images, basedir, name):
	size = images[0].shape[0:2]
	width = size[1]
	height = size[0]

	out = cv2.VideoWriter(os.path.join(basedir, name), cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

	for i in range(len(images)):
		img = images[i]
		out.write(img)
	out.release()
	cv2.destroyAllWindows()


def export_as_video_run():
	images = []
	basedir = "../../logs_eval/falcor/kitchen/sphere_video_size4/testset_final/"
	target = "rgb"
	for i in range(320):
		image = cv2.imread(basedir + target + '_{:03d}.png'.format(i))
		images.append(image)
	export_as_video(images, basedir, '%s.avi' % target)


def export_as_video_all_helper(basedir, target):
	image_paths = []
	images = []
	for path in Path(basedir).rglob('%s_*.png' % target):
		image_paths.append(path)
	image_paths = natsort.natsorted(image_paths)

	for image_path in image_paths:
		image = cv2.imread(str(image_path))
		images.append(image)
		print(image_path)

	dirname = os.path.join(basedir, "videos")
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	export_as_video(images, dirname, '%s.avi' % target)


def export_as_video_all(basedir):
	image_files = []
	for path in Path(basedir).rglob('*.png'):
		path = os.path.basename(path)
		path = path[0:-8]
		image_files.append(path)
	targets = set(image_files)
	for target in targets:
		export_as_video_all_helper(basedir, target)


#export_as_video_all()
