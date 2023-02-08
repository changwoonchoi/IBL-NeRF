from pathlib import Path
import re
import numpy as np
import natsort
import pprint
import json
import os


def find_images():
	basedir = "../../data/falcor/kitchen"
	image_name_list = []
	for path in Path(basedir).rglob("*.png"):
		image_name_list.append(str(path))
	image_name_list = natsort.natsorted(image_name_list)
	#print(len(image_name_list))
	#pprint.pprint(image_name_list)
	return image_name_list


def load_positions(path):
	parser = lambda s: np.array(s.replace("float3(", "").replace(")", "").split(",")).astype(np.float32)
	transforms = []
	positions = []
	rights = []
	ups = []
	forwards = []
	with open(path) as f:
		while True:
			line = f.readline()
			if len(line) == 0:
				break
			p = re.compile('float3\(.*?\)')
			result = p.findall(line)
			position = parser(result[0])
			target = parser(result[1])
			forward = normalize(target - position)
			up = parser(result[2])
			right = normalize(np.cross(forward, up))
			transform = get_lookat_matrix(position, target, up)
			transforms.append(transform)
			positions.append(position)
			rights.append(right)
			forwards.append(forward)
			ups.append(up)
			#print(transform)
			if not line:
				break
			#print(line)
	return transforms, positions, rights, ups, forwards


def normalize(mat):
	return mat / np.linalg.norm(mat)


def get_lookat_matrix(origin, target, up):
	forward = normalize(target - origin)
	right = normalize(np.cross(forward, up))
	up = normalize(np.cross(right, forward))

	matrix = np.array([
		[right[0], right[1], right[2], 0],
		[up[0], up[1], up[2], 0],
		[-forward[0], -forward[1], -forward[2], 0],
		[origin[0], origin[1], origin[2], 1]
	])
	# matrix = np.array([
	# 	[-up[0], -up[1], -up[2], 0],
	# 	[right[0], right[1], right[2], 0],
	# 	[-forward[0], -forward[1], -forward[2], 0],
	# 	[origin[0], origin[1], origin[2], 1]
	# ])
	return matrix


def export_custom():
	basedir = "../../data/falcor/kitchen"
	positions = load_positions(basedir+"/viewpoints.txt")
	images = find_images()

	N = len(images)
	camera_angle_x = 60 / 180 * np.pi
	width = 1280
	height = 720
	focal = .5 * width / np.tan(0.5 * camera_angle_x)
	camera_angle_y = np.arctan(0.5 * height / focal) * 2
	cx = width / 2
	cy = height / 2
	total_json_data = {
		"camera_angle_x": camera_angle_x,
		"camera_angle_y": camera_angle_y,
		"fl_x": focal,
		"fl_y": focal,
		"k1": 0,
		"k2": 0,
		"p1": 0,
		"p2": 0,
		"cx": cx,
		"cy": cy,
		"w": width,
		"h": height,
		"aabb_scale": 4
	}

	frames = []
	for i in range(N):
		frame_data = {}
		image = images[i].split("/")[-1]
		frame_data["file_path"] = "images/%s" % image
		frame_data["sharpness"] = 30
		frame_data["transform_matrix"] = positions[i].transpose().tolist()
		frames.append(frame_data)

	total_json_data["frames"] = frames

	with open(os.path.join(basedir, 'transforms.json'), 'w') as fp:
		json.dump(total_json_data, fp, indent=4)


def export_custom_2():
	basedir = "../../data/falcor/kitchen"
	transforms, positions, rights, ups, forwards = load_positions(basedir+"/viewpoints.txt")
	images = find_images()

	N = len(images)

	total_json_data = {}

	frames = []
	for i in range(N):
		frame_data = {}
		#image = images[i].split("/")[-1]
		#frame_data["file_path"] = "images/%s" % image
		#frame_data["sharpness"] = 30
		frame_data["position"] = positions[i].tolist()
		frame_data["right"] = rights[i].tolist()
		frame_data["up"] = ups[i].tolist()
		frame_data["forward"] = forwards[i].tolist()
		frame_data["transform"] = transforms[i].transpose().tolist()

		#frame_data["transform_matrix"] = positions[i].transpose().tolist()
		frames.append(frame_data)

	total_json_data["frames"] = frames

	with open(os.path.join(basedir, 'nerf_form_transforms.json'), 'w') as fp:
		json.dump(total_json_data, fp, indent=4)

export_custom_2()
