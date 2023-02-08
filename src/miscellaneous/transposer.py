import json
import os
import numpy as np


def set_transpose(basedir, split):
	with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
		meta = json.load(fp)

	for frame in meta["frames"]:
		transform = np.array(frame['transform'])
		frame['transform'] = transform.transpose().tolist()

	with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'w') as fp:
		json.dump(meta, fp, indent=4)


def set_transpose_all():
	target_scenes = ["living-room", "classroom", "dining-room", "bathroom"]
	target_splits = ["train", "test", "val"]
	for scene in target_scenes:
		for split in target_splits:
			print(scene, split)
			set_transpose("../../data/mitsuba/%s" % scene, split)

set_transpose_all()
