import numpy as np
import glob
import imageio
import json


def find_representative_irradiance_value(dataset_type: str, room_name: str):
	if dataset_type == 'mitsuba':
		bell_irradiance_files = glob.glob(
			'../../data/mitsuba_no_transparent_with_prior/{}/train/*_bell_s.png'.format(
				room_name))
		ting_irradiance_files = glob.glob(
			'../../data/mitsuba_no_transparent_with_prior/{}/train/*_ting_s.png'.format(
				room_name))
	elif dataset_type == 'falcor':
		bell_irradiance_files = glob.glob(
			'../../data/falcor/{}/*_bell_s.png'.format(room_name))
		ting_irradiance_files = glob.glob(
			'../../data/falcor/{}/*_ting_s.png'.format(room_name))
	elif dataset_type == 'replica':
		bell_irradiance_files = glob.glob(
			'../../data/replica/{}/train/*_bell_s.png'.format(room_name))
		ting_irradiance_files = glob.glob(
			'../../data/replica/{}/train/*_bell_s.png'.format(room_name))
	else:
		raise ValueError

	bell_irradiances = []
	ting_irradiances = []

	for irradiance_file in bell_irradiance_files:
		bell_irradiances.append(imageio.imread(irradiance_file) / 255.)
	for irradiance_file in ting_irradiance_files:
		ting_irradiances.append(imageio.imread(irradiance_file) / 255.)

	bell_irradiances = np.stack(bell_irradiances, axis=0)
	ting_irradiances = np.stack(ting_irradiances, axis=0)

	mean_bell = np.mean(bell_irradiances)
	mean_ting = np.mean(ting_irradiances)
	# median_bell = np.median(bell_irradiances)
	# median_ting = np.median(ting_irradiances)

	mean = {'bell': mean_bell, 'ting': mean_ting}
	# median = {'bell': median_bell, 'ting': median_ting}
	return mean  # , median

import os

if __name__ == "__main__":
	#replica scenes
	replica_scenes = os.listdir('../../data/replica')
	for room in replica_scenes:
		print("replica {} processing".format(room))
		irradiance_mean = find_representative_irradiance_value('replica', room)
		with open('../../data/replica/{}/avg_irradiance.json'.format(room), "w") as f:
			data = {
				"mean_bell": float(irradiance_mean['bell']),
				"mean_ting": float(irradiance_mean['ting'])
			}
			json.dump(data, f)
	# mitsuba scenes
	# mitsuba_rooms = ['bathroom', 'bathroom2', 'bedroom', 'classroom', 'dining-room', 'kitchen', 'living-room', 'living-room-2', 'living-room-3', 'staircase', 'veach-ajar', 'veach_door_simple']
	# # rooms = ['kitchen']
	# for room in mitsuba_rooms:
	# 	print("mitsuba {} processing".format(room))
	# 	irradiance_mean = find_representative_irradiance_value('mitsuba', room)
	# 	with open('../../data/mitsuba_no_transparent_with_prior/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)

	# falcor_rooms = ['kitchen', 'living-room-2']
	#
	# # falcor scenes
	# for room in falcor_rooms:
	# 	print("falcor {} processing".format(room))
	# 	irradiance_mean = find_representative_irradiance_value('falcor', room)
	# 	with open('../../data/falcor/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)
