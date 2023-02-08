import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config_parser import config_parser
from dataset.dataset_interface import load_dataset

from utils.generator_utils import *
import matplotlib.pyplot as plt
from utils.timing_utils import *


def data_load_speed_evaluation():
	parser = config_parser()
	args = parser.parse_args()
	args.device = device

	# Load dataset
	dataset = load_dataset(args.dataset_type, args.datadir)
	with time_measure("data load traverse - all data load - multi"):
		dataset.load_all_data(10)
		plt.imshow(dataset.images[0])
		plt.show()

	dataset = load_dataset(args.dataset_type, args.datadir)
	with time_measure("data load traverse - all data load - single_worker"):
		dataset.load_all_data(1)
		plt.imshow(dataset.images[0])
		plt.show()

	return


def data_iteration_evaluation():
	parser = config_parser()
	args = parser.parse_args()
	args.device = device

	N_iter = 2000
	batch_size = 1024

	# Load dataset
	dataset = load_dataset(args.dataset_type, args.datadir)
	with time_measure("sample_generator_all_image_merged"):
		generator = sample_generator_all_image_merged(dataset, batch_size)
		for i in range(N_iter):
			sample = next(generator)

	dataset = load_dataset(args.dataset_type, args.datadir)
	with time_measure("sample_generator_exhaustive_single_image"):
		generator = sample_generator_exhaustive_single_image(dataset, batch_size)
		for i in range(N_iter):
			sample = next(generator)

	dataset = load_dataset(args.dataset_type, args.datadir)
	with time_measure("sample_generator_single_image"):
		generator = sample_generator_single_image(dataset, batch_size)
		for i in range(N_iter):
			sample = next(generator)

	return
