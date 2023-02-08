from config_parser import config_parser
from pathlib import Path
import os


def load_all_include(config_file):
	parser = config_parser()
	args = parser.parse("--config %s" % config_file)
	path = Path(config_file)

	include = []
	if args.include:
		include.append(os.path.join(path.parent, args.include))
		return include + load_all_include(os.path.join(path.parent, args.include))
	else:
		return include

config_file = "../../configs/mitsuba/bedroom_test_3.txt"
include_files = load_all_include(config_file)
include_files = list(reversed(include_files))
print(include_files)
parser = config_parser(include_files)
args = parser.parse("--config %s" % config_file)
print(args)

# args = parser.parse("--config ../../configs/mitsuba/bedroom_test_2.txt")
# # args2 = parser.parse("--config %s" % args.include)
# parser2 = config_parser([args.include])
# args2 = parser2.parse("--config ../../configs/mitsuba/bedroom_test_2.txt")
# print(args2)
# #args = parser.parse("--config ../../configs/mitsuba/bedroom_test_2.txt")
# #print(args)