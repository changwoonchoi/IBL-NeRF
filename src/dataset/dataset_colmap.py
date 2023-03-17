import json
from dataset.dataset_interface import NerfDataset
from utils.image_utils import *


class ColmapDataset(NerfDataset):
    def __init__(self, basedir, **kwargs):
        super().__init__("colmap", **kwargs)
        self.scene_name = basedir.split("/")[-1]

        if self.load_priors:
            with open(os.path.join(basedir, 'avg_irradiance.json'), 'r') as fp:
                f = json.load(fp)
                self.prior_irradiance_mean = f["mean_" + self.prior_type]

        with open(os.path.join(basedir, 'transforms.json'.format(self.split)), 'r') as fp:
            self.meta = json.load(fp)

        self.basedir = basedir

        self.skip = kwargs.get("skip", 1)
        if self.split == "train":
            self.skip = 1

        self.camera_angle_x = float(self.meta['camera_angle_x'])

        self.original_height = self.meta["h"]
        self.original_width = self.meta["w"]

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)
        self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)

        total_dataset_len = len(self.meta['frames'])
        if self.split == "train":
            index_list_tmp = [i * 8 + j + 1 for i in range(total_dataset_len // 8 + 1) for j in range(7)]
            self.index_list = [i for i in index_list_tmp if i < total_dataset_len]
        elif self.split in ["val", "test"]:
            index_list_tmp = [i * 8 for i in range(total_dataset_len // 8 + 1)]
            self.index_list = [i for i in index_list_tmp if i < total_dataset_len]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        sample = {}

        frame = self.meta['frames'][::self.skip][self.index_list[index]]
        image_file_path = os.path.join(self.basedir, "images", os.path.split(frame["file_path"])[-1])
        prior_albedo_file_path = os.path.join(
            self.basedir, "images", os.path.split(frame["file_path"])[-1][:-4] + "_{}_r.png".format(self.prior_type)
        )
        prior_irradiance_file_path = os.path.join(
            self.basedir, "images", os.path.split(frame["file_path"])[-1][:-4] + "_{}_s.png".format(self.prior_type)
        )

        if self.load_image:
            sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
        if self.load_priors:
            sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
            sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)

        pose = np.array(frame['transform_matrix']).astype(np.float32)
        sample["pose"] = pose

        return sample

    def get_test_render_poses(self):
        return None
