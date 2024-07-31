import os
from enum import Enum

# ----- Constants ----- #
project_dir = '/Users/nick_1/PycharmProjects/Western Summer Research/CoolingTowerSyntheticDataGenerator'  # on macbook

dust_icon_path = os.path.join(project_dir, 'synthetic_data_generator/image_files/dust1.png')
grate_mask_path = os.path.join(project_dir, 'synthetic_data_generator/image_files/metal_mask_v2.png')

data_base_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'  # on macbook
data_base_dir_ubuntu = '/mnt/storage_1/bell_5g_datasets/synth_datasets'  # ubuntu server
data_base_dir_windows = 'C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\synth_datasets'  # windows

src_data_dir = os.path.join(data_base_dir, 'src_images')
train_dataset_dir = os.path.join(data_base_dir, 'train')
test_dataset_dir = os.path.join(data_base_dir, 'test')

valid_hour_list = [
	'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
	'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
]


class Partition(Enum):
	TRAIN = 'train'
	TEST = 'test'


class Scenario(Enum):
	SC1 = 'scenario_1'
	SC2 = 'scenario_2'
	SC3 = 'scenario_3'
	SC4 = 'scenario_4'
