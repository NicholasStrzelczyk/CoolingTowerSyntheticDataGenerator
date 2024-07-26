import os
from enum import Enum

# ----- Constants ----- #
max_vignette_strength = 25.0  # vignette strength will start at this value
min_vignette_strength = 2.50  # smallest possible vignette strength

project_dir = '/Users/nick_1/PycharmProjects/Western Summer Research/UWO_Maitenance_Predictor'
dust_icon_path = os.path.join(project_dir, 'synth_data_maker/image_files/dust1.png')
grate_mask_path = os.path.join(project_dir, 'synth_data_maker/image_files/metal_mask_v2.png')

data_base_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
data_src_dir = os.path.join(data_base_dir, 'src_images')

hour_list = [
	'6am', '7am', '8am', '9am', '10am', '11am', '12pm',
	'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
]
csv_headers_sc1_sc2 = [
	'day', 'cleaning_occurred', 'growth_percent',
	'dust_rows', 'dust_cols', 'dust_x', 'dust_y', 'region'
]
csv_headers_sc3_sc4 = [
	'day', 'cleaning_occurred', 'growth_percent',
	'dust_rows', 'dust_cols', 'dust_x', 'dust_y', 'region'
]


class Partition(Enum):
	TRAIN = 'train'
	TEST = 'test'
