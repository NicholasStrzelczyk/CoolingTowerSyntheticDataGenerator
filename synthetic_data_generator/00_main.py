import time
from datetime import datetime

from synthetic_data_generator.A_timeline_generator import gen_time_series_data
from synthetic_data_generator.B_synthetic_data_generator import gen_synth_data
from synthetic_data_generator.C_segmentation_quantifier import quantify_scenario_fouling
from synthetic_data_generator.D_make_dataset_list import make_platform_ds_list
from synthetic_data_generator.utils.constants import Partition

if __name__ == '__main__':
	# hyperparameters
	num_data_days = 57
	cleaning_period = 12
	part = Partition.TRAIN
	dust_icon_path = './image_files/dust1.png'
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets_2'

	# ----- ----- ----- #
	print('Creating synthetic {} dataset...'.format(part.value))
	start_time = datetime.now()

	# step A: timeline generator
	gen_time_series_data(data_dir, part, num_data_days, cleaning_period)

	# step B: data generator
	gen_synth_data(data_dir, part, dust_icon_path)

	# step C: segmentation quantifier
	quantify_scenario_fouling(data_dir, part)

	# step D: make dataset list
	make_platform_ds_list(data_dir, part)

	# ----- ----- ----- #
	total_time = datetime.now() - start_time
	print('Script took {} to complete'.format(total_time))


