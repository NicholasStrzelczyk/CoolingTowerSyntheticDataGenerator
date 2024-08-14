import os
import shutil
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from synthetic_data_generator.utils.constants import Partition

if __name__ == '__main__':
	# hyperparameters
	num_scs = 10
	part = Partition.TRAIN
	old_dir = '/Users/nick_1/Bell_5G_Data/rand_spots_ds'
	new_dir = '/Users/nick_1/Bell_5G_Data/sm_rand_spots'

	# ----- ----- ----- ----- #
	print('Resizing dataset...')
	start_time = datetime.now()

	for scenario_num in tqdm(range(1, num_scs + 1), desc='Resizing data for scenarios'):
		old_img_dir = os.path.join(old_dir, str(part.value), 'scenario_{}'.format(scenario_num), 'images')
		old_tar_dir = os.path.join(old_dir, str(part.value), 'scenario_{}'.format(scenario_num), 'targets')

		new_img_dir = os.path.join(new_dir, str(part.value), 'scenario_{}'.format(scenario_num), 'images')
		new_tar_dir = os.path.join(new_dir, str(part.value), 'scenario_{}'.format(scenario_num), 'targets')

		os.makedirs(new_img_dir, exist_ok=True)
		os.makedirs(new_tar_dir, exist_ok=True)

		for file in os.listdir(old_img_dir):
			img = cv2.imread(os.path.join(old_img_dir, file), cv2.IMREAD_COLOR)
			img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
			cv2.imwrite(os.path.join(new_img_dir, file), img.astype(np.uint8))

		for file in os.listdir(old_tar_dir):
			tar = cv2.imread(os.path.join(old_tar_dir, file), cv2.IMREAD_GRAYSCALE)
			tar = cv2.resize(tar, (512, 512), interpolation=cv2.INTER_AREA)
			cv2.imwrite(os.path.join(new_tar_dir, file), tar.astype(np.uint8))

	old_list_path = os.path.join(old_dir, str(part.value), 'list.txt')
	new_list_path = os.path.join(new_dir, str(part.value), 'list.txt')
	shutil.copyfile(old_list_path, new_list_path)

	# ----- ----- ----- ----- #
	total_time = datetime.now() - start_time
	print('Script took {} to complete.'.format(total_time))
