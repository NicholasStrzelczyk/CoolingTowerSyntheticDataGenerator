import csv
import os
from datetime import datetime
from random import uniform

import cv2
import pandas as pd
from tqdm import tqdm

from synthetic_data_generator.B_synthetic_data_generator import make_data_sc3_sc4
from synthetic_data_generator.utils.constants import Partition, hour_list, csv_headers_sc3_sc4
from synthetic_data_generator.utils.helper import get_scenario_paths, get_growth_bounds, gen_dust_variables


def make_folders(root_dir, partition, num_scenarios):
	data_part_path = os.path.join(root_dir, str(partition.value))
	for scenario_num in range(1, num_scenarios + 1):
		sc_path = os.path.join(data_part_path, 'scenario_{}'.format(scenario_num))
		os.makedirs(os.path.join(sc_path, 'images'), exist_ok=True)
		os.makedirs(os.path.join(sc_path, 'targets'), exist_ok=True)


def make_dataset_timelines(root_dir, partition, num_days, num_scenarios):
	data_part_path = os.path.join(root_dir, str(partition.value))
	for scenario_num in tqdm(range(1, num_scenarios + 1), desc='Generating timelines for scenarios'):
		min_daily_growth, max_daily_growth = get_growth_bounds(num_days, 3.0)
		cleaning_flag = 0  # only used for visual inspection of csv file
		curr_growths = [0.0, 0.0, 0.0]
		dust_vars_list = [gen_dust_variables(), gen_dust_variables(), gen_dust_variables()]
		timeline = []
		for day in range(1, num_days + 1):
			if day > 1:
				for i in range(len(curr_growths)):
					curr_growths[i] += uniform(min_daily_growth, max_daily_growth)
					curr_growths[i] = round(curr_growths[i], 2)
					if curr_growths[i] > 100.0:  # enforce maximum total bound
						curr_growths[i] = 100.0
			timeline.append([
				day, cleaning_flag,
				curr_growths[0], dust_vars_list[0]['rows'], dust_vars_list[0]['cols'],
				dust_vars_list[0]['loc_x'], dust_vars_list[0]['loc_y'], dust_vars_list[0]['region'],
				curr_growths[1], dust_vars_list[1]['rows'], dust_vars_list[1]['cols'],
				dust_vars_list[1]['loc_x'], dust_vars_list[1]['loc_y'], dust_vars_list[1]['region'],
				curr_growths[2], dust_vars_list[2]['rows'], dust_vars_list[2]['cols'],
				dust_vars_list[2]['loc_x'], dust_vars_list[2]['loc_y'], dust_vars_list[2]['region']
			])
		save_path = os.path.join(data_part_path, "scenario_{}".format(scenario_num), 'timeline.csv')
		open(save_path, 'w+').close()  # overwrite/ make new blank file
		with open(save_path, 'a', encoding='UTF8', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(csv_headers_sc3_sc4)
			writer.writerows(timeline)


def make_dataset_images(root_dir, partition, dust_im_path, num_scenarios):
	for scenario_num in range(1, num_scenarios + 1):
		paths = get_scenario_paths(root_dir, partition, scenario_num)
		dust_img = cv2.imread(dust_im_path, cv2.IMREAD_COLOR)
		make_data_sc3_sc4(paths, dust_img, scenario_num)


def make_dataset_list(root_dir, partition, num_scenarios):
	part_path = os.path.join(root_dir, str(partition.value))
	list_name = 'list.txt'
	list_file_path = os.path.join(part_path, list_name)
	open(list_file_path, 'w+').close()  # overwrite/ make new blank file
	list_file = open(list_file_path, "a")
	for scenario in tqdm(range(1, num_scenarios + 1), desc='Creating dataset list'):
		scenario_dir = os.path.join(part_path, "scenario_{}".format(scenario))
		total_days = pd.read_csv(os.path.join(scenario_dir, "timeline.csv"))['day'].values[-1]
		for day in range(1, total_days + 1):
			label_name = "LABEL_day_{}.png".format(day)
			for hour in hour_list:
				image_name = "SYNTH_day_{}_{}.png".format(day, hour)
				if os.path.exists(os.path.join(scenario_dir, "images", image_name)):  # skip over missing hours
					img_path = os.path.join("scenario_{}".format(scenario), "images", image_name)
					tgt_path = os.path.join("scenario_{}".format(scenario), "targets", label_name)
					list_file.write(img_path + " " + tgt_path + "\n")
	list_file.close()


if __name__ == '__main__':
	# hyperparameters
	num_scs = 10
	num_data_days = 15
	part = Partition.TRAIN
	dust_icon_path = './image_files/dust1.png'
	data_dir = '/Users/nick_1/Bell_5G_Data/rand_spots_ds'

	# ----- ----- ----- ----- #
	print('Creating synthetic {} dataset...'.format(part.value))
	start_time = datetime.now()

	make_folders(data_dir, part, num_scs)  # make directories for dataset
	make_dataset_timelines(data_dir, part, num_data_days, num_scs)  # make timeline files for each scenario
	make_dataset_images(data_dir, part, dust_icon_path, num_scs)  # make images and targets for each scenario
	make_dataset_list(data_dir, part, num_scs)  # make list file for dataset

	# ----- ----- ----- ----- #
	total_time = datetime.now() - start_time
	print('Script took {} to complete.'.format(total_time))


