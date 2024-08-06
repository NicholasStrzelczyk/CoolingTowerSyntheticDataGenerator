from tqdm import tqdm

from synthetic_data_generator.utils.helper import *


def make_data_sc1_sc2(paths, dust_img, scenario):
	csv_data = interpret_sc1_sc2_csv(paths['csv'])

	for day in tqdm(range(1, csv_data['total_days'] + 1), desc='Generating images for scenario {}/4'.format(scenario)):
		dust_cloud = make_dust_cloud(dust_img, csv_data, day - 1)
		pt1, pt2 = get_img_slice_pts(dust_cloud, csv_data, day - 1)

		# Part A: create synth label for day
		label_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # blank image
		label_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = dust_cloud  # insert dust cloud
		label_img = denoise_to_binary(label_img)  # denoise & convert to b/w
		label_save_path = os.path.join(paths['label'], 'LABEL_day_{}.png'.format(day))
		cv2.imwrite(label_save_path, label_img.astype(np.uint8))  # save synthetic label

		# Part B: generate synth image for each valid hour of current day
		for hour in hour_list:
			raw_img_name = 'day_{}_{}.png'.format(day, hour)
			raw_img_path = os.path.join(paths['src'], raw_img_name)
			synth_img_save_path = os.path.join(paths['img'], 'SYNTH_{}'.format(raw_img_name))

			if os.path.isfile(raw_img_path):  # skip over missing data
				image = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)  # read raw img file
				image_slice = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # cut out RoI for modification
				image_slice = apply_synthetic_dust_basic(image_slice, dust_cloud)  # apply dust cloud to RoI
				image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = image_slice  # re-insert edited RoI
				cv2.imwrite(synth_img_save_path, image.astype(np.uint8))  # save synthetic img


def make_data_sc3_sc4(paths, dust_img, scenario):
	csv_dicts = interpret_sc3_sc4_csv(paths['csv'])

	for day in tqdm(range(1, csv_dicts[0]['total_days'] + 1), desc='Generating images for scenario {}/4'.format(scenario)):

		# Part A: create list of dust cloud data for current day
		dust_clouds = []

		for deg_spot_data in csv_dicts:
			dust_cloud = make_dust_cloud(dust_img, deg_spot_data, day - 1)
			pt1, pt2 = get_img_slice_pts(dust_cloud, deg_spot_data, day - 1)
			dust_clouds.append((dust_cloud, pt1, pt2))

		# Part B: create synth label for current day
		label_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # blank image

		for dust_cloud, pt1, pt2 in dust_clouds:  # apply each dust cloud
			label_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = dust_cloud  # insert dust cloud to label

		label_img = denoise_to_binary(label_img)  # denoise & convert to b/w
		label_save_path = os.path.join(paths['label'], 'LABEL_day_{}.png'.format(day))
		cv2.imwrite(label_save_path, label_img.astype(np.uint8))  # save synthetic label

		# Part C: generate synth image for each valid hour of current day
		for hour in hour_list:
			raw_img_name = 'day_{}_{}.png'.format(day, hour)
			raw_img_path = os.path.join(paths['src'], raw_img_name)
			synth_img_save_path = os.path.join(paths['img'], 'SYNTH_{}'.format(raw_img_name))

			if os.path.isfile(raw_img_path):  # skip over missing data
				image = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)  # read raw img file

				for dust_cloud, pt1, pt2 in dust_clouds:  # apply each dust cloud
					image_slice = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # cut out RoI for modification
					image_slice = apply_synthetic_dust_basic(image_slice, dust_cloud)  # apply dust cloud to RoI
					image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = image_slice  # re-insert edited RoI

				cv2.imwrite(synth_img_save_path, image.astype(np.uint8))  # save synthetic img


def gen_synth_data(
		base_dir=data_base_dir,
		partition=Partition.TRAIN,
		dust_img_path=dust_icon_path,
):
	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)  # dust sample image

	for scenario in range(1, 5):
		paths = get_scenario_paths(base_dir, partition, scenario)  # dict of scenario-dependant paths

		# get scenario csv data
		if scenario <= 2:
			make_data_sc1_sc2(paths, dust_img, scenario)
		else:
			make_data_sc3_sc4(paths, dust_img, scenario)


if __name__ == '__main__':
	# hyperparameters
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
	part = Partition.VAL
	dust_icon_path = './image_files/dust1.png'

	# --- --- --- #
	gen_synth_data(data_dir, part, dust_icon_path)
