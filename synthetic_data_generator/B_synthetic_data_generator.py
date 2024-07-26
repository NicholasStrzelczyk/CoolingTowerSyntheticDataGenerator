from tqdm import tqdm

from synth_data_maker.utils.helper import *


def gen_synth_data(
		base_dir=data_base_dir,
		partition=Partition.TRAIN,
		dust_img_path=dust_icon_path,
):
	src_dir = os.path.join(base_dir, 'src_images')  # where raw images are stored
	dust_img = cv2.imread(dust_img_path, cv2.IMREAD_COLOR)  # dust sample image

	for scenario in tqdm(range(1, 5), desc='Generating synthetic data scenarios'):
		# get scenario-dependant paths
		dest_dir = os.path.join(base_dir, '{}/scenario_{}'.format(partition.value, scenario))
		csv_file_path = os.path.join(dest_dir, 'timeline.csv')  # timeline csv file location
		img_dir_path = os.path.join(dest_dir, 'images')  # location to save synth images
		label_dir_path = os.path.join(dest_dir, 'targets')  # location to save synth labels

		# check if timeline csv file exists
		if not os.path.exists(csv_file_path):
			print('[ERROR] failed to find csv file for scenario {}'.format(scenario))
			continue

		# get scenario csv data
		if scenario <= 2:
			csv_data = interpret_sc1_sc2_csv(csv_file_path)
		else:
			csv_data = interpret_sc3_sc4_csv(csv_file_path)

		for day in range(1, csv_data['total_days'] + 1):
			dust_cloud = make_dust_cloud(dust_img, csv_data, day - 1)
			pt1, pt2 = get_img_slice_pts(dust_cloud, csv_data, day - 1)

			# create synth label for day
			gen_synth_label(day, label_dir_path, dust_cloud, pt1, pt2)

			# generate synth image for each valid hour of current day
			gen_hourly_synth_images(day, src_dir, img_dir_path, dust_cloud, pt1, pt2)


if __name__ == '__main__':
	# hyperparameters
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
	part = Partition.TRAIN
	dust_icon_path = './image_files/dust1.png'

	# --- --- --- #
	gen_synth_data()
