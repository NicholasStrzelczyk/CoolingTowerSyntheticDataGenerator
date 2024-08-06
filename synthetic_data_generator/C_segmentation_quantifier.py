from synthetic_data_generator.utils.helper import *

col_headers = ['day', 'percent_fouling']


def quantify_scenario_fouling(data_dir_path, partition):
	dataset_dir = os.path.join(data_dir_path, partition.value)

	for sc in range(1, 5):
		# create paths
		scenario_path = os.path.join(dataset_dir, 'scenario_{}'.format(sc))
		targets_path = os.path.join(scenario_path, 'targets')
		timeline_path = os.path.join(scenario_path, 'timeline.csv')

		# get total number of days
		num_days = pd.read_csv(timeline_path)['day'].values[-1]

		# gather target fouling percentages
		fouling_data = []
		for day in range(1, num_days + 1):
			tgt_name = 'LABEL_day_{}.png'.format(day)
			tgt_image = cv2.imread(os.path.join(targets_path, tgt_name), cv2.IMREAD_GRAYSCALE)
			dust_count = np.count_nonzero(tgt_image > 0)
			percentage = round((100 * (dust_count / (1920 * 1080))), 4)
			fouling_data.append([day, percentage])

		# create new csv
		save_path = os.path.join(scenario_path, 'fouling_percentages.csv')
		open(save_path, 'w+').close()  # overwrite/ make new blank file
		with open(save_path, 'a', encoding='UTF8', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(col_headers)
			writer.writerows(fouling_data)


if __name__ == '__main__':
	# hyperparameters
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
	part = Partition.VAL
	# ----- ----- ----- #
	quantify_scenario_fouling(data_dir, part)
