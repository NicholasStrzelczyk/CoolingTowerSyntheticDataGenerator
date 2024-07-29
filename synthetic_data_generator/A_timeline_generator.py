from random import uniform

from tqdm import tqdm

from synthetic_data_generator.utils.helper import *


def gen_time_series_sc1_sc2(
		scenario_num,
		data_dir_path=data_base_dir,
		num_days=28,  # days-worth of data
		periodic_cleaning=False,  # controls whether scheduled maintenance occurs (for scenario 2 and 4)
		cleaning_interval=12,  # intervals between maintenance occurrences (only if periodic_cleaning=True)
):
	min_daily_growth, max_daily_growth = get_growth_bounds(num_days, 3.0)

	cleaning_flag = 0  # only used for visual inspection of csv file
	curr_growth = 0.0
	dust_vars = gen_dust_variables()

	timeline = []

	for day in range(1, num_days + 1):
		if day > 1:
			curr_growth += uniform(min_daily_growth, max_daily_growth)  # grow dust by random percentage each day
			curr_growth = round(curr_growth, 2)

			if curr_growth > 100.0:  # enforce maximum total bound
				curr_growth = 100.0

			if periodic_cleaning and day % cleaning_interval == 0:  # case where cleaning occurs
				dust_vars = gen_dust_variables()  # generate new dust cloud size and location
				curr_growth = 0.0  # reset growth
				cleaning_flag = 1
			else:
				cleaning_flag = 0

		timeline.append([
			day, cleaning_flag, curr_growth,
			dust_vars['rows'], dust_vars['cols'],
			dust_vars['loc_x'], dust_vars['loc_y'],
			dust_vars['region']
		])

	# write completed timeline to the csv file
	timeline_to_csv(timeline, scenario_num, data_dir_path)


def gen_time_series_sc3_sc4(
		scenario_num,
		data_dir_path=data_base_dir,
		num_days=28,  # days-worth of data
		periodic_cleaning=False,  # controls whether scheduled maintenance occurs (for scenario 2 and 4)
		cleaning_interval=12,  # intervals between maintenance occurrences (only if periodic_cleaning=True)
):
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

			if periodic_cleaning and day % cleaning_interval == 0:  # case where cleaning occurs
				dust_vars_list = [gen_dust_variables(), gen_dust_variables(), gen_dust_variables()]
				curr_growths = [0.0, 0.0, 0.0]  # reset growth
				cleaning_flag = 1
			else:
				cleaning_flag = 0

		timeline.append([
			day, cleaning_flag,

			curr_growths[0], dust_vars_list[0]['rows'], dust_vars_list[0]['cols'],
			dust_vars_list[0]['loc_x'], dust_vars_list[0]['loc_y'], dust_vars_list[0]['region'],

			curr_growths[1], dust_vars_list[1]['rows'], dust_vars_list[1]['cols'],
			dust_vars_list[1]['loc_x'], dust_vars_list[1]['loc_y'], dust_vars_list[1]['region'],

			curr_growths[2], dust_vars_list[2]['rows'], dust_vars_list[2]['cols'],
			dust_vars_list[2]['loc_x'], dust_vars_list[2]['loc_y'], dust_vars_list[2]['region']
		])

	# write completed timeline to the csv file
	timeline_to_csv(timeline, scenario_num, data_dir_path)


def gen_time_series_data(
		data_dir_path=data_base_dir,
		partition=Partition.TRAIN,
		num_days=28,
		cleaning_interval=12,
):
	data_dir_path = os.path.join(data_dir_path, partition.value)
	for scenario_num in tqdm(range(1, 5), desc='Generating time series files'):
		if scenario_num == 1:
			gen_time_series_sc1_sc2(scenario_num, data_dir_path, num_days, False, cleaning_interval)
		elif scenario_num == 2:
			gen_time_series_sc1_sc2(scenario_num, data_dir_path, num_days, True, cleaning_interval)
		elif scenario_num == 3:
			gen_time_series_sc3_sc4(scenario_num, data_dir_path, num_days, False, cleaning_interval)
		elif scenario_num == 4:
			gen_time_series_sc3_sc4(scenario_num, data_dir_path, num_days, True, cleaning_interval)


if __name__ == '__main__':
	# hyperparameters
	num_data_days = 28
	cleaning_period = 12
	data_dir = '/Users/nick_1/Bell_5G_Data/synth_datasets'
	part = Partition.TRAIN

	# --- --- --- #
	gen_time_series_data(data_dir, part, num_data_days, cleaning_period)
