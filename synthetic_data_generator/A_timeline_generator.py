import csv
from random import uniform

from tqdm import tqdm

from synth_data_maker.utils.helper import *


def gen_time_series_sc1_sc2(
		data_dir_path,
		num_days=28,  # days-worth of data
		max_daily_growth=6.0,  # the most amount of daily dust cloud growth (percentage out of 100)
		min_daily_growth=1.0,  # the least amount of daily dust cloud growth (percentage out of 100)
		periodic_cleaning=False,  # controls whether scheduled maintenance occurs (for scenario 2 and 4)
		cleaning_interval=12,  # intervals between maintenance occurrences (only if periodic_cleaning=True)
):
	curr_growth = 0.0
	cleaning_flag = 0  # only used for visual inspection of csv file
	dust_vars = gen_dust_variables()
	timeline = []

	for day in tqdm(range(1, num_days + 1), desc='Generating time series'):
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

	# write complete timeline to the csv file
	timeline_file_path = os.path.join(data_dir_path, 'timeline.csv')
	open(timeline_file_path, 'w+').close()  # overwrite/ make new blank file
	with open(timeline_file_path, 'a', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(csv_headers_sc1_sc2)
		writer.writerows(timeline)

