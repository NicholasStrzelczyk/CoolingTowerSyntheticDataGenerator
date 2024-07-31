import csv

import pandas as pd

from synthetic_data_generator.synth_data_utils.constants import *
from synthetic_data_generator.synth_data_utils.fouling_growth import growth_percent_to_vig_strength


def timeline_to_csv(timeline_data, scenario_num, data_dir):
	save_path = os.path.join(data_dir, "scenario_{}".format(scenario_num), 'timeline.csv')
	open(save_path, 'w+').close()  # overwrite/ make new blank file
	with open(save_path, 'a', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		if scenario_num <= 2:
			writer.writerow(csv_headers_sc1_sc2)
		else:
			writer.writerow(csv_headers_sc3_sc4)
		writer.writerows(timeline_data)


def interpret_sc1_sc2_csv(timeline_file_path):
	df = pd.read_csv(timeline_file_path)
	total_days = df['day'].values[-1]
	growths = df['growth_percent'].values.tolist()
	dust_rows = df['dust_rows'].values.tolist()
	dust_cols = df['dust_cols'].values.tolist()
	dust_x_vals = df['dust_x'].values.tolist()
	dust_y_vals = df['dust_y'].values.tolist()

	vignettes = []
	for g in growths:
		vignettes.append(growth_percent_to_vig_strength(g))

	return {
		'total_days': total_days,
		'vignettes': vignettes,
		'dust_rows': dust_rows,
		'dust_cols': dust_cols,
		'dust_x_vals': dust_x_vals,
		'dust_y_vals': dust_y_vals,
	}


def interpret_sc3_sc4_csv(timeline_file_path):
	csv_dicts = []
	df = pd.read_csv(timeline_file_path)
	total_days = df['day'].values[-1]

	for i in range(1, 4):
		growths = df['growth_percent_{}'.format(i)].values.tolist()
		dust_rows = df['dust_rows_{}'.format(i)].values.tolist()
		dust_cols = df['dust_cols_{}'.format(i)].values.tolist()
		dust_x_vals = df['dust_x_{}'.format(i)].values.tolist()
		dust_y_vals = df['dust_y_{}'.format(i)].values.tolist()

		vignettes = []
		for g in growths:
			vignettes.append(growth_percent_to_vig_strength(g))

		csv_dicts.append({
			'total_days': total_days,
			'vignettes': vignettes,
			'dust_rows': dust_rows,
			'dust_cols': dust_cols,
			'dust_x_vals': dust_x_vals,
			'dust_y_vals': dust_y_vals,
		})
	return csv_dicts
