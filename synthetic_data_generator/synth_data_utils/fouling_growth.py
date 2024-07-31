from random import randint

import cv2

from synthetic_data_generator.synth_data_utils.constants import *


def vig_strength_to_growth_percent(val):
	return round(100.0 * (1.0 - ((val - min_vignette_strength) / (max_vignette_strength - min_vignette_strength))), 2)


def growth_percent_to_vig_strength(val):
	return round(min_vignette_strength + ((1.0 - (0.01 * val)) * (max_vignette_strength - min_vignette_strength)), 2)


def get_growth_bounds(num_days, plusminus_percent=3.0):
	avg_growth_percent = round(100.0 / num_days, 1)
	max_val = avg_growth_percent + plusminus_percent
	min_val = avg_growth_percent - plusminus_percent
	return min_val, max_val


def gen_dust_variables(
		dust_icon_img_path=dust_icon_path,
		max_cols=35,
		max_rows=10,
):
	rows = randint(3, max_rows)
	cols = randint(rows, max_cols)
	region_num = randint(1, 2)  # determines whether dust will be in top or bottom region (1 or 2)

	dust_h, dust_w = cv2.imread(dust_icon_img_path).shape[:2]  # [width=20, height=36]

	# valid region 1 = [x: 120-1840, y: 0-510]
	# valid region 2 = [x: 120-1840, y: 700-1080]
	min_location_y = 0 if region_num == 1 else 700
	max_location_y = 510 - (rows * dust_h) if region_num == 1 else 1080 - (rows * dust_h)
	min_location_x = 120
	max_location_x = 1840 - (cols * dust_w)

	location_y = randint(min_location_y, max_location_y)
	location_x = randint(min_location_x, max_location_x)

	return {
		'rows': rows,
		'cols': cols,
		'loc_x': location_x,
		'loc_y': location_y,
		'region': region_num,
	}
