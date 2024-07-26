from random import randint

import cv2
import numpy as np
import pandas as pd

from synth_data_maker.utils.constants import *


# ----- Utility Methods ----- #

def vig_strength_to_growth_percent(val):
	return round(100.0 * (1.0 - ((val - min_vignette_strength) / (max_vignette_strength - min_vignette_strength))), 2)


def growth_percent_to_vig_strength(val):
	return round(min_vignette_strength + (1.0 - (0.01 * val)) * (max_vignette_strength - min_vignette_strength), 2)


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


def interpret_sc3_sc4_csv(timeline_file_path):  # NOT DONE YET!
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


def make_dust_region(dust_img, rows, cols):
	h, w, c = dust_img.shape
	result = np.zeros((h * rows, w * cols, c), dust_img.dtype)
	for row in range(rows):
		for col in range(cols):
			result[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = dust_img[:, :, :]
	return result


def apply_vignette(img, strength):
	result = np.copy(img)
	h, w = result.shape[:2]
	x_resultant_kernel = cv2.getGaussianKernel(w, w / strength)
	y_resultant_kernel = cv2.getGaussianKernel(h, h / strength)
	resultant_kernel = y_resultant_kernel * x_resultant_kernel.T
	mask = resultant_kernel / resultant_kernel.max()
	for ch in range(3):
		result[:, :, ch] = result[:, :, ch] * mask
	return result


def make_dust_cloud(dust_img, csv_data, idx):
	dust_cloud = make_dust_region(dust_img, csv_data['dust_rows'][idx], csv_data['dust_cols'][idx])
	dust_cloud = apply_vignette(dust_cloud, csv_data['vignettes'][idx])
	return dust_cloud


def get_img_slice_pts(dust_cloud_img, csv_data, idx):
	pt1_x = csv_data['dust_x_vals'][idx]
	pt1_y = csv_data['dust_y_vals'][idx]
	pt2_x = pt1_x + dust_cloud_img.shape[1]
	pt2_y = pt1_y + dust_cloud_img.shape[0]
	pt1 = (pt1_x, pt1_y)
	pt2 = (pt2_x, pt2_y)
	return pt1, pt2


def apply_synthetic_dust_basic(raw_img, deg_img):
	gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for raw img
	result = np.zeros(raw_img.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels
				if gray[y, x] < thresh:  # if this pixel belongs to the background of raw img (not metal)
					result[y, x, c] = deg_img[y, x, c]  # color pixel same as deg_img
				else:  # else this pixel belongs to the foreground of raw img (the metal)
					result[y, x, c] = raw_img[y, x, c]  # color pixel same as raw img
	return result


# not used anymore
def apply_synthetic_dust(img1, img2, alpha=0.99):
	gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img1
	result = np.zeros(img1.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels

				if gray_img1[y, x] < thresh_img1:  # if this pixel belongs to the background of img1 (not metal grate)
					result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
				else:  # else this pixel belongs to the foreground of img1 (the metal grate)
					result[y, x, c] = img1[y, x, c]  # color pixel same as img1

	return result


# not used anymore
def apply_synthetic_dust_strict(img1, img2, alpha=0.99):
	gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img1
	thresh_img2 = cv2.threshold(gray_img2, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for img2
	result = np.zeros(img1.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels

				if gray_img1[y, x] < thresh_img1:  # if this pixel belongs to the background of img1 (not metal grate)
					if gray_img2[y, x] >= thresh_img2:  # blend pixels according to alpha
						result[y, x, c] = ((1 - alpha) * img1[y, x, c]) + (alpha * img2[y, x, c])
					else:  # makes pixel blend more fair toward edges of vignette
						result[y, x, c] = (0.5 * img1[y, x, c]) + (0.5 * img2[y, x, c])
				else:  # else this pixel belongs to the foreground of img1 (the metal grate)
					result[y, x, c] = img1[y, x, c]  # color pixel same as img1

	return result


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
	return result


def gen_synth_label(day, save_dir_path, dust_cloud, pt1, pt2):
	label_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # blank image
	label_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = dust_cloud  # insert dust cloud
	label_img = denoise_to_binary(label_img)  # denoise & convert to b/w
	save_path = os.path.join(save_dir_path, 'LABEL_day_{}.png'.format(day))
	cv2.imwrite(save_path, label_img.astype(np.uint8))  # save synthetic label


def gen_synth_image(src_path, save_path, dust_cloud, pt1, pt2):
	image = cv2.imread(src_path, cv2.IMREAD_COLOR)  # read raw img file
	image_slice = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # cut out RoI for modification
	image_slice = apply_synthetic_dust_basic(image_slice, dust_cloud)  # apply dust cloud to RoI
	image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = image_slice  # re-insert edited RoI
	cv2.imwrite(save_path, image.astype(np.uint8))  # save synthetic img


def gen_hourly_synth_images(day, src_dir_path, save_dir_path, dust_cloud, pt1, pt2):
	for hour in hour_list:
		raw_img_name = 'day_{}_{}.png'.format(day, hour)
		raw_img_path = os.path.join(src_dir_path, raw_img_name)
		synth_img_path = os.path.join(save_dir_path, 'SYNTH_{}'.format(raw_img_name))

		if os.path.isfile(raw_img_path):  # skip over missing data
			gen_synth_image(raw_img_path, synth_img_path, dust_cloud, pt1, pt2)



