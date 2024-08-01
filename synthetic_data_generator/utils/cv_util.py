import cv2
import numpy as np


def gen_fouling_pattern(dust_img, rows, cols):
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


def make_fouling_spot(dust_img, csv_data, idx):
	dust_cloud = gen_fouling_pattern(dust_img, csv_data['dust_rows'][idx], csv_data['dust_cols'][idx])
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


def apply_synthetic_fouling(raw_img, deg_img):
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


def denoise_to_binary(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
	result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)[1]
	return result