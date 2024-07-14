'''
Adapted from https://github.com/naver/oasis/blob/master/image_helpers.py
'''

import os

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class ImageOps:
	def __init__(self):

		self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

		self.LABELS_DICT = {0:"road",
							1:"sidewalk",
							2:"building",
							3:"wall",
							4:"fence",
							5:"pole",
							6:"light",
							7:"sign",
							8:"vegetation",
							9:"terrain",
							10:"sky",
							11:"person",
							12:"rider",
							13:"car",
							14:"truck",
							15:"bus",
							16:"train",
							17:"motocycle",
							18:"bicycle"
							}

		self.PALETTE = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
						220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
						0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32] # 19x3 values, for Image's palette() module

		zero_pad = 256 * 3 - len(self.PALETTE)
		for i in range(zero_pad):
			self.PALETTE.append(0)

	def colorize_mask(self, mask):
		# mask: numpy array of the mask
		new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
		new_mask.putpalette(self.PALETTE)
		return new_mask


	def get_concat_h(self, im1, im2):
		dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
		dst.paste(im1, (0, 0))
		dst.paste(im2, (im1.width, 0))
		return dst


	def get_concat_v(self, im1, im2):
		dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
		dst.paste(im1, (0, 0))
		dst.paste(im2, (0, im1.height))
		return dst


	def process_image_for_saving(self, image, interp=None):

		# handling RGB input image
# 		image = interp(image).cpu().numpy().squeeze()
# 		image = np.transpose(image, (1, 2, 0))
# 		image += self.IMG_MEAN
# 		image = image[:, :, ::-1]
		image = image.astype(np.uint8)
		image = Image.fromarray(image)
		return image


	def process_rescaled_image_for_saving(self, image, mean, std):

		# handling RGB input image
		image = image.cpu().numpy().squeeze()
		image = np.transpose(image, (1, 2, 0))
		image *= std
		image += mean
		image *= 255.
		#image = image[:, :, ::-1]
		image = image.astype(np.uint8)
		return image


	def save_concat_image(self, image, gt, pred, unc_map, save_path, image_name):

		"""
		Save concatenation of image, ground truth and prediction
		"""

		image_concat = self.get_concat_v(image, pred)
		image_concat = self.get_concat_v(gt, image_concat)
		unc_map = np.stack(
				(unc_map, np.zeros_like(unc_map), np.zeros_like(unc_map)),
				axis=2)
		unc_map_on_img = (0.4*np.array(image) + 0.6*unc_map).astype(np.uint8)
		unc_map_on_img = Image.fromarray(unc_map_on_img)
		image_concat = self.get_concat_v(image_concat, unc_map_on_img)
		image_concat_path = os.path.join(save_path, image_name)
		image_concat.save(image_concat_path)
