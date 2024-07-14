'''
MIT License

Copyright (c) 2019 Riccardo Volpi

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import numpy as np
import numpy.random as npr

try:
	import cPickle
except:
	pass

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageFilter
import PIL.Image
import matplotlib
import noise_utils

class TransfOps(object):
	'''
	Class to handle data transformations.
	'''
	def __init__(self, transformation_list):

		'''
		Init object to deal with image transformations')
		'''
		self.transformation_list = transformation_list
		#self.transformation_list = ['identity', 'rotate', 'brightness', 'color', 'contrast', 'RGB_rand', 'solarize']
		self.define_code_correspondances()


	def decode_string(self, transf_string):
		'''
		Code to decode the string used by the genetic algorithm
		String example: 't1,l1_3,t4,l4_0,t0,l0_1'. First transformation is the one
		associated with index '1', with level set to '3', and so on.
		'random_N' with N integer gives N rnd transformations with rnd levels.
		'''
		if 'random' in transf_string:
			transformations = npr.choice(self.transformation_list, int(transf_string.split('_')[-1])) # the string is 'random_N'
			levels = [npr.choice(list(self.code_to_level_dict[t].values()), 1)[0] for t in transformations] # list() to make it compatible with Python3
		else:
			transformation_codes = transf_string.split(',')[0::2] 
			level_codes = transf_string.split(',')[1::2]
			
			transformations = [self.code_to_transf(code) for code in transformation_codes] 	
			levels = [self.code_to_level(transf,level) for transf,level in zip(transformations, level_codes)] 	

		return transformations, levels		


	def transform_dataset(self, dataset, transf_string = 't0,l0_0', transformations=None, levels=None):
		'''
		dataset: set of images, shape should be N x width x height x #channels
		transf_string: transformations and levels encoded in a string 
		'''

		dataset = np.copy(dataset)

		#print('Dataset size:{}'.format(dataset.shape))
		if len(dataset.shape) == 3: # if 'dataset' is a single image
			dataset = np.expand_dims(dataset, 0) 
		if dataset.shape[-1] != 3:
			print('Input shape:', str(dataset.shape))
			raise Exception('The images must be in RGB format')

		tr_dataset = np.zeros((dataset.shape))
		if transformations is None:
			# decoding transformation string
			transformations, levels = self.decode_string(transf_string)
		for n,img in enumerate(dataset):
			pil_img = PIL.Image.fromarray(img.astype('uint8'), 'RGB')

			for transf,level in zip(transformations, levels):
				pil_img = self.apply_transformation(pil_img, transf, level)

			tr_dataset[n] = np.array(pil_img)

		return tr_dataset, transformations, levels


	def apply_transformation(self, image, transformation, level):
		'''
		image: image to be tranformed, shape should be 1 x width x height x #channels
		transformation: type of transformation to be applied
		level: level of the perturbation to be applied 
		'''
		if transformation == 'identity':
			return image

		elif transformation == 'brightness':
			return PIL.ImageEnhance.Brightness(image).enhance(level)

		elif transformation == 'invert':
			image = np.array(image).astype(int)
			image = np.abs(255-image)
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image

		elif transformation == 'black_and_white':
			image = np.array(image).astype(int)
			gray_image = 0.3*image[:,:,0]+ 0.50*image[:,:,1] + 0.11*image[:,:,2]
			image = np.stack((gray_image, gray_image, gray_image), axis=2)
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image
	
		elif transformation == 'color':
			return PIL.ImageEnhance.Color(image).enhance(level)

		elif transformation == 'contrast':
			return PIL.ImageEnhance.Contrast(image).enhance(level)

		elif transformation == 'rotate_upside_down':
			image = image.rotate(level, resample=PIL.Image.BILINEAR)
			return image

		elif transformation == 'rotate':
			image = image.rotate(level, resample=PIL.Image.BILINEAR)
			return image

		elif transformation == 'rotate_90':
			image = image.rotate(90, resample=PIL.Image.BILINEAR)
			return image

		elif transformation == 'rotate_180':
			image = image.rotate(180, resample=PIL.Image.BILINEAR)
			return image

		elif transformation == 'rotate_270':
			image = image.rotate(270, resample=PIL.Image.BILINEAR)
			return image

		elif transformation == 'solarize':
			return PIL.ImageOps.solarize(image, threshold=level)

		elif transformation == 'RGB_rand':
			image = np.array(image).astype(int)
			image[:,:,0] += npr.randint(low=-level,high=level)
			image[:,:,1] += npr.randint(low=-level,high=level)
			image[:,:,2] += npr.randint(low=-level,high=level)
			image[image>255] = 255
			image[image<0] = 0
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image

		elif transformation == 'gaussian_noise':
			image = np.array(image).astype(int)
			image = noise_utils.noisy('gauss', np.expand_dims(image,0), level)
			image = PIL.Image.fromarray(np.squeeze(image).astype('uint8'), 'RGB')
			return image

		elif transformation == 'blur':
			image = image.filter(PIL.ImageFilter.BLUR)
			return image

		#elif transformation == 'hue':
		#	image = np.array(image).astype(int)
		#	print 'Before: ', image.min(), image.max()
		#	image = matplotlib.colors.rgb_to_hsv(image/255.)
		#	image[:,:,0] += level
		#	image = matplotlib.colors.hsv_to_rgb(image)
		#	print 'After: ', image.min(), image.max()
		#	return(image * 255.)
		#elif transformation == 'saturation':
		#	image = np.array(image).astype(int)
		#	print 'Before: ', image.min(), image.max()
		#	image = matplotlib.colors.rgb_to_hsv(image/255.)
		#	image[:,:,1] += level
		#	image = matplotlib.colors.hsv_to_rgb(image)
		#	print 'After: ', image.min(), image.max()
		#	return(image * 255.)

		else:
			raise Exception('Unknown transformation!')


	def code_to_transf(self, code):
		'''
		Takes in input a code (e.g., 't0', 't1', ...) and gives in output 
		the related transformation.
		'''
		return self.code_to_transf_dict[code]


	def code_to_level(self, transformation, code):
		'''
		Takes in input a transfotmation (e.g., 'invert', 'colorize', ...) and 
		a level code (e.g., 'l0_1', 'l1_3', ...) and gives in output the related level.
		'''
		return self.code_to_level_dict[transformation][code]


	def define_code_correspondances(self):
		'''
		Define the correpondances between transformation/level codes
		and the actual types and values.
		'''
		self.code_to_transf_dict = dict()
		self.code_to_transf_dict['t0'] = 'identity'

		# color
		self.code_to_transf_dict['t1'] = 'brightness'
		self.code_to_transf_dict['t2'] = 'color'
		self.code_to_transf_dict['t3'] = 'contrast'
		self.code_to_transf_dict['t4'] = 'solarize'
		self.code_to_transf_dict['t5'] = 'RGB_rand'
		self.code_to_transf_dict['t6'] = 'black_and_white'
		self.code_to_transf_dict['t7'] = 'invert'

		# geometric
		self.code_to_transf_dict['t8'] = 'rotate'
		self.code_to_transf_dict['t9'] = 'rotate_90'
		self.code_to_transf_dict['t10'] = 'rotate_180'
		self.code_to_transf_dict['t11'] = 'rotate_270'
		self.code_to_transf_dict['t12'] = 'rotate_upside_down'

		# noise
		self.code_to_transf_dict['t13'] = 'gaussian_noise'
		self.code_to_transf_dict['t14'] = 'blur'

		self.code_to_level_dict = dict()
		for k in self.transformation_list:
			self.code_to_level_dict[k] = dict()

		RGB_factor_range = np.linspace(1,120,90)
		factor_range = np.linspace(0.2,1.8,90)
		degree_range = np.linspace(-60,60,90).astype(int)
		solarize_factor_range = np.linspace(255,75,90)
		noise_variance_range = np.linspace(0.0,30.,20)

		# no levels 
		self.code_to_level_dict['identity'] = dict()
		self.code_to_level_dict['identity'][str(0)] = None

		self.code_to_level_dict['invert'] = dict()
		self.code_to_level_dict['invert'][str(0)] = None

		self.code_to_level_dict['rotate_90'] = dict()
		self.code_to_level_dict['rotate_90'][str(0)] = None

		self.code_to_level_dict['rotate_180'] = dict()
		self.code_to_level_dict['rotate_180'][str(0)] = None

		self.code_to_level_dict['rotate_270'] = dict()
		self.code_to_level_dict['rotate_270'][str(0)] = None

		self.code_to_level_dict['black_and_white'] = dict()
		self.code_to_level_dict['black_and_white'][str(0)] = None

		self.code_to_level_dict['blur'] = dict()
		self.code_to_level_dict['blur'][str(0)] = None

		# factors
		self.code_to_level_dict['brightness'] = dict()
		for n,l in enumerate(factor_range):
			self.code_to_level_dict['brightness'][str(n)] = l

		self.code_to_level_dict['contrast'] = dict()
		for n,l in enumerate(factor_range):
			self.code_to_level_dict['contrast'][str(n)] = l

		self.code_to_level_dict['color'] = dict()
		for n,l in enumerate(factor_range):
			self.code_to_level_dict['color'][str(n)] = l

		# degrees clockwise
		self.code_to_level_dict['rotate'] = dict()
		for n,l in enumerate(degree_range):
			self.code_to_level_dict['rotate'][str(n)] = l

		self.code_to_level_dict['rotate_upside_down'] = dict()
		for n,l in enumerate(degree_range):
			self.code_to_level_dict['rotate_upside_down'][str(n)] = l + 180

		self.code_to_level_dict['solarize'] = dict()
		for n,l in enumerate(solarize_factor_range):
			self.code_to_level_dict['solarize'][str(n)] = l

		# rgb factors
		self.code_to_level_dict['RGB_rand'] = dict()
		for n,l in enumerate(RGB_factor_range.astype(int)):
			self.code_to_level_dict['RGB_rand'][str(n)] = l

		# variance
		self.code_to_level_dict['gaussian_noise'] = dict()
		for n,l in enumerate(noise_variance_range):
			self.code_to_level_dict['gaussian_noise'][str(n)] = l


if __name__=='__main__':

	data_dir = '../data'

	# loading 32x32 MNIST
	image_dir = os.path.join(data_dir, 'mnist', 'train.pkl')
	with open(image_dir, 'rb') as f:
	    mnist = cPickle.load(f)

	print('Loaded MNIST')

	images = mnist['X']
	labels = mnist['y']

	import scipy.io
	from load_ops import LoadOps
	load_ops = LoadOps(data_dir)
	
	images, _ = load_ops.load_mnist()

	images = images[:10] * 255.

	import matplotlib.pyplot as plt
	#plt.imshow(images[0])
	#plt.show()

	# grayscale to rgb
	# images = np.squeeze(np.stack((images,images,images), axis=3))
	
	# object to handle dataset transformations
	tr_ops = TransfOps()	
	print('Created TransfOps object')
	
	# defining a transformation string
	#tr_string = 't0,l0_0' #identity function
	#tr_string = 't0,l0_0,t1,l1_2'

	trs=['RGB_rand']
	lvls=[50]

	print(images.min(), images.max())  

	# transforming MNIST dataset
	print('Applying transformations...')
	tr_images, transformations, levels = tr_ops.transform_dataset(images, transformations=trs, levels=lvls)
	print('Done!')

	print('Saving images')
	PIL.Image.fromarray(images[0].astype('uint8')).save('img_original.png')
	PIL.Image.fromarray(tr_images[0].astype('uint8')).save('img_modified.png')

#	print('Saving transformations')
#	ofile = open('used_transformations.txt','w')
#	ofile.write('_'.join(transformations))
#	ofile.write('\n')
#	ofile.write('_'.join([str(l) for l in levels]))
#	ofile.close()









