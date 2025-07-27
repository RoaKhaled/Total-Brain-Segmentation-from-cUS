"""
Author: Roa'a Khaled
Affiliation: Department of Computer Engineering, Department of Condensed Matter Physics,
             University of Cádiz, , Puerto Real, 11519, Cádiz, Spain.
Email: roaa.khaled@gm.uca.es
Date: 2025-07-27
Project: This work is part of a predoctoral research and part of PARENT project that has received funding
         from the EU’s Horizon 2020 research and innovation program under the MSCA – ITN 2020, GA No 956394.

Description:
    This code defines a dataset class for 2d image segmenation task, using paired NIfTI (.nii or .nii.gz) images and
    masks. It loads image-mask pairs from disk, applies optional transformations, and returns them as tensors. In our
    case we segmented Total Brains from cranial Ultrasound images of premature born infants.

Usage:
    from UNet2D_pyImageSearch.dataset import SegmentationDataset_2D  --> for 2D model training
    from UNet2D_pyImageSearch.dataset import SegmentationDataset_Contextual --> for Contextual model training

Notes:
    - for environment requirements check
    - this code is adapted from (https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
"""

# import the necessary packages
from torch.utils.data import Dataset
import os
#import cv2
#from PIL import Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

#for training model with 1 slice input only
class SegmentationDataset_2D(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB if needed,
		# and read the associated mask from disk in grayscale mode
		#image = cv2.imread(imagePath)
		image = nib.load(imagePath).get_fdata().astype(np.float32) #/ 255.0 ## added intensity normalization
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#mask = cv2.imread(self.maskPaths[idx], 0)
		mask = nib.load(self.maskPaths[idx]).get_fdata().astype(np.float32)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)

##################################################################################################################
#for Contextual model training (i.e. using 3 consecutive or whatever number of input slices)
class SegmentationDataset_Contextual(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# Get the paths for three consecutive slices
		slice_paths = self.get_consecutive_slices(idx)
		mask_path = self.maskPaths[idx] # Assuming masks are for the middle slice
		mask = nib.load(mask_path).get_fdata().astype(np.float32)
		original_size = mask.shape # (width, height)
		# Load three consecutive slices and their corresponding mask
		slices = [nib.load(path).get_fdata().astype(np.float32) for path in slice_paths]

		#convert images to tensors
		#slices = [torch.from_numpy(slice) for slice in slices]
		#mask = torch.from_numpy(mask)
		# Apply transformations
		if self.transforms is not None:
			slices = [self.transforms(slice) for slice in slices]
			mask = self.transforms(mask)

		# Stack the slices along the channel dimension to form a 3-channel input
		#image = np.stack(slices, axis=0)
		image = torch.stack(slices, axis=0)

		return (image, mask, original_size, self.imagePaths[idx], mask_path)

	def get_consecutive_slices(self, idx):
		# Extract slice number from file paths
		slice_number = int(self.imagePaths[idx].split('_')[-1].split('.')[0])

		# Get paths for three consecutive slices
		#first get the total number of slices to manage  boundary slices
		slice_files = os.listdir(os.path.dirname(self.imagePaths[idx]))
		MAX_SLICE_NUMBER = len(slice_files)
		slice_paths = []
		for i in range(-1, 2):
			current_slice_number = slice_number + i
			if current_slice_number < 0:
				current_slice_number = 0  # Repeat the first slice for out-of-bounds slices at the beginning
			elif current_slice_number >= MAX_SLICE_NUMBER:
				current_slice_number = MAX_SLICE_NUMBER - 1  # Repeat the last slice for out-of-bounds slices at the end

			slice_paths.append(self.get_slice_path(idx, current_slice_number))

		return slice_paths

	def get_slice_path(self, idx, slice_number):
		# Create slice path based on slice number
		file_name = self.imagePaths[idx].split('/')[-1]
		slice_path = '/'.join(self.imagePaths[idx].split('/')[:-1]) + '/' + '_'.join(file_name.split('_')[:-1]) + '_' + str(slice_number) + '.nii.gz'
		return slice_path

