"""
Author: Roa'a Khaled
Affiliation: Department of Computer Engineering, Department of Condensed Matter Physics,
             University of Cádiz, , Puerto Real, 11519, Cádiz, Spain.
Email: roaa.khaled@gm.uca.es
Date: 2025-07-27
Project: This work is part of a predoctoral research and part of PARENT project that has received funding
         from the EU’s Horizon 2020 research and innovation program under the MSCA – ITN 2020, GA No 956394.

Description:
    This code trains a 2D Unet or a Contextual UNet (3 input slices) for a segmenation task, in our case
    we segmented Total Brains from cranial Ultrasound images of premature born infants.

Usage:
    python train_2d_or_Contextual_UNet.py

Notes:
	- if you want to train a contextual UNet, change the NUM_CHANNELS variable in UNet2D_pyImageSearch.training_config
	  to be 3 or more (depending on how many input slices you want), also import SegmentationDataset_Contextual instead
	  of SegmentationDataset_2D from UNet2D_pyImageSearch.dataset
    - for environment requirements check
    - this code is adapted from (https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
"""

# import the necessary packages
from UNet2D_pyImageSearch.dataset import SegmentationDataset_2D #or SegmentationDataset_Contextual for contextual data
from UNet2D_pyImageSearch.model import UNet
from UNet2D_pyImageSearch import training_config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from SaveDataTable import save_to_csv
#from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

# load the training image and mask filepaths in a sorted manner
patientPaths = [f.path for f in os.scandir(training_config.TRAIN_IMAGE_PATH) if f.is_dir()]
#imagePaths = sorted(list(paths.list_images(config.TRAIN_IMAGE_PATH)))
patient_GT_Paths = [f.path for f in os.scandir(training_config.TRAIN_GT_PATH) if f.is_dir()]
#maskPaths = sorted(list(paths.list_images(config.TRAIN_GT_PATH)))

# partition the training data into training and validation splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(patientPaths, patient_GT_Paths,
	test_size=training_config.VALIDATION_SPLIT, random_state=42) #train-validation split
# unpack the data split
(trainPatients, validationPatients) = split[:2]
(trainPatients_GT, validationPatients_GT) = split[2:]

#obtain the training and validation images paths
trainImage_paths = [os.path.join(root, file) for patient_path in trainPatients for root, _, files in os.walk(patient_path) for file in files]
trainImage_GT_paths = [os.path.join(root, file) for patient_GT_path in trainPatients_GT for root, _, files in os.walk(patient_GT_path) for file in files]

validationImage_paths = [os.path.join(root, file) for patient_path in validationPatients for root, _, files in os.walk(patient_path) for file in files]
validationImage_GT_paths = [os.path.join(root, file) for patient_GT_path in validationPatients_GT for root, _, files in os.walk(patient_GT_path) for file in files]

# write the training image paths to disk
print("[INFO] saving training patients paths...")
f = open(training_config.TRAINING_PATIENTS_PATHS, "w")
f.write("\n".join(trainPatients))
f.close()

print("[INFO] saving training images paths...")
f = open(training_config.TRAINING_IMAGES_PATHS, "w")
f.write("\n".join(trainImage_paths))
f.close()

# write the validation image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving validation patients paths...")
f = open(training_config.VALIDATION_PATIENTS_PATHS, "w")
f.write("\n".join(validationPatients))
f.close()

print("[INFO] saving validation images paths...")
f = open(training_config.VALIDATION_IMAGES_PATHS, "w")
f.write("\n".join(validationImage_paths))
f.close()

# load the testing image and mask filepaths in a sorted manner
testing_patientPaths = [f.path for f in os.scandir(training_config.TEST_IMAGE_PATH) if f.is_dir()]
#testing_patient_GT_Paths = [f.path for f in os.scandir(config.TEST_GT_PATH) if f.is_dir()]

#obtain the testing images paths
testImage_paths = [os.path.join(root, file) for patient_path in testing_patientPaths for root, _, files in os.walk(patient_path) for file in files]
#testImage_GT_paths = [os.path.join(root, file) for patient_GT_path in testing_patient_GT_Paths for root, _, files in os.walk(patient_GT_path) for file in files]

# write the testing image paths to disk so that we can use then
# when testing our model
print("[INFO] saving testing patients paths...")
f = open(training_config.TESTING_PATIENTS_PATHS, "w")
f.write("\n".join(testing_patientPaths))
f.close()

print("[INFO] saving testing images paths...")
f = open(training_config.TESTING_IMAGES_PATHS, "w")
f.write("\n".join(testImage_paths))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToTensor(),
 	transforms.Resize((training_config.INPUT_IMAGE_HEIGHT,
	training_config.INPUT_IMAGE_WIDTH)),
	transforms.Normalize((0.0,), (1.0,))
	]) ##removed this --> transforms.ToPILImage() ## add this to normalize--> transforms.Normalize((0.0,), (1.0,))
# create the train and test datasets
trainDS = SegmentationDataset_2D(imagePaths=trainImage_paths, maskPaths=trainImage_GT_paths,
	transforms=transforms)
testDS = SegmentationDataset_2D(imagePaths=validationImage_paths, maskPaths=validationImage_GT_paths,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the validation set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=training_config.BATCH_SIZE, pin_memory=training_config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=training_config.BATCH_SIZE, pin_memory=training_config.PIN_MEMORY,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(training_config.DEVICE)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=training_config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // training_config.BATCH_SIZE
testSteps = len(testDS) // training_config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(training_config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(training_config.DEVICE), y.to(training_config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x.squeeze())
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(training_config.DEVICE), y.to(training_config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x.squeeze())
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, training_config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))

	# save the training and testing loss in csv file
	save_to_csv(training_config.train_test_loss_path,
				[{'Epoch number': e + 1, 'Train loss': avgTrainLoss.cpu().detach().numpy(), 'Test loss': avgTestLoss.cpu().detach().numpy()}],
				append=True)


# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}min".format(
	round((endTime - startTime) / 60, 1)))
#save the training time in csv file
save_to_csv(training_config.training_time_path,
				[{'training time (min)': round((endTime - startTime) / 60, 1)}],
				append=True)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(training_config.PLOT_PATH)
# serialize the model to disk
torch.save(unet, training_config.MODEL_PATH)
