"""
# for future codes, import the following dataframes as final output from this Milestone1
# * trainDF
# * testDF

# reference paths
# BASEfldr = 'Car Images/'
# TRAINfldr = 'Train Images/'
# TESTfldr = 'Test Images/'

# use following code to invoke & import data
# from Milestone1 import trainDF, testDF;
"""

"""
Folder Structure to follow
==========================

any_location
 |
 |--- Milestone1.py
 |--- Milestone1.ipynb (optional)
 |--- sample_Code_for_milestone1_output.ipynb (optional)
 |
 |--- Car names and make.csv
 |
 |--- Annotations
 |    |
 |    |--- Test Annotation.csv
 |    |--- Train Annotations.csv
 |
 |--- Car Images
      |
      |--- Train Images
      |    |
      |    |--- *** 196 folders with train images
      |
      |--- Test Images
           |
           |--- *** 196 folders with test images

"""
#!/usr/bin/env python
# coding: utf-8

# # ***Deep Learning based Car Identification***
# ##### *- Automotive, Surveillance, Object Detection & Localisation*

# ***Project By:***<br>
# Premjeet Kumar https://www.linkedin.com/in/premjeet-kumar/<br>
# Hari Samynaath S https://www.linkedin.com/in/harinaathan/<br>
# Veena Raju https://www.linkedin.com/in/veena-raju-1b16b513b/<br>
# Javed Bhai https://www.linkedin.com/in/javedbhai/<br>
# Surabhi Joshi https://www.linkedin.com/in/surabhi-joshi-4452788/<br>
# 
# ***Project For:***<br>
# Captstone project for **Post Graduate Program in Artificial Intelligence and Machine Learning**<br>
# with *GreatLakes & Texas McCombs School of Business, The University of Texas at Austin*

# **CONTEXT:**<br>
# Computer vision can be used to automate supervision and generate action appropriate action trigger if the event is predicted from the image of interest. For example a car moving on the road can be easily identi ied by a camera as make of the car, type, colour, number plates etc.<br><br>
# **DATA DESCRIPTION:**<br>
# The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.<br>
# <br>
# ‣ *Train Images:* Consists of real images of cars as per the make and year of the car.<br>
# ‣ *Test Images:* Consists of real images of cars as per the make and year of the car.<br>
# ‣ *Train Annotation:* Consists of bounding box region for training images.<br>
# ‣ *Test Annotation:* Consists of bounding box region for testing images.<br>

# ### **MILESTONE 1:**
# ‣ *Step 1:* Import the data<br>
# ‣ *Step 2:* Map training and testing images to its classes.<br>
# ‣ *Step 3:* Map training and testing images to its annotations.<br>
# ‣ *Step 4:* Display images with bounding box<br>
# ‣ *Output:* Images mapped to its class and annotation ready to be used for deep learning<br>


# import necessary libraries for Milestone 1
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import re, cv2

from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

# read the car/class names
carsMaster = pd.read_csv("Car names and make.csv",header=None)
carsMaster.columns=["fullNames"]

# lets review the name lengths
carsMaster["wCounts"] = carsMaster["fullNames"].apply(lambda x: len(x.split()))

# before we process any information from the fullNames, lets remove any path separator '/' in the class names
carsMaster["fullNames"] = carsMaster["fullNames"].apply(lambda x: '-'.join(x.split('/')))

# lets first separate the OEM name & Year-of-Make data and review again
carsMaster["OEM"] = carsMaster["fullNames"].apply(lambda x: x.split()[0])
carsMaster["YEAR"] = carsMaster["fullNames"].apply(lambda x: x.split()[-1])

# also pickup the second word to verify if it was part of OEM name or Model name
carsMaster["chk"] = carsMaster["fullNames"].apply(lambda x: x.split()[1])

# lets review on basis of OEM
dtmp = carsMaster.groupby(by="OEM")["chk"].unique()

# the suspects for 2 word OEM names are whereever there are only 1 uniques against the extracted first name of the OE
# lets try to short list those and review better
carsMaster.loc[carsMaster.OEM.isin(dtmp.loc[carsMaster.groupby(by="OEM")["chk"].nunique()==1].index)]

# manually listing the OEM names with 2 words based on above table
twinWordOE = ['AM', 'Aston', 'Land']

# lets update the OEM  names accordingly
dtmp = carsMaster.loc[carsMaster.OEM.isin(twinWordOE)]
carsMaster.loc[carsMaster.OEM.isin(twinWordOE),["OEM"]] = dtmp.fullNames.apply(lambda x: x.split()[:2])

# update model names in the dataframe
carsMaster["MODEL"] = carsMaster.apply(lambda row: [w for w in row["fullNames"].split() if w not in row["OEM"] and w!=str(row["YEAR"])],axis=1)

# lets review the model names and extract model type information from it
carsMaster["mwCounts"] = carsMaster.MODEL.apply(lambda x: len(x))

# it should be noted that, Cab & Van comes with 2 word coach type<br>
# othewise, almost every other words are part of model name only<br>
# hence lets separate the coach type

# extract the TYPE info
carsMaster["TYPE"] = carsMaster.MODEL.apply(lambda x: x[-1])

# **findings**<br>
# * the type IPL hides Coupe type before it<br>
# * 'Type-S', 'R', 'GS',  'ZR1', 'Z06', 'Abarth', 'XKR' types are not coach types, hence to be markes as unKnown<br>
# * 'SS', 'SRT-8', 'SRT8' could be considered as car type (though not coach type) as they are technology/class of car<br>
# 
# lets update the TYPE accordingly

for t in ['Type-S', 'R', 'GS',  'ZR1', 'Z06', 'Abarth', 'XKR']:
    carsMaster.loc[carsMaster.TYPE == t,"TYPE"] = 'UnKnown'
    carsMaster.loc[carsMaster.TYPE == 'IPL',"TYPE"] = "Coupe"
    carsMaster.loc[carsMaster.TYPE == 'Cab',"TYPE"] = carsMaster.loc[carsMaster.TYPE == 'Cab',"MODEL"].apply(lambda x: x[-2:])
    carsMaster.loc[carsMaster.TYPE == 'Van',"TYPE"] = carsMaster.loc[carsMaster.TYPE == 'Van',"MODEL"].apply(lambda x: x[-2:])
    carsMaster.loc[carsMaster.TYPE == 'SRT-8',"TYPE"] = "SRT8"

# now lets update the MODEL name excluding the TYPE information
carsMaster["MODEL"] = carsMaster.apply(lambda row: [w for w in row["fullNames"].split() if w not in row["OEM"] and w!=str(row["YEAR"]) and w not in row["TYPE"]],axis=1)

# lets properly combine the OEM names & Model Names without lists
carsMaster["OEM"] = carsMaster["OEM"].apply(lambda x: x if type(x)==str else '_'.join(x))
carsMaster["MODEL"] = carsMaster["MODEL"].apply(lambda x: x if type(x)==str else '_'.join(x))
carsMaster["TYPE"] = carsMaster["TYPE"].apply(lambda x: x if type(x)==str else '_'.join(x))

# lets drop & rearrange the master data
carsMaster = carsMaster[["fullNames","OEM","MODEL","TYPE","YEAR"]]

# there are imbalances in the dataset might create bias in the model's capabilities
# Lets also read the image data files in to our notebook, and review the distribution once again

# reference paths
BASEfldr = 'Car Images/'
TRAINfldr = 'Train Images/'
TESTfldr = 'Test Images/'

# lets take a record of data about the training imagess
path = BASEfldr+TRAINfldr
iCols = ["Image","ImagePath","folderName","height","width"]
imageMasterTrain = pd.DataFrame(columns=iCols)
imPath = np.empty(0)
fldrName = np.empty(0)
imageName = np.empty(0)
imH = np.empty(0)
imW = np.empty(0)
for cls in tqdm(carsMaster.fullNames,desc="imScanTrain"):
    try:
        os.listdir(path+cls)
    except:
        print("path error: ",path+cls)
        continue
    for img in os.listdir(path+cls):
        imPath = np.append(imPath,np.array([path+cls+'/'+img]))
        fldrName = np.append(fldrName,np.array([cls]))
        imageName = np.append(imageName,np.array([img]))
        (w,h) = Image.open(path+cls+'/'+img).size
        imH = np.append(imH,np.array([h]))
        imW = np.append(imW,np.array([w]))
imageMasterTrain["Image"] = imageName
imageMasterTrain["ImagePath"] = imPath
imageMasterTrain["folderName"] = fldrName
imageMasterTrain["height"] = imH
imageMasterTrain["width"] = imW

# lets take a record of data about the testing imagess
path = BASEfldr+TESTfldr
iCols = ["Image","ImagePath","folderName","height","width"]
imageMasterTest = pd.DataFrame(columns=iCols)
imPath = np.empty(0)
fldrName = np.empty(0)
imageName = np.empty(0)
imH = np.empty(0)
imW = np.empty(0)
for cls in tqdm(carsMaster.fullNames,desc="imScanTest"):
    try:
        os.listdir(path+cls)
    except:
        print("path error: ",cls)
        continue
    for img in os.listdir(path+cls):
        imPath = np.append(imPath,np.array([path+cls+'/'+img]))
        fldrName = np.append(fldrName,np.array([cls]))
        imageName = np.append(imageName,np.array([img]))
        (w,h) = Image.open(path+cls+'/'+img).size
        imH = np.append(imH,np.array([h]))
        imW = np.append(imW,np.array([w]))
imageMasterTest["Image"] = imageName
imageMasterTest["ImagePath"] = imPath
imageMasterTest["folderName"] = fldrName
imageMasterTest["height"] = imH
imageMasterTest["width"] = imW

# compute image sizes
imageMasterTrain["pixels"] = imageMasterTrain.height * imageMasterTrain.width
imageMasterTest["pixels"] = imageMasterTest.height * imageMasterTest.width

# based on above review, we shall restrict the image size fed to the network at 50x50 pixels, so as not to detoriate lower resolution images and thus affect model capabilities

# having connected to the images directories, lets also add the annotations, and add the bounding boxes to the images

# let us read the annotations datafile to pandas dataframe
trainAnnot = pd.read_csv('./Annotations/Train Annotations.csv')
testAnnot = pd.read_csv('./Annotations/Test Annotation.csv')
Acols = ['Image Name', 'x1', 'y1', 'x2','y2', 'Image class']
trainAnnot.columns = Acols
testAnnot.columns = Acols

# create all-consildated dataframes
trainDF = pd.merge(imageMasterTrain,trainAnnot,how='outer',left_on='Image',right_on='Image Name')
testDF = pd.merge(imageMasterTest,testAnnot,how='outer',left_on='Image',right_on='Image Name')

# lets merge the OEM, MODEL, TYPE & YEAR data
trainDF = pd.merge(trainDF,carsMaster,how='outer',left_on='folderName',right_on='fullNames')
testDF = pd.merge(testDF,carsMaster,how='outer',left_on='folderName',right_on='fullNames')

# update class index to start from ZERO
trainDF["Image class"] = trainDF["Image class"]-1
testDF["Image class"] = testDF["Image class"]-1

# merge cars_names_and_make csv data with the annotation class name field
trainDF = pd.merge(trainDF,carsMaster,how='outer',left_on='Image class',right_index=True)
testDF = pd.merge(testDF,carsMaster,how='outer',left_on='Image class',right_index=True)
# though this will duplicate the already exisiting folderName, fullNames columns, this adds a cross check for data correctness

# lets us now remove unwanted columns and make the dataframe more readable
# finalize the images dataframe
trainDF = trainDF[["Image","ImagePath",'x1','y1','x2','y2','height','width',"folderName","Image class","OEM_x","MODEL_x","TYPE_x","YEAR_x",]]
testDF = testDF[["Image","ImagePath",'x1','y1','x2','y2','height','width',"folderName","Image class","OEM_x","MODEL_x","TYPE_x","YEAR_x",]]

trainDF.columns = ["Image","ImagePath",'x1','y1','x2','y2','height','width',"folderName","Image_class","OEM","MODEL","TYPE","YEAR"]
testDF.columns = ["Image","ImagePath",'x1','y1','x2','y2','height','width',"folderName","Image_class","OEM","MODEL","TYPE","YEAR"]

# All the data preprocessing & compilation have been completed so far<br>
# The data were imported and mapped against their respectivee classses & annotations<br>
# Comprehensive dataframes for training & testing datasets were created and could be used with generators for Deep Learning Networks<br>
# ### **MILESTONE 1 COMPLETE**
# 

# for future codes, import the following dataframes as final output from this Milestone1<br>
# * trainDF
# * testDF

# use following code to invoke & import data
# from Milestone1 import trainDF, testDF;
