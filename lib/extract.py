# -------------------------------------------------------------------------------------------------------------------------------
# import necessary libraries for Milestone 1
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import re, cv2

from PIL import Image
import tensorflow as tf
# -------------------------------------------------------------------------------------------------------------------------------
# read the car/class names
carsMaster = pd.read_csv("Car names and make.csv",header=None)
carsMaster.columns=["fullNames"]
# -------------------------------------------------------------------------------------------------------------------------------
# before we process any information from the fullNames, lets remove any path separator '/' in the class names
carsMaster["fullNames"] = carsMaster["fullNames"].apply(lambda x: '-'.join(x.split('/')))
# -------------------------------------------------------------------------------------------------------------------------------
# lets first separate the OEM name & Year-of-Make data and review again
carsMaster["OEM"] = carsMaster["fullNames"].apply(lambda x: x.split()[0])
carsMaster["YEAR"] = carsMaster["fullNames"].apply(lambda x: x.split()[-1])

# also pickup the second word to verify if it was part of OEM name or Model name
carsMaster["chk"] = carsMaster["fullNames"].apply(lambda x: x.split()[1])
# -------------------------------------------------------------------------------------------------------------------------------
# manually listing the OEM names with 2 words based on above table
twinWordOE = ['AM', 'Aston', 'Land']
# -------------------------------------------------------------------------------------------------------------------------------
# lets update the OEM  names accordingly
dtmp = carsMaster.loc[carsMaster.OEM.isin(twinWordOE)]
carsMaster.loc[carsMaster.OEM.isin(twinWordOE),["OEM"]] = dtmp.fullNames.apply(lambda x: x.split()[:2])
# -------------------------------------------------------------------------------------------------------------------------------
# update model names in the dataframe
carsMaster["MODEL"] = carsMaster.apply(lambda row: [w for w in row["fullNames"].split() if w not in row["OEM"] and w!=str(row["YEAR"])],axis=1)
# -------------------------------------------------------------------------------------------------------------------------------
# extract the TYPE info
carsMaster["TYPE"] = carsMaster.MODEL.apply(lambda x: x[-1])
# -------------------------------------------------------------------------------------------------------------------------------
# lets update the TYPE
for t in ['Type-S', 'R', 'GS',  'ZR1', 'Z06', 'Abarth', 'XKR']:
    carsMaster.loc[carsMaster.TYPE == t,"TYPE"] = 'UnKnown'
carsMaster.loc[carsMaster.TYPE == 'IPL',"TYPE"] = "Coupe"
carsMaster.loc[carsMaster.TYPE == 'Cab',"TYPE"] = carsMaster.loc[carsMaster.TYPE == 'Cab',"MODEL"].apply(lambda x: x[-2:])
carsMaster.loc[carsMaster.TYPE == 'Van',"TYPE"] = carsMaster.loc[carsMaster.TYPE == 'Van',"MODEL"].apply(lambda x: x[-2:])
carsMaster.loc[carsMaster.TYPE == 'SRT-8',"TYPE"] = "SRT8"
# -------------------------------------------------------------------------------------------------------------------------------
# now lets update the MODEL name excluding the TYPE information
carsMaster["MODEL"] = carsMaster.apply(lambda row: [w for w in row["fullNames"].split() if w not in row["OEM"] and w!=str(row["YEAR"]) and w not in row["TYPE"]],axis=1)
# -------------------------------------------------------------------------------------------------------------------------------
# lets properly combine the OEM names & Model Names without lists
carsMaster["OEM"] = carsMaster["OEM"].apply(lambda x: x if type(x)==str else '_'.join(x))
carsMaster["MODEL"] = carsMaster["MODEL"].apply(lambda x: x if type(x)==str else '_'.join(x))
carsMaster["TYPE"] = carsMaster["TYPE"].apply(lambda x: x if type(x)==str else '_'.join(x))
# -------------------------------------------------------------------------------------------------------------------------------
# lets drop & rearrange the master data
carsMaster = carsMaster[["fullNames","OEM","MODEL","TYPE","YEAR"]]
# -------------------------------------------------------------------------------------------------------------------------------
# reference paths
BASEfldr = 'Car Images/'
TRAINfldr = 'Train Images/'
TESTfldr = 'Test Images/'
# -------------------------------------------------------------------------------------------------------------------------------
# lets take a record of data about the training imagess
tfi = tf.keras.preprocessing.image
path = os.path.join(BASEfldr,TRAINfldr)
iCols = ["Image","ImagePath","folderName"]
imageMasterTrain = pd.DataFrame(columns=iCols)
imPath = np.empty(0)
fldrName = np.empty(0)
imageName = np.empty(0)
imH = np.empty(0)
imW = np.empty(0)
for cls in tqdm(carsMaster.fullNames,desc="imScanTrain"):
    # we can also do this with if os.isdir() check
    try:
        os.listdir(path+cls)
    except:
        print("path error: ",path+cls)
        continue
    for img in os.listdir(path+cls):
        imPath = np.append(imPath,np.array([path+cls+'/'+img]))
        fldrName = np.append(fldrName,np.array([cls]))
        imageName = np.append(imageName,np.array([img]))
        (w,h) = tfi.load_img(path+cls+'/'+img).size
        imH = np.append(imH,np.array([h]))
        imW = np.append(imW,np.array([w])) 
        
imageMasterTrain["Image"] = imageName
imageMasterTrain["ImagePath"] = imPath
imageMasterTrain["folderName"] = fldrName
imageMasterTrain["height"] = imH
imageMasterTrain["width"] = imW
# -------------------------------------------------------------------------------------------------------------------------------
# lets take a record of data about the testing imagess
path= os.path.join(BASEfldr,TESTfldr)
iCols = ["Image","ImagePath","folderName"]
imageMasterTest = pd.DataFrame(columns=iCols)
imPath = np.empty(0)
fldrName = np.empty(0)
imageName = np.empty(0)
imH = np.empty(0)
imW = np.empty(0)
for cls in tqdm(carsMaster.fullNames,desc="imScanTest"):
    # we can also do this with if os.isdir() check
    try:
        os.listdir(path+cls)
    except:
        print("path error: ",cls)
        continue
    for img in os.listdir(path+cls):
        imPath = np.append(imPath,np.array([path+cls+'/'+img]))
        fldrName = np.append(fldrName,np.array([cls]))
        imageName = np.append(imageName,np.array([img]))
        (w,h) = tfi.load_img(path+cls+'/'+img).size
        imH = np.append(imH,np.array([h]))
        imW = np.append(imW,np.array([w])) 
imageMasterTest["Image"] = imageName
imageMasterTest["ImagePath"] = imPath
imageMasterTest["folderName"] = fldrName
imageMasterTest["height"] = imH
imageMasterTest["width"] = imW
# -------------------------------------------------------------------------------------------------------------------------------
# let us read the annotations datafile to pandas dataframe
trainAnnot = pd.read_csv('./Annotations/Train Annotations.csv')
testAnnot = pd.read_csv('./Annotations/Test Annotation.csv')
Acols = ['Image Name', 'x1', 'y1', 'x2','y2', 'Image class']
trainAnnot.columns = Acols
testAnnot.columns = Acols
# -------------------------------------------------------------------------------------------------------------------------------
# create all-consolidated dataframes
trainDF = pd.merge(imageMasterTrain,trainAnnot,how='outer',left_on='Image',right_on='Image Name')
testDF = pd.merge(imageMasterTest,testAnnot,how='outer',left_on='Image',right_on='Image Name')
# -------------------------------------------------------------------------------------------------------------------------------
# lets merge the OEM, MODEL, TYPE & YEAR data
trainDF = pd.merge(trainDF,carsMaster,how='outer',left_on='folderName',right_on='fullNames')
testDF = pd.merge(testDF,carsMaster,how='outer',left_on='folderName',right_on='fullNames')
# -------------------------------------------------------------------------------------------------------------------------------
# update class index to start from ZERO
trainDF["Image class"] = trainDF["Image class"]-1
testDF["Image class"] = testDF["Image class"]-1
# -------------------------------------------------------------------------------------------------------------------------------
# merge cars_names_and_make csv data with the annotation class name field
trainDF = pd.merge(trainDF,carsMaster,how='outer',left_on='Image class',right_index=True)
testDF = pd.merge(testDF,carsMaster,how='outer',left_on='Image class',right_index=True)
# though this will duplicate the already exisiting folderName, fullNames columns, this adds a cross check for data correctness
# -------------------------------------------------------------------------------------------------------------------------------
# finalize the images dataframe
trainDF = trainDF[["ImagePath",'x1','y1','x2','y2',"height","width","folderName","OEM_x","MODEL_x","TYPE_x","YEAR_x",]]
testDF = testDF[["ImagePath",'x1','y1','x2','y2',"height","width","folderName","OEM_x","MODEL_x","TYPE_x","YEAR_x",]]

trainDF.columns = ["ImagePath",'x1','y1','x2','y2',"height","width","className","OEM","MODEL","TYPE","YEAR"]
testDF.columns = ["ImagePath",'x1','y1','x2','y2',"height","width","className","OEM","MODEL","TYPE","YEAR"]
