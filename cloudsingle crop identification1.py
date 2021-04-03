

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:53:41 2020

@author: jaykumar.d.padariya
"""

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
#from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from keras.preprocessing.image import load_img
import glob
import os
import h5py
import tensorflow as tf
from tensorflow import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras.optimizers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pandas import DataFrame as df
path=r"D:\DRIMA\PHOTOS"
dataforpics=[]
dataforpics=df(dataforpics)
names_crop=[]

all_pic=[]



#%%
path = 'D:\DRIMA\PHOTOS'

#path=r"D:\DRIMA\PHOTOS"
dataforpics=[]
dataforpics=df(dataforpics)
names_crop=[]
Avastha=[]
all_pic=[]       
listmajorcrop= ['Bajra','castor','cotton','wheat','Maize']        
all_pic=[]        
for j in listmajorcrop:   
    dist = os.path.join(path,j)
    print(j)
    print(dist)            
#    print(dist)
    for jj in os.listdir(dist):
        
#        print(len(os.listdir(dist)))
        dist1 = os.path.join(dist,jj)
        print(jj)
        print(dist1)
      
        for jjj in os.listdir(dist1):
#            lenght=(len(os.listdir(dist1)))
#    print(j,jj,len(os.listdir(dist1)))
#    for i in range (num):
##        print(i)
#        df1["crop_nam"] = j1[0].values
#        df1.loc[i]= [j,jj,len(os.listdir(dist1))]
#            
#            print(jjj)
#            print(jj)
            dist2 = os.path.join(dist1,jjj)
#            print(dist2)
            all_pic.append(dist2)
            name=dist2.split("PHOTOS\\")[1].split("\\")[0]
            names_crop.append(name)
            avstha=dist2.split("\\",4)[-1].split("\\")[0]
            Avastha.append(avstha)

            
            
 
            
            
      
all_pic1=df(all_pic,columns=['images'])
all_pic1=df({'images':all_pic , 'names' : names_crop , 'Avastha' : Avastha})



#%%
#all_pic1["id"] = np.nan
# 
#indexNames = all_pic1[ all_pic1['Avastha'] == 'char pan avastha' ].index
# 
## Delete these row indexes from dataFrame
#all_pic1.drop(indexNames , inplace=True)
#indexNames = all_pic1[ all_pic1['Avastha'] == 'char pand avastha' ].index
# 
## Delete these row indexes from dataFrame
#all_pic1.drop(indexNames , inplace=True)
#
#
#indexNames = all_pic1[ all_pic1['Avastha'] == 'chae pan avastha'  ].index
# 
## Delete these row indexes from dataFrame
#all_pic1.drop(indexNames , inplace=True)
#
#
#indexNames = all_pic1[ all_pic1['Avastha'] == 'ful avastha' ].index
# 
## Delete these row indexes from dataFrame
#all_pic1.drop(indexNames , inplace=True)
#
#all_pic1.names.value_counts()


result=pd.read_csv("D:\model\cloud\inception_generator_COLLAB40epochandmodifiedmodeltotal-18100False-1011True-17089with_avstha.csv")


#%%

all_pic1["id"] = np.nan
 
all_pic1=all_pic1.reset_index(drop=True)
    

for index, i in enumerate(all_pic1['names']):
#    print(i,index)
#    
    if i == 'Bajra':
       all_pic1['id'][index] = '0'
        
    if i == 'castor':
       all_pic1['id'][index] = '1'
       
    if i == 'wheat':
       all_pic1['id'][index] = '2'
       
    if i == 'cotton':
       all_pic1['id'][index] = '3'    
       
    if i == 'Maize':
       all_pic1['id'][index] = '4'       

all_pic1=all_pic1.dropna()   


#      
#
#all_pic1["id"] = np.nan
# 
#
#    
#
#for index, i in enumerate(all_pic1['names']):
##    print(i,index)
##    
#    if i == 'Bajra':
#       all_pic1['id'][index] = '0'
#        
#    if i == 'castor':
#       all_pic1['id'][index] = '1'
#       
#    if i == 'wheat':
#       all_pic1['id'][index] = '2'
#       
#    if i == 'cotton':
#       all_pic1['id'][index] = '3'    
#       
#    if i == 'Maize':
#       all_pic1['id'][index] = '4'       
        
    #    
#
#for index, i in enumerate(all_pic1['names']):
##    print(i,index)
##    
#    if i == 'Bajra':
#       all_pic1['id'][index] = '0'
#        
#    if i == 'castor':
#       all_pic1['id'][index] = '1'
#       
#    if i == 'wheat':
#       all_pic1['id'][index] = '2'
#       
#    if i == 'cotton':
#       all_pic1['id'][index] = '3'    
        


bajra1 = all_pic1[((all_pic1['names'] == 'Bajra') & (all_pic1['Avastha'] == 'char pan avastha'))]  
wheat1=all_pic1[((all_pic1['names'] == 'wheat') & (all_pic1['Avastha'] == 'char pan avastha'))]  
cotton1=all_pic1[((all_pic1['names'] == 'cotton') & (all_pic1['Avastha'] == 'char pan avastha'))]  
castor1=all_pic1[((all_pic1['names'] == 'castor') & (all_pic1['Avastha'] == 'char pan avastha'))]  
Maize1=all_pic1[((all_pic1['names'] == 'Maize') & (all_pic1['Avastha'] == 'char pan avastha'))]  

bajra2=all_pic1[((all_pic1['names'] == 'Bajra') & (all_pic1['Avastha'] == 'ful avastha'))]  
wheat2=all_pic1[((all_pic1['names'] == 'wheat') & (all_pic1['Avastha'] == 'ful avastha'))]  
cotton2=all_pic1[((all_pic1['names'] == 'cotton') & (all_pic1['Avastha'] == 'ful avastha'))]  
castor2=all_pic1[((all_pic1['names'] == 'castor') & (all_pic1['Avastha'] == 'ful avastha'))]  
Maize2=all_pic1[((all_pic1['names'] == 'Maize') & (all_pic1['Avastha'] == 'ful avastha'))]  
   
bajra3=all_pic1[((all_pic1['names'] == 'Bajra') & (all_pic1['Avastha'] == 'fal avastha'))]  
wheat3=all_pic1[((all_pic1['names'] == 'wheat') & (all_pic1['Avastha'] == 'fal avastha'))]  
cotton3=all_pic1[((all_pic1['names'] == 'cotton') & (all_pic1['Avastha'] == 'fal avastha'))]  
castor3=all_pic1[((all_pic1['names'] == 'castor') & (all_pic1['Avastha'] == 'fal avastha'))]  
Maize3=all_pic1[((all_pic1['names'] == 'Maize') & (all_pic1['Avastha'] == 'fal avastha'))]  


#all_pic1.Avastha.value_counts()



#all_pic2=[]    
#all_pic2=df(all_pic2)      
#all_pic2=all_pic2.append(bajra1.head(2500))
#all_pic2=all_pic2.append(wheat1.head(2500))
#all_pic2=all_pic2.append(cotton1.head(2500))
#all_pic2=all_pic2.append(castor1.head(2500))
#all_pic2=all_pic2.append(Maize1.head(2500))
#
#all_pic2=all_pic2.append(bajra2.head(1000))
#all_pic2=all_pic2.append(wheat2.head(1000))
#all_pic2=all_pic2.append(cotton2.head(1000))
#all_pic2=all_pic2.append(castor2.head(1000))
#all_pic2=all_pic2.append(Maize2.head(1000))
#
#all_pic2=all_pic2.append(bajra3.head(700))
#all_pic2=all_pic2.append(wheat3.head(700))
#all_pic2=all_pic2.append(cotton3.head(700))
#all_pic2=all_pic2.append(castor3.head(700))
#all_pic2=all_pic2.append(Maize3.head(700))
###
#
all_pic2=[]    
all_pic2=df(all_pic2)     
all_pic2=all_pic2.append(bajra1[2500:])
all_pic2=all_pic2.append(wheat1[2500:])
all_pic2=all_pic2.append(cotton1[2500:])
all_pic2=all_pic2.append(castor1[2500:])
all_pic2=all_pic2.append(Maize1[2500:])

all_pic2=all_pic2.append(bajra2[1000:])
all_pic2=all_pic2.append(wheat2[1000:])
all_pic2=all_pic2.append(cotton2[1000:])
all_pic2=all_pic2.append(castor2[1000:])
all_pic2=all_pic2.append(Maize2[1000:])

all_pic2=all_pic2.append(bajra3[1000:])
all_pic2=all_pic2.append(wheat3[1000:])
all_pic2=all_pic2.append(cotton3[1000:])
all_pic2=all_pic2.append(castor3[1000:])
all_pic2=all_pic2.append(Maize3[1000:])
all_pic2.names.value_counts()
      

#all_pic1["Avastha"] = 'ful avastha'
# 
#all_pic1=all_pic1.reset_index(drop=True)
#    
#
#for index, i in enumerate(all_pic2['names']):
##    print(i,index)
##    
#    if i == 'Bajra':
#       all_pic2['id'][index] = '0'
#        
#    if i == 'castor':
#       all_pic2['id'][index] = '1'
#       
#    if i == 'wheat':
#       all_pic2['id'][index] = '2'
#       
#    if i == 'cotton':
#       all_pic2['id'][index] = '3'    
#       
#    if i == 'Maize':
#       all_pic2['id'][index] = '4'       

del bajra1,wheat1,cotton1,castor1,Maize1
del bajra2,wheat2,cotton2,castor2,Maize2
del bajra3,wheat3,cotton3,castor3,Maize3

#msk = np.random.rand(len(all_pic2)) < 0.8

#train = all_pic2[msk]

#test = all_pic2[~msk]








#a=all_pic2[((result['original_image'] == 'Bajra'))]
#a.resu.value_counts()
## 
#%%


msk = np.random.rand(len(all_pic2)) < 0.8

train = all_pic2[msk]

test = all_pic2[~msk]

testfile=list(test['images'])
Label=list(test['id'])
ClassName=list(test['names'])


test_df = df({'FileName': testfile, 'Label': Label,'ClassName': ClassName})

testfile=list(train['images'])
Label=list(train['id'])
ClassName=list(train['names'])

testfile=list(all_pic2['images'])
Label=list(all_pic2['id'])
ClassName=list(all_pic2['names'])



train_df = df({'FileName': testfile, 'Label': Label,'ClassName': ClassName})


#%%




from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

datagen=ImageDataGenerator(rescale=1/255,validation_split=0.25)








train_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory=None,
x_col="FileName",
y_col="ClassName",
subset="training",
batch_size=32,
seed=None,
shuffle=True,
class_mode="categorical",
target_size=(299,299))

valid_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory=None,
x_col="FileName",
y_col="ClassName",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(299,299))




test_datagen=ImageDataGenerator(rescale=1/255)
test_generator=test_datagen.flow_from_dataframe(
dataframe=test_df,
directory=None,
x_col="FileName",
y_col=None,
batch_size=32,
seed=NOne,
shuffle=True,
class_mode=None,
target_size=(299,299))










#%%

batch_size=32
STEP_SIZE_TRAIN=len(train_df)/batch_size
STEP_SIZE_VALID=5111/batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

#%%

data = []
labels = []
da=[]


for imagePath in all_pic2['images'][:10]:
#    print(imagePath)
	# load the image, pre-process it, and store it in the data list
    img = image.load_img(imagePath,target_size=(299,299))
    img = image.img_to_array(img)
    img = img/255
    data.append(img)
data = np.array(data, dtype="float")

#%%save aaray
#np.save(r'D:\ml_aray_for_moedl\trainingarray299_299', data)


#%%load array

#loaded_array = np.load('D:\ml_aray_for_moedl\Majorcropwithallavsthadata.npy')
#%%
labels = to_categorical(all_pic2['id'][:10])
print(labels.shape)

#del all_pic1,all_pic2
    
        

#%%
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=100)



aug = ImageDataGenerator(rotation_range=50, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.3, zoom_range=0.3,
	horizontal_flip=True, fill_mode="nearest")
#%%
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
        if K.image_data_format() == "channels_first":
        	inputShape = (depth, height, width)
        	chanDim = 1

		# CONV => RELU => POOL
        model.add(Conv2D(8, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        		# (CONV => RELU) * 2 => POOL
        model.add(Conv2D(1024, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
                
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
       
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

       
        		# first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(20))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        		# softmax classifier
        model.add(Dense(classes))
        model.add(Activation(finalAct))
        
        		# return the constructed network architecture
        return model

    
#%%
# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")

EPOCHS = 2
INIT_LR = 1e-3
BS = 1
IMAGE_DIMS = (299, 299, 3)


model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=5,
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


model.summary()


#%%

from keras.callbacks import LearningRateScheduler
epochs = EPOCHS
initial_lrate = INIT_LR
import math
def decay(epoch, steps=100):
    initial_lrate = 0.001
    drop = 0.96
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


lr_sc = LearningRateScheduler(decay, verbose=1)



#%%
del all_pic,Avastha,avstha,data,dataforpics,dist,dist1,dist2,names_crop,path,da
del img,i,imagePath,j,jj,jjj,label,labels,listmajorcrop,name



model.fit_generator(
        train_generator,
        steps_per_epoch=8 // BS,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=8 // BS)



#%%

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1,callbacks=[lr_sc])

score = model.evaluate(testX,testY, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

#%%
#model save

import pickle
with open('D:\model\MODELSsingle_crop-26-2_OWN_INCEPTIONepoch90PERACCU{}.h5'.format(epochs), 'wb') as f:
    pickle.dump(model, f)
    
    
    
#save model history
with open('D:\model\HISTORYSsingle_crophostory2000EPPOCH26-2_OWN_INCEPTIONepoch90PERACCU{}'.format(epochs), 'wb') as file_pi:
    pickle.dump(model.history, file_pi)


    #wights
model.save_weights('D:\model\WEIGHTSsingle_crop26-2_OWN_INCEPTIONepoch90PERACCU{}.h5'.format(epochs))
#%%

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("D:\model\1labelbintest13-2-2020{}".format(EPOCHS), "wb")
f.write(pickle.dumps(mlb))
f.close()
    
#%%
#testing
img = image.load_img(r'D:\DRIMA\PHOTOS\wheat\char pan avastha\Wheat_88f53_1545118544.jpg',target_size=(100,100,3))
img = image.img_to_array(img)
img = img/255

classes = np.array(all_pic2.columns[2:])
proba = model.predict(img.reshape(1,100,100,3))

print(proba)

probas = np.array(proba)
labels = np.argmax(probas, axis=-1)   
#print(labels)
list11=['Bajra','castor','wheat','cotoon']
for i in range(0,4):
    if labels == i:
        print(labels,list11[i])
        


#%%   

indexNames = all_pic1[ all_pic1['Avastha'] == 'ful avastha' ].index
 
# Delete these row indexes from dataFrame
all_pic1.drop(indexNames , inplace=True)

indexNames = all_pic1[ all_pic1['Avastha'] == 'fal avastha' ].index
 
# Delete these row indexes from dataFrame
all_pic1.drop(indexNames , inplace=True)




#%% for internal images

result=[] 
photo1=[]
labels1=[]
labname=[]       
def funb(photo):     
    img = load_img(photo,target_size=(299,299,3))
    img = img_to_array(img)
    img = img/255
    
    classes = np.array(all_pic1.columns[2:])
    proba = model.predict(img.reshape(1,299,299,3))
    
    print(proba)
    
    probas = np.array(proba)
    labels = np.argmax(probas, axis=-1)   
    print(labels)
    list11=['Bajra','castor','wheat','cotton','Maize']
    for i in range(len(list11)):
        if labels == i:
            print(labels,list11[i])
            lab=list11[i]
            photo1.append(photo)
            labels1.append(labels)
            labname.append(lab)

randomchoice=np.random.choice(all_pic2['images'], 1000, replace=False)
randomchoice=list(randomchoice)
for photo in all_pic2['images']:
    print(photo) 
    funb(photo)
      
result = df({'photo':photo1 , 'label' : labels1 , 'name' : labname })  


li=[]
for i in result['photo']:
    ii=i.split("PHOTOS\\")[1].split("\\")[0]
   
    li.append(ii)
result = df({'photo':photo1 , 'label' : labels1 , 'predicted_crop' : labname , 'original_image':li })  
result["resu"] = np.nan
for index,co in enumerate(result['predicted_crop']):
    
    
    if result['predicted_crop'][index] == result['original_image'][index]:
        result['resu'][index] = 'True'
    else:
        result['resu'][index] ='False'
        
        
print(result.resu.value_counts())
result.predicted_crop.value_counts()
result.original_image.value_counts()

#
#
#new=[]
#for i in randomchoice:
#    ii=i.split('\\PHOTOS\\')[1].split('\\')[0]
#    print(ii)
#    new.append(ii)
#    
#result1 = df({'photo':randomchoice , 'label' : new })    
#result1.label.value_counts()
#%% for outside images
path=r"D:\images_crop"
listmajorcrop= ['Bajra','castor','cotton','wheat','Maize']        
all_pic=[] 
label=[]       
for j in listmajorcrop:   
    dist = os.path.join(path,j)
    print(j)
    print(dist)            
#    print(dist)
    for jj in os.listdir(dist):
        
#        print(len(os.listdir(dist)))
        dist1 = os.path.join(dist,jj)
        print(jj)
        print(dist1)
        all_pic.append(dist1)
        lab=dist.split('\\',2)[2]
        label.append(lab)
        
      
all_pic1=df(all_pic,columns=['images'])
all_pic1=df({'images':all_pic , 'name_crop_original' : label})
        

#%%  For outside images
result=[] 
photo1=[]
labels1=[]
labname=[]       
def funb(photo):     
    img = load_img(photo,target_size=(100,100,3))
    img = img_to_array(img)
    img = img/255
    
    classes = np.array(all_pic1.columns[2:])
    proba = model.predict(img.reshape(1,100,100,3))
    
    print(proba)
    
    probas = np.array(proba)
    labels = np.argmax(probas, axis=-1)   
    print(labels)
    list11=['Bajra','castor','wheat','cotton','Maize']
    for i in range(len(list11)):
        print(i)
        if labels == i:
            print(labels,list11[i])
            lab=list11[i]
            photo1.append(photo)
            labels1.append(labels)
            labname.append(lab)

randomchoice=np.random.choice(all_pic1['images'], len(all_pic1), replace=False)
randomchoice=list(randomchoice)
for photo in False1['photo']:
    print(photo) 
    funb(photo)
      
result = df({'photo':photo1 , 'label' : labels1 , 'name' : labname })  


li=[]
ava=[]
for i in result['photo']:
    ii=i.split('\\',4)[4].split('\\')[0]
    iii=i.split('\\',5)[5].split("\\")[0]
    print(ii)
    li.append(ii)
    ava.append(iii)
result1 = df({'photo':photo1 , 'label' : labels1 , 'predicted_crop' : labname , 'original_image':li ,'Avastha' : ava })  
result1["resu"] = np.nan
for index,co in enumerate(result1['predicted_crop']):
    print(index)
    
    if result1['predicted_crop'][index] == result1['original_image'][index]:
        result1['resu'][index] = 'True'
    else:
        result1['resu'][index] ='False'
        
       
result1.resu.value_counts()
#%%
#RESULT TO CSV
result.to_csv(r'D:\model\cloud\resnet\RESNET-20200306T092104Z-001\RESNET\RESNET_299-299-F-1163-T-16937.csv') 

#false images to csv
False1.to_csv(r'D:\model\Fasleimagesafterallmodel.csv') 


#%% CONFUSION METRICS


y_classes = [np.argmax(y, axis=None, out=None) for y in testY]


y_pred=model.predict_classes(testX)

con_mat = tf.math.confusion_matrix(labels=y_classes, predictions=y_pred).numpy()


index = ['Bajra','castor','wheat','cotton','Maize']  
columns =  ['Bajra','castor','wheat','cotton','Maize']  
cm_df = pd.DataFrame(con_mat,columns,index)                      


#%%false have take in one Df
False1 = result[(result['resu'] == 'False')]
False1.reset_index(inplace = True) 


df = pd.DataFrame({'photo': [], 'label': [], 'original_image': [], 'resu': [], 'predicted_iamge': []})

for index,i in enumerate(all_pic2['resu']):
    if i == 'False':
        for i in enumerate(False1['resu']):
            df.append({'photo': all_pic2['photo'][index] , 'original_image':all_pic2['original_image'][index]  , 'resu': i, 'predicted_iamge':all_pic2['predicted_iamge'][index] }, ignore_index=True)
    else:
        pass
                
charpan=False1[:4936]
fulavas=False1[4935:6000]
falavas=False1[6000:]

#%%
#load model
pkl_filename=r"D:\model\MODELFOR_50EPOCH MAJOR 5 CROPALL_AVSTHA_12000_IMAGE_ACCU_90_\MODELSsingle_crop-14-2-2020epoch90PERACCU50.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)
    

pkl_filename=r"D:\model\MODELFOR_10EPOCH MAJOR 5 CROP_80CHAR_PANL_AVASTHA_12000_IMAGE_ACCU_79_\MODELSsingle_crop-13-2-2020epoch10.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)
    
pkl_filename=r"D:\model\MODELFOR_10EPOCH MAJOR 5 CROP_80_PER_FUL_AVASTHA_1000_IMAGE_ACCU_77\MODELSsingle_crop-13-2-2020epoch10.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)

pkl_filename=r"D:\model\MODELFOR_10EPOCH MAJOR 5 CROP_80CHAR_PANL_AVASTHA_12000_IMAGE_ACCU_79_\MODELSsingle_crop-13-2-2020epoch10.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)


D:\model\MODELFOR_50EPOCH MAJOR 5 CROPALL_AVSTHA_12000_IMAGE_ACCU_90_\MODELSsingle_crop-14-2-2020epoch90PERACCU50.h5


pkl_filename=r"D:\model\cloud\resnet\RESNET-20200306T092104Z-001\RESNET\Modelresnet_299_299_20_epoch.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)

#%%for false
result=[] 
photo1=[]
labels1=[]
labname=[]       
def funb(photo):     
    img = load_img(photo,target_size=(100,100,3))
    img = img_to_array(img)
    img = img/255
    
    classes = np.array(all_pic1.columns[2:])
    
    proba = model.predict(img.reshape(1,100,100,3))
    
    print(proba)
    
    probas = np.array(proba)
    labels = np.argmax(probas, axis=-1)   
    print(labels)
    list11=['Bajra','castor','wheat','cotton','Maize']
    for i in range(len(list11)):
        if labels == i:
            print(labels,list11[i])
            lab=list11[i]
            photo1.append(photo)
            labels1.append(labels)
            labname.append(lab)

rando=np.random.choice(False1['photo'], len(False1), replace=False)
rando=list(rando)
for photo in rando:
    print(photo) 
    funb(photo)
      
result = df({'photo':photo1 , 'label' : labels1 , 'name' : labname })  


li=[]
for i in result['photo']:
    ii=i.split("PHOTOS\\")[1].split('\\')[0]
    print(ii)
    li.append(ii)
result = df({'photo':photo1 , 'label' : labels1 , 'predicted' : labname , 'image':li })  
result["resu"] = np.nan
for index,co in enumerate(result['predicted']):
    print(index)
    
    if result['predicted'][index] == result['image'][index]:
        result['resu'][index] = 'True'
    else:
        result['resu'][index] ='False'
        
        
result.resu.value_counts()

result.predicted.value_counts()

#%%

import pickle
with open('D:\model\MULTILABLEmodel4-2-2020{}.h5'.format(EPOCHS), 'wb') as f:
    pickle.dump(model, f)
    
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("D:\model\labelbintest{}".format(EPOCHS), "wb")
f.write(pickle.dumps(mlb))
f.close()

#weights   
    
model.save_weights('D:\model\MULTICLASSweightsg4-2-2020{}.h5'.format(EPOCHS))
    


#%%load model

# load the trained convolutional neural network and the multi-label
# binarizer
pkl_filename=r"D:\model\MODELFOR_10EPOCH MAJOR 5 CROP_80CHAR_PANL_AVASTHA_12000_IMAGE_ACCU_79_\MODELSsingle_crop-13-2-2020epoch10.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)
    
#score = clf.score(X_test, y_test)
#
#Ypredict = pickle_model.predict(Xtest)
#        
#    
#    
#print("[INFO] loading network...")
#model = load_model(args["model"])
mlb = pickle.loads(open("D:\model\labelbintest50EPPOCH-7-2-2020.h5", "rb").read())

#%%


from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

img = load_img(r'D:\DRIMA\PHOTOS\Bajra\fal avastha\Bajara_958a9_1556347738.jpg',target_size=(100,100,3))
img = load_img(r'D:\DRIMA\PHOTOS\castor\ful avastha\aeranda_aff02_1544694326.jpg',target_size=(100,100,3))
img = load_img(r'D:\DRIMA\PHOTOS\cotton\ful avastha\Cotton#કપાસ_68e78_1564989188.jpg',target_size=(100,100,3))


img = load_img(r'D:\DRIMA\PHOTOS\wheat\char pan avastha\Wheat_88f53_1545118544.jpg',target_size=(100,100,3))
img = img_to_array(img)
img = img/255
image =img
output = img
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
#	cv2.putText(output, label, (10, (i * 30) + 25), 
#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
    
    
    
    
#%%



# load the trained convolutional neural network and the multi-label
# binarizer
pkl_filename=r"D:\model\MULTILABLEmodel4-2-20205080PERCENT.h5"
with open(pkl_filename, 'rb') as file:
     model = pickle.load(file)
    
#score = clf.score(X_test, y_test)
#
#Ypredict = pickle_model.predict(Xtest)
#        
#    
#    
#print("[INFO] loading network...")
#model = load_model(args["model"])
mlb = pickle.loads(open("D:\model\labelbintest", "rb").read())


    
 #%%
from operator import itemgetter
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import time
result=[]
result1=[]
result2=[]
resultcrop=[]
resultavstha=[]
teriphoto1=[]
teriAVASTHA=[]
randomchoice=np.random.choice(all_pic, 100, replace=False)
randomchoice=list(randomchoice)

def fun(teriphoto):
    
    img = load_img(teriphoto,target_size=(100,100,3))
    img = img_to_array(img)
    img = img/255
    image1 =img
    output = img
    image2 = np.expand_dims(image1, axis=0)
    
    # load the trained convolutional neural network and the multi-label
    # binarizer
    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image2)[0]
    print(proba)
    idxs = np.argsort(proba)[::-1][:2]
    print(teriphoto)
    print(idxs)
    print(mlb.classes_[idxs][0],)
    print(mlb.classes_[idxs][1],)
   
    
    
    
    
    
    
    if mlb.classes_[idxs][0] == mlb.classes_[0]:
        
        resultcrop.append(mlb.classes_[idxs][0])   
        resultavstha.append(mlb.classes_[idxs][1]) 
        
    elif mlb.classes_[idxs][0] == mlb.classes_[1]:
        resultcrop.append(mlb.classes_[idxs][0])   
        resultavstha.append(mlb.classes_[idxs][1]) 
    elif mlb.classes_[idxs][0] ==mlb.classes_[3]:
        resultcrop.append(mlb.classes_[idxs][0])   
        resultavstha.append(mlb.classes_[idxs][1]) 
    elif mlb.classes_[idxs][0] == mlb.classes_[6]:
        resultcrop.append(mlb.classes_[idxs][0])   
        resultavstha.append(mlb.classes_[idxs][1]) 
    elif mlb.classes_[idxs][0] == mlb.classes_[2]:
        resultavstha.append(mlb.classes_[idxs][0])  
        resultcrop.append(mlb.classes_[idxs][1])     
    elif mlb.classes_[idxs][0] == mlb.classes_[4]:  
        resultavstha.append(mlb.classes_[idxs][0])  
        resultcrop.append(mlb.classes_[idxs][1])     
    elif mlb.classes_[idxs][0] == mlb.classes_[5]:
        resultavstha.append(mlb.classes_[idxs][0])  
        resultcrop.append(mlb.classes_[idxs][1])     
    else:
        pass
        
#        label = "{}: {:.2f}%".format(mlb.classes_[idxs][0], proba[idxs][0] * 100)
#        label1 = "{}: {:.2f}%".format(mlb.classes_[idxs][1], proba[idxs][1] * 100)
#        resultcrop.append(mlb.classes_[idxs][0])   
#        resultavstha.append(mlb.classes_[idxs][1]) 
#    else:
#        pass
#    if mlb.classes_[idxs][0] == 'char pan avastha' or mlb.classes_[idxs][0] == 'fal avastha' or mlb.classes_[idxs][0] == 'ful avastha':
##        label1 = "{}: {:.2f}%".format(mlb.classes_[idxs][0], proba[idxs][0] * 100)
##        label = "{}: {:.2f}%".format(mlb.classes_[idxs][1], proba[idxs][1] * 100)
#        resultavstha.append(mlb.classes_[idxs][0])  
#        resultcrop.append(mlb.classes_[idxs][1])     
#    else:
#        pass
#    
##    
#    
#    if mlb.classes_[idxs][0] == mlb.classes_[0] or mlb.classes_[idxs][0] == mlb.classes_[1] or mlb.classes_[idxs][0] == mlb.classes_[3] or mlb.classes_[idxs][0] == mlb.classes_[6]:
#        label = "{}: {:.2f}%".format(mlb.classes_[idxs][0], proba[idxs][0] * 100)
#        label1 = "{}: {:.2f}%".format(mlb.classes_[idxs][1], proba[idxs][1] * 100)
#        resultcrop.append(mlb.classes_[idxs][0])   
#        resultavstha.append(mlb.classes_[idxs][1]) 
#    else:
#        pass
#    if mlb.classes_[idxs][0] == mlb.classes_[2] or mlb.classes_[idxs][0] == mlb.classes_[4] or mlb.classes_[idxs][0] == mlb.classes_[5]:
#        label1 = "{}: {:.2f}%".format(mlb.classes_[idxs][0], proba[idxs][0] * 100)
#        label = "{}: {:.2f}%".format(mlb.classes_[idxs][1], proba[idxs][1] * 100)
#        resultavstha.append(mlb.classes_[idxs][0])  
#        resultcrop.append(mlb.classes_[idxs][1])     
#    else:
#        pass
  
    teriphoto2=teriphoto.split("PHOTOS\\")[1].split('\\')[0]  
    TERIAVSTHA=teriphoto.split("\\",4)[-1].split("\\")[0]
    teriphoto1.append(teriphoto2)
    teriAVASTHA.append(TERIAVSTHA)
    
for i in randomchoice:
    fun(i)
    time.sleep(0.3)
    
        
result=df(result,columns=['mainphoto'])
result=df({'mainphoto':teriphoto1 , 'mainavstha' : teriAVASTHA , 'predictedcrop':resultcrop,'predictedavstha':resultavstha})
#result=df({'name':result1 , 'percen' : result2 , 'resultcrop':resultcrop,'resultavstha':resultavstha})

result["resucrop"] = np.nan
result["resuavsatha"] = np.nan


for index,co in enumerate(result['mainphoto']):
    print(index)
    
    if result['mainphoto'][index] == result['predictedcrop'][index]:
        result['resucrop'][index] = 'True'
    else:
        result['resucrop'][index] ='False'
    if result['mainavstha'][index] == result['predictedavstha'][index]:
        result['resuavsatha'][index] = 'True'
    else:
        result['resuavsatha'][index] ='False'    
        
print('crop----',result.resucrop.value_counts())
print('avastha---',result.resuavsatha.value_counts())


#listforcropprob=list(itemgetter(0,1,3,6)(bhaibhai))
#listforavsthaprob=list(itemgetter(2,4,5)(bhaibhai))
#
#res_maxcrop = max(float(sub) for sub in listforcropprob)     
#res_maxavstha = max(float(sub) for sub in listforavsthaprob)     
#print("")
#        
      