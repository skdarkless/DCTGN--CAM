import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pydot
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import copy
from collections import namedtuple
import os
import random
import shutil
import time
from PIL import Image,ImageDraw,ImageEnhance,ImageColor 
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_efficientnet_v2
import cv2
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

strategy = tf.distribute.MirroredStrategy()

def timebase_get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list
def namebase_get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list)
        return dir_list
    
all_patch_list = namebase_get_file_list(r'./train/patch512/')

train_temp_list_x = np.zeros((len(all_patch_list),7), dtype = np.object)
for i in  progress_bar(range(0,len(all_patch_list))):
    temp_label = all_patch_list[i].replace('.png','').split('_')
    train_temp_list_x[i,1] = temp_label[-3]
    train_temp_list_x[i,2] = 1 - float(temp_label[-3])
    train_temp_list_x[i,3] = temp_label[-2]
    train_temp_list_x[i,4] = 1 - float(temp_label[-2])
    train_temp_list_x[i,5] = temp_label[-1]
    train_temp_list_x[i,6] = 1 - float(temp_label[-1])
    
train_temp_list_x[:,0] = all_patch_list

train_temp_list_x = pd.DataFrame(train_temp_list_x)

train_temp_list_x.columns = ['x','bad11','bad12','bad21','bad22','good11','good12']



print(train_temp_list_x)

train_temp_list_x[['bad11']] = train_temp_list_x[['bad11']].astype('float')
train_temp_list_x[['bad21']] = train_temp_list_x[['bad21']].astype('float')
train_temp_list_x[['good11']] = train_temp_list_x[['good11']].astype('float')
train_temp_list_x[['bad12']] = train_temp_list_x[['bad12']].astype('float')
train_temp_list_x[['bad22']] = train_temp_list_x[['bad22']].astype('float')
train_temp_list_x[['good12']] = train_temp_list_x[['good12']].astype('float')


print(train_temp_list_x.dtypes)

print('')

print('total patchs: ',len(train_temp_list_x))
print('bad1:         ', np.sum(np.asarray(train_temp_list_x['bad11']>0)))
print('bad2:         ', np.sum(np.asarray(train_temp_list_x['bad21']>0)))
print('good1:        ', np.sum(np.asarray(train_temp_list_x['good11']>0)))

adam  = keras.optimizers.Adam(0.00005)
with strategy.scope():
#     model = keras_efficientnet_v2.EfficientNetV2S(pretrained="imagenet",num_classes=4,input_shape = (512,512,3))
#     model= tf.keras.applications.ResNet50(
#         include_top=True,
#         weights=None,
#         input_tensor=None,
#         input_shape=(256,256,3),
#         pooling=None,
#         classes=1,
#     )
    model= tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(256,256,3),
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
#                   loss=tf.keras.losses.BinaryCrossentropy(),
#                   loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
adam  = keras.optimizers.Adam(0.00005)
with strategy.scope():
    model = keras_efficientnet_v2.EfficientNetV2S(pretrained="imagenet",num_classes=4,input_shape = (512,512,3))
#     model= tf.keras.applications.ResNet152(
#         include_top=True,
#         weights=None,
#         input_tensor=None,
#         input_shape=(256,256,3),
#         pooling=None,
#         classes=1,
#     )

    model.compile(optimizer=adam,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
#                   loss=tf.keras.losses.BinaryCrossentropy(),
#                   loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
datagen=ImageDataGenerator(   
#             featurewise_center=True, 
#             featurewise_std_normalization=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,  # 比例平移
        zoom_range=[0.6,1.4],
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255)
#     datagen.mean = np.asarray([0.6577, 0.4877, 0.6195],dtype=np.float32)
#     datagen.std  = np.asarray([0.2125, 0.2510, 0.1968],dtype=np.float32)
train_generator=datagen.flow_from_dataframe(dataframe=train_temp_list_x, directory='./train/patch512/',
                                            x_col='x', y_col=list(['bad11','bad12']), class_mode='raw', target_size=(256,256), batch_size=128)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

rootpath = "./result/vgg16_bad1/"
filepath = rootpath + "/{epoch:02d}-{loss:.2f}.h5"
if(os.path.isdir(rootpath) == False):
    os.mkdir(rootpath)

checkpoint = keras.callbacks.ModelCheckpoint(filepath,  monitor='loss', save_best_only=False, verbose=1, period=1) 

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=None,
                    validation_steps=None,
                    epochs=50,
                    workers=30,
                    callbacks= [checkpoint]
        
                   )

