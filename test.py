
import numpy as np
np.set_printoptions(suppress=True)

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
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
data_array = np.load('./result/data_array.npy',allow_pickle=True)
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Image.MAX_IMAGE_PIXELS = 10000000000


model_path = 'vgg16_bad1' #Xception
layer = -1
seek = True



model_id = 50

def namebase_get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list)
        return dir_list

all_patch_list = namebase_get_file_list('./result/' + str(model_path))
model_name = [s for s in np.asarray(all_patch_list) if (str(model_id)+'-') in s][0]
model1 = keras.models.load_model('./result/' + str(model_path) + '/' + model_name)
model2 = keras.models.load_model('./result/' + str(model_path) + '/' + model_name)


from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score0 = CategoricalScore([0, 0, 0, 0])
score1 = CategoricalScore([1, 1, 1, 1])

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

# Instead of using the ReplaceToLinear instance above,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

gc = Gradcam(model1,
                  model_modifier=replace2linear,
                  clone=True)

gcpp = GradcamPlusPlus(model2,
                  model_modifier=replace2linear,
                  clone=True)


def gradcam_3d(X, score):

    # Generate heatmap with GradCAM
    cam = gc(score,
                  X,
                  penultimate_layer=layer,seek_penultimate_conv_layer=seek)
    
    aa = copy.deepcopy((images)*0.)       
    aa[0] = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    aa[1] = np.flip((np.uint8(cm.jet(cam[1])[..., :3] * 255)),0)   
    aa[2] = np.rollaxis(np.uint8(cm.jet(cam[2])[..., :3] * 255),0,2)
    aa[3] = np.rot90((np.uint8(cm.jet(cam[3])[..., :3] * 255)),2)
    
    temp_cam_all = aa[0] + aa[1] + aa[2]  + aa[3]
    return temp_cam_all

def gradcamplusplus_3d(X, score):

    # Generate heatmap with GradCAM
    cam = gcpp(score,
                  X,
                  penultimate_layer=layer,seek_penultimate_conv_layer=seek)
    
    aa = copy.deepcopy((images)*0.)       
    aa[0] = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    aa[1] = np.flip((np.uint8(cm.jet(cam[1])[..., :3] * 255)),0)   
    aa[2] = np.rollaxis(np.uint8(cm.jet(cam[2])[..., :3] * 255),0,2)
    aa[3] = np.rot90((np.uint8(cm.jet(cam[3])[..., :3] * 255)),2)
    
    temp_cam_all = aa[0] + aa[1] + aa[2]  + aa[3]
    return temp_cam_all

def scorecam_3d(X,score):

    cam = sc(score,
                   X,
                   penultimate_layer=layer,
                   max_N=10)
    
    aa = copy.deepcopy((images)*0.)       
    aa[0] = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    aa[1] = np.flip((np.uint8(cm.jet(cam[1])[..., :3] * 255)),0)   
    aa[2] = np.rollaxis(np.uint8(cm.jet(cam[2])[..., :3] * 255),0,2)
    aa[3] = np.rot90((np.uint8(cm.jet(cam[3])[..., :3] * 255)),2)
    
    temp_cam_all = aa[0] + aa[1] + aa[2]  + aa[3]
    return temp_cam_all

patch_size = 256
overlap = 128
print(model_path, ' --> ', layer, '     seek = ' ,seek)

for types,mask_id,shapex,shapey in [(1,1,27579,19018),
                                     (1,3,14304,9020),
                                     (1,4,27558,28009),
                                     (4,1,5616,8022),
                                     (4,2,6122,10019),
                                     (4,3,18403,12020),
                                     (4,4,7136,11022),
                                     (4,5,5610,7022),
                                     (7,1,14819,12020),
                                     (7,4,11756,15018)]:
# for types,mask_id in progress_bar([(7,1)]):

    
    gc_result0 = np.zeros((shapex//2,shapey//2,3), dtype = np.float64)
    gc_result1 = copy.deepcopy(gc_result0)
    gcpp_result0 = copy.deepcopy(gc_result0)
    gcpp_result1 = copy.deepcopy(gc_result0)
#         sc_result = copy.deepcopy(gc_result)
    sub_mask    = np.zeros((shapex//2,shapey//2), dtype = np.float64)

#             total_size = bad_img_mask.shape[1],bad_img_mask.shape[0]

    for temp_patch in progress_bar(data_array[types][mask_id]):
        test_img = temp_patch[2]
        x = temp_patch[0]//2
        y = temp_patch[1]//2

        timg1 = np.asarray(test_img)
        images = np.asarray([np.array(timg1), np.array(np.flip(timg1,0)), np.array(np.rollaxis(timg1,0,2)), np.array((np.rot90(timg1,2)))])
        X = images/255

        gc_result_temp0 = gradcam_3d(X, score0)
        gc_result_temp1 = gradcam_3d(X, score1)
        gcpp_result_temp0 = gradcamplusplus_3d(X, score0)
        gcpp_result_temp1 = gradcamplusplus_3d(X, score1)
#             sc_result_temp = scorecam_3d(X)

        gc_result0[y:y+patch_size,x:x+patch_size] += gc_result_temp0
        gc_result1[y:y+patch_size,x:x+patch_size] += gc_result_temp1
        gcpp_result0[y:y+patch_size,x:x+patch_size] += gcpp_result_temp0
        gcpp_result1[y:y+patch_size,x:x+patch_size] += gcpp_result_temp1
#             sc_result[y:y+patch_size,x:x+patch_size] += sc_result_temp            

        sub_mask[y:y+patch_size,x:x+patch_size] += 4

    sub_mask = (np.stack([sub_mask,sub_mask,sub_mask],2))
    sub_mask[sub_mask==0] = 1
    gc_result0 = gc_result0/sub_mask
    gc_result1 = gc_result1/sub_mask
    gcpp_result0 = gcpp_result0/sub_mask
    gcpp_result1 = gcpp_result1/sub_mask
#         sc_result = sc_result/sub_mask

    Image.fromarray(np.asarray(gc_result0,   dtype = np.uint8)).save('./result/' + str(model_path) + '/gc0_'   + str(types) + '_' + str(mask_id) + '.png')
    Image.fromarray(np.asarray(gc_result1,   dtype = np.uint8)).save('./result/' + str(model_path) + '/gc1_'   + str(types) + '_' + str(mask_id) + '.png')
    Image.fromarray(np.asarray(gcpp_result0, dtype = np.uint8)).save('./result/' + str(model_path) + '/gcpp0_' + str(types) + '_' + str(mask_id) + '.png')
    Image.fromarray(np.asarray(gcpp_result1, dtype = np.uint8)).save('./result/' + str(model_path) + '/gcpp1_' + str(types) + '_' + str(mask_id) + '.png')