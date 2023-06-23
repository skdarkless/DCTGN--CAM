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
import datetime
import pandas as pd
import cv2
# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
from fastprogress import progress_bar
import torchvision.models as models
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision

#网络==============

model_path = 'vgg16_bad2'
save_path = './torchresult/' + model_path + '/'

temp_y = 'bad11'

model = models.vgg16().to(device)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=2, bias=True).to(device)
model = nn.DataParallel(model)

#结束===============

batch_size = 128
validation_split = .05
shuffle_dataset = True
random_seed= 42

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


dataset_size = len(train_temp_list_x)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation((0,180)),
#     transforms.RandomResizedCrop(224,scale=(0.7,1.3),ratio=(0.75, 1.33),interpolation=2)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    
    transforms.RandomAffine(degrees = 180, translate= (0.2,0.2), scale=(0.7,1.3), shear=25, resample=False, fillcolor=0) ,

    
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0,180)),
    transforms.ToTensor()])

class Arthopod_Dataset(Dataset):
    def __init__(self, img_data,img_path,transform=None):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,
                                self.img_data.loc[index, 'x'])
#         print(img_name)
        image = Image.open(img_name)
#         image = np.asarray(Image.open(img_name), dtype = np.double)/255
        #image = image.convert('RGB')
        image = image.resize((224,224))
#         label = torch.tensor([self.img_data.loc[index, 'bad11'],self.img_data.loc[index, 'bad12']]).type(torch.LongTensor)
        label = torch.tensor(self.img_data.loc[index, temp_y]).type(torch.LongTensor)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

dataset  = Arthopod_Dataset(train_temp_list_x,'./train/patch512/',train_transform)
dataset2 = Arthopod_Dataset(train_temp_list_x,'./train/patch512/',test_transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, num_workers=35)

validation_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=35)


if(os.path.isdir(save_path) == False):
    os.mkdir(save_path)



#============



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

train_temp_list_x[['bad11']] = train_temp_list_x[['bad11']].astype('float')
train_temp_list_x[['bad21']] = train_temp_list_x[['bad21']].astype('float')
train_temp_list_x[['good11']] = train_temp_list_x[['good11']].astype('float')
train_temp_list_x[['bad12']] = train_temp_list_x[['bad12']].astype('float')
train_temp_list_x[['bad22']] = train_temp_list_x[['bad22']].astype('float')
train_temp_list_x[['good12']] = train_temp_list_x[['good12']].astype('float')

#训练============

n_epochs = 50
print_every = 1000
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs+1):
    print(datetime.datetime.now())
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_loader):
        data_, target_ = data_.to(device), target_.to(device)# on GPU
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t in (validation_loader):
            data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(1000 * correct_t / total_t)
        val_loss.append(batch_loss/len(validation_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight 
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), save_path + 'best.pth')
#             torch.save('model_classification_tutorial.pth')
            print('Detected network improvement, saving current model')
    model.train()