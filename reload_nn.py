import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageFile
import PIL.ImageOps    
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import glob
import copy
import time

ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
# torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu" # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create the Siamese Neural Network
resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT) 
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Flatten()
for param in resnet.parameters():
            param.requires_grad = False
resnet = resnet.to(device)
resnet.eval()

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Linear(num_ftrs_resnet*2, 1)

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors which are returned
        print(np.shape(input1))
        print(np.shape(input2))
        output12 = torch.cat((input1, input2),1)
        print(np.shape(output12))
        output = self.fc(output12)
        return output

net = SiameseNetwork()

net.load_state_dict(torch.load('./data_224/my_net_2023_08_14_18_44_04.pt'))

net = net.to(device)

net.eval()

## Testing Method 1: input each sample one by one without using data loader (without resnet)
# label_files = glob.glob('./data_224/labels/*')
# num_files = len(label_files)

# correct = 0
# wrong = 0
# # output the specific failure sample if neccessary
# for i in range(num_files):
#     rnd_label = label_files[i]
#     label = np.load(rnd_label)     

#     rnd_pile = np.char.replace(rnd_label, 'labels', 'pile_imgs')
#     img_pile = np.load(str(rnd_pile))

#     rnd_mask = np.char.replace(rnd_label, 'labels', 'mask_imgs')
#     img_mask = np.load(str(rnd_mask))

#     img = torch.from_numpy(np.array(img_pile))
#     mask = torch.from_numpy(np.array(img_mask))
#     label = torch.from_numpy(np.array([int(label)], dtype=np.float32))

#     img, mask, label = img.to(device), mask.to(device), label.to(device)
#     output = net(img.unsqueeze(0), mask.unsqueeze(0))
#     pred = (torch.sigmoid(output) > 0.5)

#     if pred==label:
#         correct += 1
#     else:
#         wrong += 1

# print(f'The accuracy of the trained model is {100*correct/num_files}')

## Testing Method 2: input each sample one by one without using data loader (with resnet)
transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

label_files = glob.glob('./data_224/labels/*')
num_files = len(label_files)

correct = 0
wrong = 0
# output the specific failure sample if neccessary
# with torch.no_grad():
for i in range(num_files):
    rnd_label = label_files[i]
    label = np.load(rnd_label)     

    rnd_pile = np.char.replace(rnd_label, 'labels', 'pile_imgs')
    rnd_pile = np.char.replace(rnd_pile, 'npy', 'jpg')
    img_pile = Image.open(str(rnd_pile))     

    rnd_mask = np.char.replace(rnd_label, 'labels', 'mask_imgs')
    rnd_mask = np.char.replace(rnd_mask, 'npy', 'jpg')
    img_mask = Image.open(str(rnd_mask))

    img_pile = transformation(img_pile)
    img_mask = transformation(img_mask)
    img_pile, img_mask = img_pile.to(device), img_mask.to(device)

    img = resnet(img_pile.unsqueeze(0))
    mask = resnet(img_mask.unsqueeze(0))

    label = torch.from_numpy(np.array([int(label)], dtype=np.float32))
    label = label.to(device)

    output = net(img, mask)
    pred = (torch.sigmoid(output) > 0.5)

    if pred==label:
        correct += 1
    else:
        wrong += 1

print(f'The accuracy of the trained model is {100*correct/num_files}')

## Testing Method 3: input the sample using data loader as training

# def split_train_val_tes(file_path, num_ = None, ratio_=None):
#     pile_files = glob.glob(file_path)
#     len_ = len(pile_files)

#     if num_ is not None:
#         num_train, num_val, num_tes = num_
#     else:
#         num_train = np.int32(len_*ratio_[0])
#         num_val = min(np.int32(len_*ratio_[1]), len_-num_train)
#         num_tes = min(np.int32(len_*ratio_[2]), len_-num_train-num_val)

#     pile_train = np.random.choice(pile_files, size=num_train, replace=False)    

#     pile_files = list(set(pile_files)-set(pile_train))
#     pile_val = np.random.choice(pile_files, size=num_val, replace=False)

#     pile_files = list(set(pile_files)-set(pile_val))
#     pile_tes = np.random.choice(pile_files, size=num_tes, replace=False)

#     train_ = pile_train
#     val_ = pile_val
#     tes_ = pile_tes

#     return pile_files, train_, val_, tes_

# class SiameseNetworkDataset(Dataset):
#     def __init__(self,file_path):
#         self.label_files = file_path
#         self.len_ = len(self.label_files)

#     def __getitem__(self,idx):
#         rnd_label = self.label_files[idx]
#         label = np.load(rnd_label)     

#         rnd_pile = np.char.replace(rnd_label, 'labels', 'pile_imgs')
#         img_pile = np.load(str(rnd_pile))

#         rnd_mask = np.char.replace(rnd_label, 'labels', 'mask_imgs')
#         img_mask = np.load(str(rnd_mask))

#         # return img_pile, img_mask, torch.from_numpy(label_, dtype=np.float32)
#         return torch.from_numpy(np.array(img_pile)), \
#             torch.from_numpy(np.array(img_mask)), \
#                 torch.from_numpy(np.array([int(label)], dtype=np.float32))

#     def __len__(self):
#         return self.len_

# # Load the training dataset
# pile_files, train_, val_, tes_= split_train_val_tes(file_path='./data_224/labels/*', ratio_=[0.7,0.2,0.1])

# all_dataset = SiameseNetworkDataset(file_path=pile_files)
# all_dataloader = DataLoader(all_dataset,
#                         shuffle=True,
#                         num_workers=1,
#                         batch_size=1)

# # with torch.no_grad():
# correct_t = 0
# total_t = 0
# i = 0 
# for img0, img1, label in all_dataloader:
#     img0, img1, label = img0.to(device), img1.to(device), label.to(device)

#     output = net(img0, img1)
#     pred = (torch.sigmoid(output) > 0.5)
#     correct_t += torch.sum(pred==label).item()
#     total_t += label.size(0)
#     i += 1

#     output_ = torch.sigmoid(output)
# #   print(output_.cpu().numpy().reshape(-1))
# #   print(label.cpu().numpy().reshape(-1)) 

# tes_acc = 100 * correct_t/total_t
# print(f'test acc: {tes_acc:.4f}\n')

######################################################################################

# pile_files = glob.glob('./data_224/pile_imgs/*')
# num_files = len(pile_files)

# correct_his = []
# for j in range(100):
#     print(j)
#     pile_file = []
#     pile_file.append(pile_files[j])

#     single_dataset = SiameseNetworkDataset(file_path=pile_file, transform=transformation)
#     single_dataloader = DataLoader(single_dataset,shuffle=True,num_workers=1,batch_size=1)
#     with torch.no_grad():
#         correct_t = 0
#         total_t = 0
#         for i, (img0, img1, label) in enumerate(single_dataloader, 0):
#             img0, img1, label = img0.to(device), img1.to(device), label.to(device)

#             output = net(img0, img1)
#             pred = (torch.sigmoid(output) > 0.5)
#             correct_t += torch.sum(pred==label).item()
#             total_t += label.size(0)

#             output_ = torch.sigmoid(output)
#             print(output_.cpu().numpy().reshape(-1))
#             print(label.cpu().numpy().reshape(-1)) 

#             tes_acc = 100 * correct_t/total_t
#             print(f'test acc: {tes_acc:.4f}\n')
#     correct_his.append(tes_acc)
#     print(np.mean(correct_his))



















