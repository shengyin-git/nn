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

#create the Siamese Neural Network
device = "cuda" if torch.cuda.is_available() else "cpu" # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        # self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT) #.to(device)
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT) 
        # self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT) 
       
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs_resnet = self.resnet.fc.in_features

        self.resnet.fc = nn.Flatten() # Identity() #

        # # Setting up the Fully Connected Layers
        self.fc = nn.Linear(num_ftrs_resnet*2, 1)

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors which are returned
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)

        output12 = torch.cat((output1, output2),1)
        output = self.fc(output12)
        return output


net = SiameseNetwork()

net.load_state_dict(torch.load('./data_224/my_net_2023_08_14_14_06_10.pt'))

net = net.to(device)

# net.eval()

################################################################################

pile_files = glob.glob('./data_224/pile_imgs/*')

num_files = len(pile_files)

correct = 0
wrong = 0
# output the specific failure sample if neccessary

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])
with torch.no_grad():
    for i in range(100):
        img_path = pile_files[i]
        img = Image.open(img_path)

        msk_path = np.char.replace(img_path, 'pile_imgs', 'mask_imgs')
        mask = Image.open(str(msk_path))

        label_path = np.char.replace(img_path, 'pile_imgs', 'labels')
        label_path = np.char.replace(label_path, 'jpg', 'npy')
        label = np.load(str(label_path))

        print(label)
        
        img, mask = transformation(img), transformation(mask)
        label = torch.from_numpy(np.array([int(label)], dtype=np.float32))

        img = torch.cat(8*[img.unsqueeze(0)])
        mask = torch.cat(8*[mask.unsqueeze(0)])
        label = torch.cat(8*[label])

        # print(np.shape(img))
        # print(np.shape(mask))
        # print(np.shape(label))
        # input()

        img, mask, label = img.to(device), mask.to(device), label.to(device)

        # print(np.shape(img.unsqueeze(0)))
        # input()

        output = net(img, mask)

        pred = (torch.sigmoid(output) > 0.5)

        print(torch.sigmoid(output))

        correct = torch.sum(pred==label).item()

        print(correct)

        # if pred==label:
        #     correct += 1
        #     # print('correct')
        # else:
        #     wrong += 1
        #     # print('wrong')

        # input()

print(f'The accuracy of the trained model is {100*correct/num_files}')

input()

######################################################################################

def split_train_val_tes(file_path, num_ = None, ratio_=None):
    pile_files = glob.glob(file_path)
    len_ = len(pile_files)

    ori_pile_files = pile_files

    if num_ is not None:
        num_train, num_val, num_tes = num_
    else:
        num_train = np.int32(len_*ratio_[0])
        num_val = min(np.int32(len_*ratio_[1]), len_-num_train)
        num_tes = min(np.int32(len_*ratio_[2]), len_-num_train-num_val)

    pile_train = np.random.choice(pile_files, size=num_train, replace=False)    

    pile_files = list(set(pile_files)-set(pile_train))
    pile_val = np.random.choice(pile_files, size=num_val, replace=False)

    pile_files = list(set(pile_files)-set(pile_val))
    pile_tes = np.random.choice(pile_files, size=num_tes, replace=False)

    train_ = pile_train
    val_ = pile_val
    tes_ = pile_tes

    return ori_pile_files, train_, val_, tes_

class SiameseNetworkDataset(Dataset):
    def __init__(self,file_path,transform=None):
        self.transform = transform
        self.pile_files = file_path
        self.len_ = len(self.pile_files)

    def __getitem__(self,idx):
        rnd_pile = self.pile_files[idx]
        img_pile = Image.open(rnd_pile)     
        ##########################################
        # rnd_pile = random.choice(self.pile_files)
        # img_pile = Image.open(rnd_pile)
        ##########################################

        rnd_mask = np.char.replace(rnd_pile, 'pile_imgs', 'mask_imgs')
        img_mask = Image.open(str(rnd_mask))

        rnd_label = np.char.replace(rnd_pile, 'pile_imgs', 'labels')
        rnd_label = np.char.replace(rnd_label, 'jpg', 'npy')
        label_ = np.load(str(rnd_label))

        if self.transform is not None:
            img_pile = self.transform(img_pile)
            img_mask = self.transform(img_mask)

        # return img_pile, img_mask, torch.from_numpy(label_, dtype=np.float32)
        return img_pile, img_mask, torch.from_numpy(np.array([int(label_)], dtype=np.float32))

    def __len__(self):
        return self.len_

# Load the training dataset
# Resize the images and transform to tensors
pile_files, train_, val_, tes_ = split_train_val_tes(file_path='./data_224/pile_imgs/*', num_=[1500,150,150])
# pile_files, train_, val_, tes_= split_train_val_tes(file_path='./data_224/pile_imgs/*', ratio_=[0.7,0.2,0.1])

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

# all_dataset = SiameseNetworkDataset(file_path=pile_files,
#                                         transform=transformation)

# train_dataset = SiameseNetworkDataset(file_path=train_,
#                                         transform=transformation)
# val_dataset = SiameseNetworkDataset(file_path=val_,
#                                         transform=transformation)

# tes_dataset = SiameseNetworkDataset(file_path=tes_,
#                                         transform=transformation)

# tes_dataloader = DataLoader(tes_dataset,
#                         shuffle=True,
#                         num_workers=16,
#                         batch_size=32)

# all_dataloader = DataLoader(all_dataset,
#                         shuffle=True,
#                         num_workers=1,
#                         batch_size=1)

# # with torch.no_grad():
# # net.eval()
# correct_t = 0
# total_t = 0
# i = 0 
# for img0, img1, label in all_dataloader:
#     # print(i)
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



















