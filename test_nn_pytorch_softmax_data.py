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

# Showing images
def imshow(img, text=None):
    plt.figure()
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

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

class SiameseNetworkDataset(Dataset):
    def __init__(self,file_path):
        self.label_files = file_path
        self.len_ = len(self.label_files)

    def __getitem__(self,idx):
        rnd_label = self.label_files[idx]
        label = np.load(rnd_label)     
        ##########################################
        # rnd_pile = random.choice(self.pile_files)
        # img_pile = Image.open(rnd_pile)
        ##########################################

        rnd_pile = np.char.replace(rnd_label, 'labels', 'pile_features')
        img_pile = np.load(str(rnd_pile))

        rnd_mask = np.char.replace(rnd_label, 'labels', 'mask_features')
        img_mask = np.load(str(rnd_mask))

        # torch.from_numpy(np.array(label_, dtype=np.int32).squeeze())
        return torch.from_numpy(np.array(img_pile).squeeze()), \
            torch.from_numpy(np.array(img_mask).squeeze()), \
                torch.from_numpy(np.array([int(label)], dtype=np.int32).squeeze())

    def __len__(self):
        return self.len_

# Load the training dataset
data_path = './data_224_res101/'
# pile_files, train_, val_, tes_= split_train_val_tes(file_path=data_path+'labels/*', ratio_=[0.8,0.2,0.0])

files_ = glob.glob(data_path+'labels/*')
train_ = glob.glob(data_path+'train/labels/*')
val_ = glob.glob(data_path+'val/labels/*')
tes_ = glob.glob(data_path+'tes/labels/*')

all_dataset = SiameseNetworkDataset(file_path=files_)
train_dataset = SiameseNetworkDataset(file_path=train_)
val_dataset = SiameseNetworkDataset(file_path=val_)
tes_dataset = SiameseNetworkDataset(file_path=tes_)

#create the Siamese Neural Network
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT) 
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Flatten()
for param in resnet.parameters():
            param.requires_grad = False
resnet.eval()

device = "cuda" if torch.cuda.is_available() else "cpu" 

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.fc = nn.Linear(num_ftrs_resnet*2, 2)

        # self.fc = nn.Sequential(
        #     nn.Linear(num_ftrs_resnet*2, 1024),
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(inplace=True),            
            
        #     nn.Linear(1024, 512),
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(512, 256),
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(256,1))

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors which are returned
        output12 = torch.cat((input1, input2),1)
        output = self.fc(output12) 

        return output

# Load the training dataset
train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=64)

val_dataloader = DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=64)

# from torch.optim import lr_scheduler

net = SiameseNetwork().to(device)

#loss
loss_fn = nn.CrossEntropyLoss() #torch.nn.MSELoss()
#optimizer
# optimizer = torch.optim.Adam(net.fc.parameters(), lr = 1e-7)
optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-3, momentum=0.8)

#######################################################################################
## training process 
iteration_number= 0
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)

# Iterate throught the epochs
for epoch in range(500):
    running_loss = 0.0
    correct = 0
    total=0
    total_vt=0
    correct_vt = 0
    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):
        # Send the images and labels to CUDA
        label = label.type(torch.LongTensor)
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = loss_fn(output, label)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        # Accuracy
        running_loss += loss_contrastive.item()
        _, pred = output.max(1)
        correct += torch.sum(pred==label).item()
        total += label.size(0)

        # Every 10 batches print out the loss
        if i % 100 == 0 :
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()} and accuracy {(100 * correct / total)}\n")

        # with torch.no_grad():
        #     net.eval()
        #     # here i use the same data for both training and validation, however it seems the validation loss is mostly greater than the training loss,
        #     # which should be reduced from my understanding because the optimization step means to modify the weights towards reducing the loss, 
        #     # and i don't quite understand this part.

        #     output = net(img0, img1)
        #     loss_t = loss_fn(output, label)
        #     _, pred = output.max(1)
        #     correct_ += torch.sum(pred==label).item()

        #     if i % 10 == 0 :
        #         print(f"Epoch number {epoch}\n Current val loss {loss_t.item()} and accuracy {(100 * correct_ / total)}\n")       

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'Training loss: {np.mean(train_loss):.4f}, training acc: {(100 * correct/total):.4f}\n')
    # validation
    batch_loss = 0
    total_t=0
    correct_t=0
    net.eval()
    with torch.no_grad():
        for img0, img1, label in val_dataloader:
            label = label.type(torch.LongTensor)
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output = net(img0, img1)
            loss_t = loss_fn(output, label)
            batch_loss += loss_t.item()

            _, pred = output.max(1)
            correct_t += torch.sum(pred==label).item()
            total_t += label.size(0)

            # print(f"Epoch number {epoch}\n Validation loss {loss_t.item()} and accuracy {(100 * correct_t / total_t)}\n")
    net.train()
    val_acc.append(100 * correct_t/total_t)
    val_loss.append(batch_loss/ len(val_dataloader))#
    network_learned = batch_loss < valid_loss_min
    print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

    if network_learned:
        valid_loss_min = batch_loss
        torch.save(net.state_dict(), data_path+'/my_net_'+ str(ts)+'.pt')
        print('Improvement-Detected, save-model')   

fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.savefig('./results/training_val_accuracy_' + str(ts) + '.png')

fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Loss")
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')
plt.savefig('./results/training_val_loss_' + str(ts) + '.png')





















