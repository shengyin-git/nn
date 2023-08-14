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

def split_train_val_tes(file_path, num_ = None, ratio_=None):
    pile_files = glob.glob(file_path)
    len_ = len(pile_files)

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

    return pile_files, train_, val_, tes_

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
# train_, val_, tes_ = split_train_val_tes(file_path='./data_224/pile_imgs/*', num_=[1500,150,150])
pile_files, train_, val_, tes_= split_train_val_tes(file_path='./data_224/pile_imgs/*', ratio_=[0.7,0.2,0.1])

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

all_dataset = SiameseNetworkDataset(file_path=pile_files,
                                        transform=transformation)

train_dataset = SiameseNetworkDataset(file_path=train_,
                                        transform=transformation)
val_dataset = SiameseNetworkDataset(file_path=val_,
                                        transform=transformation)

tes_dataset = SiameseNetworkDataset(file_path=tes_,
                                        transform=transformation)

#create the Siamese Neural Network
device = "cuda" if torch.cuda.is_available() else "cpu" 

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

        self.resnet.fc =  nn.Flatten() # Identity()

        # # Setting up the Fully Connected Layers
        self.fc = nn.Linear(num_ftrs_resnet*2, 1)


    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors which are returned
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)

        output12 = torch.cat((output1, output2),1)
        output = self.fc(output12)
        return output

# Load the training dataset
train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=1)

val_dataloader = DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=1)

from torch.nn.modules.loss import BCEWithLogitsLoss
# from torch.optim import lr_scheduler

net = SiameseNetwork().to(device)

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(net.fc.parameters()) #, lr = 1e-6

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
for epoch in range(50):
    running_loss = 0.0
    correct = 0
    total=0
    total_vt=0
    correct_vt = 0
    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):
        # Send the images and labels to CUDA
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
        pred = (torch.sigmoid(output) > 0.5)
        correct += torch.sum(pred==label).item()
        total += label.size(0)

        # Every 100 batches print out the loss
        if i % 100 == 0 :
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()} and accuracy {(100 * correct / total)}\n")

            # with torch.no_grad():
            #     # net.eval()
            # # here i use the same data for both training and validation, however it seems the validation loss is mostly greater than the training loss,
            # # which should be reduced from my understanding because the optimization step means to modify the weights towards reducing the loss, 
            # # and i don't quite understand this part.
            #     output = net(img0, img1)
            #     loss_vt = loss_fn(output, label)
            #     pred = (torch.sigmoid(output) > 0.5)
            #     correct_vt += torch.sum(pred==label).item()
            #     total_vt += label.size(0)
            #     print(f"Epoch number {epoch}\n Current val loss {loss_vt.item()} and accuracy {(100 * correct_vt / total_vt)}\n")       
            # # net.train()
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step) #total_step
    print(f'Training loss: {np.mean(train_loss):.4f}, training acc: {(100 * correct/total):.4f}\n')
    
    # validation
    batch_loss = 0
    total_t=0
    correct_t=0
    # net.eval()
    with torch.no_grad():
        # net.eval()
        for img0_, img1_, label_ in val_dataloader:
            img0_, img1_, label_ = img0_.to(device), img1_.to(device), label_.to(device)
            output_ = net(img0_, img1_)
            loss_t = loss_fn(output_, label_)
            batch_loss += loss_t.item()

            pred_ = (torch.sigmoid(output_) > 0.5)
            correct_t += torch.sum(pred_==label_).item()
            total_t += label_.size(0)

            # print(f"Epoch number {epoch}\n Validation loss {loss_t.item()} and accuracy {(100 * correct_t / total_t)}\n")
    # net.train()
    val_acc.append(100 * correct_t/total_t)
    val_loss.append(batch_loss/ len(val_dataloader))#
    network_learned = batch_loss < valid_loss_min
    print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

    if network_learned:
        valid_loss_min = batch_loss
        torch.save(net.state_dict(), './data_224/my_net_'+ str(ts)+'.pt')
        print('Improvement-Detected, save-model') 
   
# torch.save(net.state_dict(), './data_224/my_net_'+ str(ts)+'.pt')

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

############################################################
tes_dataloader = DataLoader(tes_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=32)

all_dataloader = DataLoader(all_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=32)

# net.eval()
with torch.no_grad():
    correct_t = 0
    total_t = 0
    batch_loss = 0 
    for img0, img1, label in all_dataloader:
      img0, img1, label = img0.to(device), img1.to(device), label.to(device)

      output = net(img0, img1)
      loss_t = loss_fn(output, label)
      batch_loss += loss_t.item()

      pred = (torch.sigmoid(output) > 0.5)
      correct_t += torch.sum(pred==label).item()
      total_t += label.size(0)

      output_ = torch.sigmoid(output)
      # print(output.cpu().numpy().reshape(-1))
    #   print(output_.cpu().numpy().reshape(-1))
    #   print(label.cpu().numpy().reshape(-1)) 

    tes_acc = 100 * correct_t/total_t
    tes_loss = batch_loss/len(all_dataloader) #
    print(f'test loss: {np.mean(tes_loss):.4f}, test acc: {tes_acc:.4f}\n')

################################################################################
## show example test
vis_dataloader = DataLoader(tes_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=8)
example_batch = next(iter(vis_dataloader))

# net.eval()
with torch.no_grad():
    correct_t = 0
    total_t = 0
    batch_loss = 0
    img0, img1, label = example_batch
    img0, img1, label = img0.to(device), img1.to(device), label.to(device)

    output = net(img0, img1)
    loss_t = loss_fn(output, label)
    batch_loss += loss_t.item()

    pred = (torch.sigmoid(output) > 0.5)
    correct_t += torch.sum(pred==label).item()
    total_t += label.size(0)

    output_ = torch.sigmoid(output)
    print(loss_t.item())
    print(output_.cpu().numpy().reshape(-1))
    print(label.cpu().numpy().reshape(-1)) 

    tes_acc = 100 * correct_t/total_t
    tes_loss = batch_loss
    print(f'example test loss: {np.mean(tes_loss):.4f}, test acc: {tes_acc:.4f}\n')

concatenated = torch.cat((example_batch[0], example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))





























