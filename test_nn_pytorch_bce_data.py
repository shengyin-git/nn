import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageFile
import PIL.ImageOps    
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights
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

# Showing images
def imshow(img, text=None):
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
    # mask_train = copy.deepcopy(pile_train)
    # label_train = copy.deepcopy(pile_train)
    # for i in range(num_train):
    #     mask_train[i] = np.char.replace(mask_train[i], 'pile_imgs', 'mask_imgs')
    #     label_train[i] = np.char.replace(mask_train[i], 'pile_imgs', 'labels')
    #     label_train[i] = np.char.replace(mask_train[i], 'jpg', 'npy')

    pile_files = list(set(pile_files)-set(pile_train))
    pile_val = np.random.choice(pile_files, size=num_val, replace=False)
    # mask_val = copy.deepcopy(pile_val)
    # label_val = copy.deepcopy(pile_val)
    # for i in range(num_val):
    #     mask_val[i] = np.char.replace(mask_val[i], 'pile_imgs', 'mask_imgs')
    #     label_val[i] = np.char.replace(mask_val[i], 'pile_imgs', 'labels')
    #     label_val[i] = np.char.replace(mask_val[i], 'jpg', 'npy')

    pile_files = list(set(pile_files)-set(pile_val))
    pile_tes = np.random.choice(pile_files, size=num_tes, replace=False)
    # mask_tes = copy.deepcopy(pile_tes)
    # label_tes = copy.deepcopy(pile_tes)
    # for i in range(num_tes):
    #     mask_tes[i] = np.char.replace(mask_tes[i], 'pile_imgs', 'mask_imgs')
    #     label_tes[i] = np.char.replace(mask_tes[i], 'pile_imgs', 'labels')
    #     label_tes[i] = np.char.replace(mask_tes[i], 'jpg', 'npy')

    train_ = pile_train
    val_ = pile_val
    tes_ = pile_tes

    return train_, val_, tes_

class SiameseNetworkDataset(Dataset):
    def __init__(self,file_path,transform=None):
        self.transform = transform

        self.pile_files = file_path
        # self.mask_files = file_path[1]
        # self.label_path = label_path

        self.len_ = len(self.pile_files)

    def __getitem__(self,index):
        rnd_pile = random.choice(self.pile_files)
        img_pile = Image.open(rnd_pile)

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
train_, val_, tes_ = split_train_val_tes(file_path='./data/pile_imgs/*', num_=[1500,150,150])
# train_, val_, tes_ = split_train_val_tes(file_path='./data/pile_imgs/*', ratio_=[0.7,0.15,0.15])

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()
                                    ])

train_dataset = SiameseNetworkDataset(file_path=train_,
                                        transform=transformation)
val_dataset = SiameseNetworkDataset(file_path=val_,
                                        transform=transformation)

tes_dataset = SiameseNetworkDataset(file_path=tes_,
                                        transform=transformation)
# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]),0)

# imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy().reshape(-1)) 

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
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT) #.to(device)
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs_resnet = self.resnet.fc.in_features

        self.resnet.fc = nn.Flatten() # Identity()

        # # Setting up the Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs_resnet*2, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,1)
        )

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
                        num_workers=8,
                        batch_size=64)

val_dataloader = DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=64)

from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler

net = SiameseNetwork().to(device)# to(device) cuda()

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(net.fc.parameters()) 

#######################################################################################
## training process 
counter = []
loss_history = [] 
iteration_number= 0

# Iterate throught the epochs
for epoch in range(100):

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

        # Every 10 batches print out the loss
        if i % 10 == 0 :
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

show_plot(counter, loss_history)

################################################################################################
# def make_train_step(model, optimizer, loss_fn):
#   def train_step(x,y):
#     #make prediction
#     yhat = model(x[0],x[1])
#     #enter train mode
#     model.train()
#     #compute loss
#     loss = loss_fn(yhat,y)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     #optimizer.cleargrads()

#     return loss
#   return train_step

# train_step = make_train_step(net, optimizer, loss_fn)

# from tqdm import tqdm

# losses = []
# val_losses = []

# epoch_train_losses = []
# epoch_test_losses = []

# n_epochs = 500
# early_stopping_tolerance = 5
# early_stopping_threshold = 0.001

# for epoch in range(n_epochs):
#   epoch_loss = 0
#   for i ,data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)): #iterate ove batches
#     x1_batch , x2_batch , y_batch = data
#     x1_batch = x1_batch.to(device) #move to gpu
#     x2_batch = x2_batch.to(device)
#     # y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
#     y_batch = y_batch.to(device) #move to gpu


#     loss = train_step((x1_batch,x2_batch), y_batch)
#     epoch_loss += loss/len(train_dataloader)
#     losses.append(loss)
    
#   epoch_train_losses.append(epoch_loss)
#   print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

#   #validation doesnt requires gradient
#   with torch.no_grad():
#     cum_loss = 0
#     for x1_batch , x2_batch, y_batch in train_dataloader:
#       x1_batch = x1_batch.to(device)
#       x2_batch = x2_batch.to(device)
#     #   y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
#       y_batch = y_batch.to(device) #move to gpu

#       #model to eval mode
#       net.eval()

#       yhat = net(x1_batch,x2_batch)
#       val_loss = loss_fn(yhat,y_batch)
#       cum_loss += loss/len(train_dataloader)
#       val_losses.append(val_loss.item())


#     epoch_test_losses.append(cum_loss)
#     print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
    
#     best_loss = min(epoch_test_losses)
    
#     #save best model
#     if cum_loss <= best_loss:
#       best_model_wts = net.state_dict()
    
#     #early stopping
#     early_stopping_counter = 0
#     if cum_loss > best_loss:
#       early_stopping_counter +=1

#     if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
#       print("/nTerminating: early stopping")
#       break #terminate training
    
# #load best model
# # model.load_state_dict(best_model_wts)

############################################################
tes_dataloader = DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=8)

with torch.no_grad():

    for x1_batch , x2_batch, y_batch in tes_dataloader:
      x1_batch = x1_batch.to(device)
      x2_batch = x2_batch.to(device)
    #   y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device) #move to gpu

      #model to eval mode
      net.eval()

      yhat = net(x1_batch,x2_batch)
      yhat_ = torch.sigmoid(yhat)
      print(yhat.cpu().numpy().reshape(-1))
      print(yhat_.cpu().numpy().reshape(-1))
      print(y_batch.cpu().numpy().reshape(-1)) 


























