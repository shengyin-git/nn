# %matplotlib inline
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

class SiameseNetworkDataset(Dataset):
    def __init__(self,file_path,transform=None):
        self.transform = transform

        files = glob.glob(file_path)
        self.cat_files = [fn for fn in files if 'cat' in fn]
        self.dog_files = [fn for fn in files if 'dog' in fn]

        self.len = len(self.cat_files)
        print(self.len)
        self._index = np.arange(self.len)

    def __getitem__(self,index):
        rnd_0 = random.choice(self.cat_files)
        img0_ = Image.open(rnd_0)

        #We need to approximately 50% of images to be in the same class
        should_get_diff_class = random.randint(0,1) 
        if should_get_diff_class:                
            rnd_1 = random.choice(self.dog_files)
            img1_ = Image.open(rnd_1)
            label_ = 1
        else:
            while True:
                rnd_1 = random.choice(self.cat_files)
                img1_ = Image.open(rnd_1)
                label_ = 0
                if rnd_0 != rnd_1:
                    break

        img0 = img0_
        img1 = img1_

        # img0 = img0_.convert("L")
        # img1 = img1_.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(label_)], dtype=np.float32))

    def __len__(self):
        return self.len

# Load the training dataset
# folder_dataset = datasets.ImageFolder(root="./data/train/*")

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])

# Initialize the network
siamese_dataset = SiameseNetworkDataset(file_path="./data/train/*",
                                        transform=transformation)

# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]),0)

imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy().reshape(-1)) 

#create the Siamese Neural Network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        output12 = self.cat((output1, output2),1)
        output = self.fn(output12)
        return output

# Load the training dataset
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=64)

from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler

net = SiameseNetwork().to(device)#cuda()
# criterion = ContrastiveLoss()
# optimizer = optim.Adam(net.parameters(), lr = 0.0005 )

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(model.fc.parameters()) 

# #train step
# def make_train_step(model, optimizer, loss_fn):
#   def train_step(x,y):
#     #make prediction
#     yhat = model(x)
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
# train_step = make_train_step(model, optimizer, loss_fn)

counter = []
loss_history = [] 
iteration_number= 0

# Iterate throught the epochs
for epoch in range(100):

    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):

        # Send the images and labels to CUDA
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

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

# show_plot(counter, loss_history)

# # Locate the test dataset and load it into the SiameseNetworkDataset
# folder_dataset_test = datasets.ImageFolder(root="./data/faces/testing/")
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                         transform=transformation)
# test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

# # Grab one image that we are going to test
# dataiter = iter(test_dataloader)
# x0, _, _ = next(dataiter)

# for i in range(5):
#     # Iterate over 5 images and test them with the first image (x0)
#     _, x1, label2 = next(dataiter)

#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)
    
#     output1, output2 = net(x0.cuda(), x1.cuda())
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')


























