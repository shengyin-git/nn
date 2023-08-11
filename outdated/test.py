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
        return img_pile, img_mask, torch.from_numpy(np.array(label_, dtype=np.int32).squeeze())

    def __len__(self):
        return self.len_

# Load the training dataset
# Resize the images and transform to tensors
# train_, val_, tes_ = split_train_val_tes(file_path='./data_224/pile_imgs/*', num_=[1500,150,150])
train_, val_, tes_ = split_train_val_tes(file_path='./data_224/pile_imgs/*', ratio_=[0.7,0.2,0.1])

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

train_dataset = SiameseNetworkDataset(file_path=train_,
                                        transform=transformation)
val_dataset = SiameseNetworkDataset(file_path=val_,
                                        transform=transformation)

tes_dataset = SiameseNetworkDataset(file_path=tes_,
                                        transform=transformation)
# Create a simple dataloader just for simple visualization
# vis_dataloader = DataLoader(train_dataset,
#                         shuffle=True,
#                         num_workers=1,
#                         batch_size=8)

# # Extract one batch
# example_batch = next(iter(vis_dataloader))

# # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# # If the label is 1, it means that it is not the same person, label is 0, same person in both images
# concatenated = torch.cat((example_batch[0], example_batch[1]),0)

# print(example_batch[2].numpy().reshape(-1)) 
# imshow(torchvision.utils.make_grid(concatenated))
# input()

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
        self.fc = nn.Linear(num_ftrs_resnet*2, 2)
        # nn.Sequential(
        #     nn.Linear(num_ftrs_resnet*2, 2),
        #     # nn.ReLU(), #inplace=True
        #     # nn.Dropout(p=0.3),
            
        #     # # nn.Linear(1024, 512),
        #     # # nn.ReLU(inplace=True),

        #     # # nn.Linear(512, 256),
        #     # # nn.ReLU(inplace=True),
            
        #     # nn.Linear(1024,1)
        # )

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors which are returned
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)

        output12 = torch.cat((output1, output2),1)
        output = self.fc(output12)
        return output
    
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# Load the training dataset
trainloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=64)

testloader = DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=64)

net = SiameseNetwork().to(device)# to(device) cuda()
# net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainloader)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total=0

    for i, data in enumerate(trainloader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        img0, img1, labels = data
        labels = labels.type(torch.LongTensor)
        img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(img0, img1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 0 : #if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'Training [{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f} and accuracy {(100 * correct / total)}')

            with torch.no_grad():
                net.eval()
                outputs = net(img0, img1)
                loss_ = criterion(outputs, labels)
                print(f'Validation [{epoch + 1}, {i + 1:5d}] loss: {(loss_.item()-loss.item()):.3f}\n')

        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
    train_loss.append(running_loss/total_step)
    train_acc.append(100 * correct / total)
    print(f'Epoch number {epoch}\n Training loss: {np.mean(train_loss):.4f} and accuracy {(100 * correct / total)}\n')

    # validation
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(trainloader, 0):
            img0, img1, labels = data
            labels = labels.type(torch.LongTensor)
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

            outputs = net(img0, img1)
            _, predicted = torch.max(outputs.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted == labels).sum().item()

            loss_t = criterion(outputs, labels)
            batch_loss += loss_t.item()

        val_loss.append(batch_loss/ len(trainloader))
        val_acc.append(100 * correct_t/total_t)
        print(f'Epoch number {epoch}\n Validation loss: {np.mean(val_loss):.4f} and accuracy {(100 * correct_t / total_t)}\n')

print('Finished Training')

fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
# plt.savefig('./results/training_val_accuracy_' + str(ts) + '.png')

fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Loss")
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')
# plt.savefig('./results/training_val_loss_' + str(ts) + '.png')

plt.show()
