import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
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
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
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
        output12 = torch.cat((input1, input2),1)
        output = self.fc(output12)
        return output

net = SiameseNetwork()

net.load_state_dict(torch.load('./data_224/my_net_2023_08_14_18_44_04.pt'))

net = net.to(device)

net.eval()

## greedy planning

# step 1: input the original image and preprocess
image_path = './trial_0/rgb_image.jpg'
image = Image.open(str(image_path))
# get the masks
masks = np.load('./trial_0/pruned_masks_no_back_vit_h_rgb_image.npz', allow_pickle=True)
masks_min = masks['min']
num_masks = len(masks_min)

pile_mask = masks_min[0]['segmentation'] 
for i in range(num_masks-1):
    pile_mask = pile_mask | masks_min[i+1]['segmentation'] 

def apply_mask(image, mask):
    # Step 1: Convert the image and mask to PyTorch tensors
    image_tensor = TF.to_tensor(image)
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    
    masked_image_tensor = image_tensor * mask_tensor
    # Step 3: Convert the selected pixels tensor back to a PIL image
    masked_image = TF.to_pil_image(masked_image_tensor)

    return masked_image

masked_pile_images = apply_mask(image, pile_mask)

# step 2: sampling positions and poses, uniformly or bias on masked items
# mask_idx = np.transpose(np.nonzero(pile_mask))
# num_pos_samp = 100
# rand_perm = list(np.arange(len(mask_idx)))
# # rand_perm = list(np.random.permutation(len(mask_idx)))
# rand_pos_idx = random.sample(rand_perm, num_pos_samp)

plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
plt.title('Piled Image')
plt.imshow(masked_pile_images)

for i in range(num_pos_samp):
    coords = mask_idx[rand_pos_idx[i]]
    plt.plot(coords[1], coords[0], "og", markersize=10)

plt.axis('off')

plt.show()





# step 3: evaluate the proposed grasping pose


















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





















