import numpy as np
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

resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT) 
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Flatten()
for param in resnet.parameters():
            param.requires_grad = False
resnet.eval()

transformation = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])

files = glob.glob('./pile_imgs/*')
num_files = len(files)

for i in range(num_files):
    img_path = files[i]
    img = Image.open(img_path)
    img= transformation(img)

    img_feature = resnet(img.unsqueeze(0))

    img_array = img_feature.cpu().numpy().reshape(-1)

    array_path = str(np.char.replace(img_path, 'jpg', 'npy'))

    np.save(array_path, img_array)

