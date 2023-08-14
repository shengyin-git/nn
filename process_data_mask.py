import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import glob
import json

import torchvision
from torchvision import models
from torchvision.models import ResNet101_Weights
import torch
import torch.nn as nn

resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT) 
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Flatten()
for param in resnet.parameters():
            param.requires_grad = False
resnet.eval()

class process_data(object):   
    def __init__(self, pos_path = 'im_pick_pt.npy', \
                        ori_path = 'pick_orn.npy', \
                        image_path = 'rgb_image.jpg', \
                        masks_path = 'pruned_masks_no_back_vit_h_rgb_image.npz', \
                        label_path = 'masks_vit_h_removed_rgb_image.npy', \
                        target_size = [500,500], \
                        save_path = None):
        self.pos_path = pos_path
        self.ori_path = ori_path
        self.image_path = image_path
        self.masks_path = masks_path
        self.label_path = label_path
        self.target_size = target_size
        self.save_path = save_path        

    def process_(self):
        # load picking
        pos = np.load(self.pos_path)
        orientation = np.load(self.ori_path)
        # load original image and label 
        if not os.path.exists(self.label_path):
            return False
        elif os.path.exists(self.image_path):
            image = Image.open(self.image_path)  
            labels = np.load(self.label_path)         
        else:
            image_path_ = np.char.replace(self.image_path, 'jpg', 'png')
            image = Image.open(str(image_path_))
            labels = np.load(self.label_path)  

        self.w, self.h = image.size
        _image = Image.new("RGB", (self.w, self.h), color=(255, 255, 255))

        # get masked pile image
        masks = np.load(self.masks_path, allow_pickle=True)
        masks_min = masks['min']
        num_masks = len(masks_min)

        pile_mask = masks_min[0]['segmentation'] 
        for i in range(num_masks-1):
            pile_mask = pile_mask | masks_min[i+1]['segmentation'] 
        
        masked_pile_images = self.apply_mask(image, pile_mask)
        masked_pile_images_ = self.extend_translate_rotate_cut(ori_img=masked_pile_images, tran=pos, rot=orientation, size_=self.target_size)

        # get masked individual image
        # plt.ion()
        for i in range(num_masks):
            mask = masks_min[i]['segmentation'] 
            masked_image = self.apply_mask(_image, mask)
            masked_image_ = self.extend_translate_rotate_cut(ori_img=masked_image, tran=pos, rot=orientation, size_=self.target_size)

            masked_image_arr = np.asarray(masked_image_)
            if np.any(masked_image_arr):
                # plt.figure(figsize=(10, 10))
                # plt.subplot(1, 3, 1)
                # plt.title('Piled Image')
                # plt.imshow(masked_pile_images_)
                # plt.plot(self.target_size[0]/2, self.target_size[1]/2, "og", markersize=10)
                # plt.axis('off')

                # plt.subplot(1, 3, 2)
                # plt.title('Mask Image')
                # plt.imshow(masked_image_)
                # plt.plot(self.target_size[0]/2, self.target_size[1]/2, "og", markersize=10)
                # plt.axis('off')

                # plt.subplot(1, 3, 3)
                # plt.title('Original Image')
                # plt.imshow(image)
                # plt.plot(pos[1], pos[0], "og", markersize=10)
                # plt.axis('off')

                # plt.show()
                # time.sleep(2)
                # plt.close("all")

                # save paired images
                if self.save_path is not None:
                    ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    rnd_ = np.random.randint(0,10000)

                    pile_img_path = os.path.join(self.save_path,'pile_imgs/')                    
                    if os.path.exists(pile_img_path):
                        masked_pile_images_.save(pile_img_path + str(ts) + str(rnd_) + '.jpg')
                    else:
                        os.makedirs(pile_img_path, exist_ok=True)
                        masked_pile_images_.save(pile_img_path + str(ts) + str(rnd_) + '.jpg')

                    mask_img_path = os.path.join(self.save_path,'mask_imgs/')
                    if os.path.exists(mask_img_path):
                        masked_image_.save(mask_img_path + str(ts) + str(rnd_) + '.jpg')
                    else:
                        os.makedirs(mask_img_path, exist_ok=True)
                        masked_image_.save(mask_img_path + str(ts) + str(rnd_) + '.jpg')

                    label_path = os.path.join(self.save_path,'labels/')
                    if os.path.exists(label_path):
                        np.save(label_path + str(ts) + str(rnd_) + '.npy', labels[i])
                    else:
                        os.makedirs(label_path, exist_ok=True)
                        np.save(label_path + str(ts) + str(rnd_) + '.npy', labels[i])

                    # pile_img_path = os.path.join(self.save_path,'pile_imgs/')                    
                    # if os.path.exists(pile_img_path):
                    #     masked_pile_images_.save(pile_img_path + str(ts) + str(rnd_) + '.jpg')
                    # else:
                    #     os.makedirs(pile_img_path, exist_ok=True)
                    #     masked_pile_images_.save(pile_img_path + str(ts) + str(rnd_) + '.jpg')

                    # mask_img_path = os.path.join(self.save_path,'mask_imgs/')
                    # if os.path.exists(mask_img_path):
                    #     masked_image_.save(mask_img_path + str(ts) + str(rnd_) + '.jpg')
                    # else:
                    #     os.makedirs(mask_img_path, exist_ok=True)
                    #     masked_image_.save(mask_img_path + str(ts) + str(rnd_) + '.jpg')

                    # label_path = self.save_path + 'labels.json'
                    # if os.path.exists(label_path): # label file exist
                    #     with open(label_path, 'r') as file:
                    #         data_ = json.load(file)
                    #     data_[str(ts) + str(rnd_)] = labels[i]
                    #     with open(label_path, 'w') as file:
                    #         json.dump(data_, file, indent=4)
                 
                    # elif os.path.exists(self.save_path): # label file not exist but saving dir exist
                    #     data_ = {str(ts) + str(rnd_): labels[i]}
                    #     with open(label_path, 'w') as file:
                    #         json.dump(data_, file, indent=4)
                    # else: # none exist
                    #     os.makedirs(self.save_path, exist_ok=True)
                    #     data_ = {str(ts) + str(rnd_): labels[i]}
                    #     with open(label_path, 'w') as file:
                    #         json.dump(data_, file, indent=4)

        return True

    def apply_mask(self, image, mask):
        # Step 1: Convert the image and mask to PyTorch tensors
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        
        masked_image_tensor = image_tensor * mask_tensor
        # Step 3: Convert the selected pixels tensor back to a PIL image
        masked_image = TF.to_pil_image(masked_image_tensor)

        return masked_image
    
    def extend_translate_rotate_cut(self, ori_img, tran, rot, size_):
        # extend
        new_w = self.w*3
        new_h = self.h*3
        expanded_image = Image.new("RGB", (new_w, new_h), color=(0, 0, 0))
        x_offset = int((new_w - self.w) / 2)
        y_offset = int((new_h - self.h) / 2)
        expanded_image.paste(ori_img, (x_offset, y_offset))

        # translate
        translate_x = self.w//2-tran[1]
        translate_y = self.h//2-tran[0]

        image_tensor = TF.to_tensor(expanded_image)
        translated_image_tensor = TF.affine(image_tensor, angle=0, translate=(translate_x, translate_y), scale=1, fill=[0,], shear=0)

        # rotate
        degrees_to_rotate = rot/np.pi*180
        if degrees_to_rotate > 180:
            degrees_to_rotate -= 360
        rotation_center = (new_w//2, new_h//2)
        rotated_image_tensor = TF.affine(translated_image_tensor, angle=degrees_to_rotate, translate=(0, 0), scale=1, shear=0, fill=[0,], center = rotation_center)

        # cut
        crop_x1 = rotation_center[0] - size_[0] // 2
        crop_y1 = rotation_center[1] - size_[1] // 2
        crop_x2 = crop_x1 + size_[0]
        crop_y2 = crop_y1 + size_[1]

        cropped_image_tensor = TF.crop(rotated_image_tensor, crop_y1, crop_x1, size_[1], size_[0])

        cropped_image = TF.to_pil_image(cropped_image_tensor)

        return cropped_image

def main():
    # p = process_data(save_path = './data/')
    # p.process_()

    current_dir = os.path.abspath(os.path.join(os.path.dirname('__file__')))

    files_ = glob.glob(current_dir+'/*')
    len_ = len(files_)

    for i in range(len_):
        files__ = glob.glob(files_[0]+'/*')
        trial_files = [fn for fn in files__ if 'trial' in fn]
        len__ = len(trial_files)

        for j in range(len__):
            p = process_data(\
                            pos_path = trial_files[j]+'/im_pick_pt.npy',\
                            ori_path = trial_files[j]+'/pick_orn.npy',\
                            image_path = trial_files[j]+'/rgb_image.jpg',\
                            masks_path = trial_files[j]+'/pruned_masks_no_back_vit_h_rgb_image.npz',\
                            label_path = trial_files[j]+'/masks_vit_h_removed_rgb_image.npy',\
                            target_size = [224,224],\
                            save_path = './data_224/')
            success = p.process_()
            print(success)

if __name__ == "__main__":
    main()