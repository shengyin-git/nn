import os
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

current_dir = os.path.abspath(os.path.join(os.path.dirname('__file__')))
data_dir = current_dir + '/method_random_full'
sample_files = glob.glob(data_dir+'/*')
len_sample = len(sample_files)

def apply_mask(image, mask):
    # Step 1: Convert the image and mask to PyTorch tensors
    image_tensor = TF.to_tensor(image)
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    
    masked_image_tensor = image_tensor * mask_tensor
    # Step 3: Convert the selected pixels tensor back to a PIL image
    masked_image = TF.to_pil_image(masked_image_tensor)

    return masked_image

sample_i = 0
checked_sample = 0
while checked_sample < len_sample-1:
    sample_file = data_dir + '/sample_' + str(sample_i) #sample_files[i]
    print(sample_file)
    input()
    sample_i += 1
    if sample_file in sample_files:
        checked_sample += 1
        temp_trial_files = glob.glob(sample_file+'/*')
        trial_files = [fn for fn in temp_trial_files if 'trial' in fn]
        len_trial = len(trial_files)

        for j in range(len_trial-2):
            before_file = sample_file + '/trial_' + str(j)
            print(before_file)
            before_label_path = before_file+'/masks_vit_h_removed_rgb_image_convolved.npy'
            before_img_path = before_file + '/rgb_image.jpg'
            before_pos = np.load(before_file+'/im_pick_pt.npy')
            if not os.path.exists(before_img_path):
                before_img_path = before_file + '/rgb_image.png'

            after_file = sample_file + '/trial_' + str(j+1)
            after_label_path = after_file+'/masks_vit_h_removed_rgb_image_convolved.npy'
            after_img_path = after_file + '/rgb_image.jpg'
            if not os.path.exists(after_img_path):
                after_img_path = after_file + '/rgb_image.png'
            
            ## load two images
            if not before_file in trial_files:
                continue
            elif not os.path.exists(before_label_path):
                continue
            else:
                before_img = Image.open(before_img_path)
                after_img = Image.open(after_img_path)      

            ## load masks
            before_mask_path = before_file + '/pruned_masks_no_back_vit_h_rgb_image_convolved.npz'
            masks = np.load(before_mask_path, allow_pickle=True)
            masks_min = masks['min']
            num_masks = len(masks_min)

            if num_masks == 0:
                print(before_mask_path)
                continue

            labels = np.zeros(num_masks)

            for k in range(num_masks):
                mask = masks_min[k]['segmentation'] 
                masked_img = apply_mask(before_img, mask)

                plt.figure(figsize=(10, 10))
                plt.subplot(2, 2, 1)
                plt.imshow(before_img)
                plt.plot(before_pos[1], before_pos[0], "og", markersize=10)
                plt.axis('off')

                plt.subplot(2, 2, 2)
                plt.imshow(after_img)
                plt.plot(before_pos[1], before_pos[0], "og", markersize=10)
                plt.axis('off')

                plt.subplot(2, 2, 3)
                plt.imshow(masked_img)
                plt.plot(before_pos[1], before_pos[0], "og", markersize=10)
                plt.axis('off')

                plt.subplot(2, 2, 4)
                plt.imshow(mask)
                plt.plot(before_pos[1], before_pos[0], "og", markersize=10)
                plt.axis('off')

                plt.show()
                plt.close("all")

                flag = input('y or n? y for removed, n for not removed, and enter for not sre.')
                if flag == 'y':
                    labels[k] = 1
                elif flag == 'n':
                    labels[k] = 0
                else:
                    labels[k] = 0.5

            label_path = before_file + '/human_label.npy'
            np.save(label_path, labels)
    else:
        continue

    