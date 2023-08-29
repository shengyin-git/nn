import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

data_path = 'data_180_convolved_res50'
def generateMaskMaps(file_path=data_path):

    label_path = file_path + '/labels/'
    mask_path = file_path + '/mask_imgs/'
    
    masks_tot_ones = np.zeros([180, 180])
    masks_tot_zeros = np.zeros([180, 180])
    
    labels_tot_ones = 0
    labels_tot_zeros = 0
    
    for label in os.listdir(label_path):
        if label[-4:] == '.npy':
            file_name = label[0:-4]
            
            mask = cv2.imread(mask_path + file_name + ".jpg", cv2.IMREAD_GRAYSCALE)
            mask_normed = mask/np.max(np.max(mask))
            label_val = np.load(label_path + label, allow_pickle=True)
            
            if label_val == 0:
                masks_tot_zeros += mask_normed
                labels_tot_zeros += 1
            else:
                masks_tot_ones += mask
                labels_tot_ones += 1
    
        if (labels_tot_ones + labels_tot_zeros)%1000 == 0:
            print(labels_tot_ones + labels_tot_zeros)
    
    masks_avg_ones = masks_tot_ones/labels_tot_ones
    masks_avg_zeros = masks_tot_zeros/labels_tot_zeros
    
    return masks_avg_zeros, masks_avg_ones, labels_tot_zeros, labels_tot_ones

masks_avg_zeros, masks_avg_ones, labels_tot_zeros, labels_tot_ones = generateMaskMaps()

plt.figure()
plt.imshow(masks_avg_zeros)
plt.show()

plt.imshow(masks_avg_ones)
plt.show()

print(labels_tot_zeros)

print(labels_tot_ones)

print(np.min(np.min(masks_avg_zeros)))

print(masks_avg_zeros)


def countUniqueMasks(file_path = data_path):
    label_path = file_path + '/labels/'
    mask_path = file_path + '/mask_imgs/'
    
    masks_ones_set = set()
    masks_zeros_set = set()
    
    #masks_ones_unique = 0
    #masks_zeros_unique = 0
    
    tot_done = 0
    
    for label in os.listdir(label_path):
        if label[-4:] == '.npy':
            file_name = label[0:-4]
            
            mask = cv2.imread(mask_path + file_name + ".jpg", cv2.IMREAD_GRAYSCALE)
            mask_tobytes = np.ndarray.tobytes(mask)
            label_val = np.load(label_path + label, allow_pickle=True)
            
            if label_val == 0:
                masks_zeros_set.add(mask_tobytes)
                #labels_tot_zeros += 1
            else:
                masks_ones_set.add(mask_tobytes)
                #labels_tot_ones += 1
    
        if tot_done%1000 == 0:
            print(tot_done)
            
        tot_done += 1
    
    #masks_avg_ones = masks_tot_ones/labels_tot_ones
    #masks_avg_zeros = masks_tot_zeros/labels_tot_zeros
    
    masks_zeros_unique = len(masks_zeros_set)
    masks_ones_unique = len(masks_ones_set)
    
    return masks_zeros_unique, masks_ones_unique

masks_zeros_unique, masks_ones_unique = countUniqueMasks()

print('www')
print(masks_zeros_unique)
print(masks_ones_unique)

