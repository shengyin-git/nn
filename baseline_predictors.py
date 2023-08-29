import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

## test data
# hello_label = int(np.load("./data_224/labels/2023_08_18_16_21_4043.npy", allow_pickle=True))
# hello_mask = np.load("./data_224/mask_features/2023_08_18_16_21_4043.npy", allow_pickle=True)
# hello_mask_img = cv2.imread("./data_224/mask_imgs/2023_08_18_16_21_4043.jpg", cv2.IMREAD_GRAYSCALE)

# print(hello_label)
# print(np.shape(hello_mask))
# print(np.shape(hello_mask_img))

# print(sum(sum(hello_mask_img)))
# print(len(hello_mask_img))

## statistics of all label

def dataFacts():
    label_path = "./data_224/labels/"
    
    label_count = 0
    tot_count = 0
    
    for label in os.listdir(label_path):
        label_val = int(np.load(label_path+label, allow_pickle=True))
        label_count += label_val
        tot_count += 1
        
    # print(label_count)
    # print(tot_count)
   
# dataFacts()

## dumb way of finding the minimum distance from mask to center
def findMinDistDumb(mask, rat=1):
    l = len(mask)
    
    min_dist = l
    
    for i in range(l):
        for j in range(l):
            if mask[i,j] != 0:
                min_dist = min(np.sqrt((l/2-i)**2 + rat*((l/2-j)**2)), min_dist) 
    
    return min_dist

# hello_mask_img = cv2.imread("./data_224/mask_imgs/2023_08_18_16_21_4043.jpg", cv2.IMREAD_GRAYSCALE)
# findMinDistDumb(hello_mask_img,1/2) 

## data statistics
data_path = './data_180_convolved_res50/'
def dataFactsFull(rat=1):
    label_path = data_path + "labels/"
    mask_path = data_path + "mask_imgs/"
    
    l = len(os.listdir(label_path))
    
    # array of distances and labels of the various masks
    label_dist_array = np.zeros([2,l])
    
    removed_dists = []
    remained_dists = []
    
    label_count = 0
    tot_count = 0
    
    for label in os.listdir(label_path):
        file_name = label[0:-4]
        
        label_val = int(np.load(label_path+label, allow_pickle=True))
        #label_count += label_val
        label_dist_array[0,tot_count] = label_val
        
        mask_array = cv2.imread(mask_path + file_name + ".jpg", cv2.IMREAD_GRAYSCALE)
        mask_dist = findMinDistDumb(mask_array, rat)
        label_dist_array[1,tot_count] = mask_dist
        
        if label_val == 0:
            remained_dists.append(mask_dist)
        else:
            removed_dists.append(mask_dist)
            
        tot_count += 1
        
        if tot_count % 500 == 0:
            print(tot_count)
    
    return label_dist_array, removed_dists, remained_dists

label_dist_array, removed_dists, remained_dists = dataFactsFull(12)

print(f'remained dist is {np.size(remained_dists)}')

# Plot the histogram
removed_dists.sort()

plt.hist(removed_dists, bins=100)
plt.show()

## 
def getCDFofArray(array_in):
    array_sorted = sorted(array_in)
    l = len(array_in)
    max_val = math.ceil(array_sorted[l-1])
    
    cdf_out = np.zeros(max_val+1)
    
    curr_cdf = 0
    
    for i in range(l):
        while curr_cdf < math.ceil(array_sorted[i]):
            cdf_out[curr_cdf+1] = i
            curr_cdf += 1
    
    #cdf_out = l-cdf_out
    
    return cdf_out

remained_cdf = getCDFofArray(remained_dists)
removed_cdf = getCDFofArray(removed_dists)

plt.plot(remained_cdf)
plt.plot(removed_cdf)
plt.show()

##
def getEmpiricalBestThreshold(remained_cdf, removed_cdf):
    # remained_cdf = getCDFofArray(remained_dists)
    # removed_cdf = getCDFofArray(removed_dists)
    
    remained_max_val = remained_cdf[len(remained_cdf)-1]
    removed_max_val = removed_cdf[len(removed_cdf)-1]
    
    l_biggest = max(len(remained_cdf), len(removed_cdf))
    
    if len(remained_cdf) > len(removed_cdf):
        to_append = np.zeros(len(remained_cdf)-len(removed_cdf)) + removed_max_val
        removed_cdf = np.concatenate([removed_cdf, to_append])
    elif len(remained_cdf) < len(removed_cdf):
        to_append = np.zeros(len(removed_cdf)-len(remained_cdf)) + remained_max_val
        remained_cdf = np.concatenate([remained_cdf, to_append])
    
    best_err = remained_max_val + removed_max_val
    best_thresh = 0
    
    for i in range(l_biggest):
        err_for_thresh = remained_cdf[i] - removed_cdf[i] + removed_max_val
        if err_for_thresh < best_err:
            best_thresh = i
            best_err = err_for_thresh
    
    tot_score = remained_max_val + removed_max_val
    
    return best_thresh, best_err , tot_score

best_thresh, best_err, tot_score= getEmpiricalBestThreshold(remained_cdf, removed_cdf)

print(f'the best threshold is {best_thresh}')
print(f'the best error is {best_err}')
print(f'the best tot score is {tot_score}')
print(f'the best error score ratio is {best_err/tot_score}')

input()

def testRatios(rats = [0.125, 0.25, 0.5, 1, 2, 4, 8]):
    
    label_path = data_path + "labels/"
    
    total_score = len(os.listdir(label_path))
    
    best_threshes = np.zeros(len(rats))
    best_errs = np.zeros(len(rats))
    
    for i in range(len(rats)):
        label_dist_array, removed_dists, remained_dists = dataFactsFull(rats[i])

        remained_cdf = getCDFofArray(remained_dists)
        removed_cdf = getCDFofArray(removed_dists)
        best_thresh, best_err, tot_score = getEmpiricalBestThreshold(remained_cdf, removed_cdf)
        
        best_threshes[i] = best_thresh
        best_errs[i] = best_err    
    
    return best_threshes, best_errs, total_score
    
    
rats = [0.125, 0.25, 0.5, 1, 2, 4, 8]
best_threshes, best_errs, total_score = testRatios(rats)
plt.figure()
plt.plot(rats, best_errs/total_score)    
plt.show()

# 2196/9271

