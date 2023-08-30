import glob
import numpy as np
import os
import copy
import shutil

def split_train_val_tes(file_path, num_ = None, ratio_=None):
    files_ = glob.glob(file_path)
    len_ = len(files_)
    _files = copy.deepcopy(files_)

    # for i in range(len(_files)):
    #     os.makedirs('./all', exist_ok=True)
    #     os.makedirs('./all/labels', exist_ok=True)
    #     os.makedirs('./all/mask_imgs', exist_ok=True)
    #     os.makedirs('./all/mask_features', exist_ok=True)
    #     os.makedirs('./all/pile_imgs', exist_ok=True)
    #     os.makedirs('./all/pile_features', exist_ok=True)
    #     # os.rename(_files[i], './all/'+_files[i])
    #     shutil.copyfile(_files[i], './all/'+_files[i])

    if num_ is not None:
        num_train, num_val, num_tes = num_
    else:
        num_train = np.int32(len_*ratio_[0])
        num_val = min(np.int32(len_*ratio_[1]), len_-num_train)
        num_tes = min(np.int32(len_*ratio_[2]), len_-num_train-num_val)

    train_ = np.random.choice(files_, size=num_train, replace=False)    

    files_ = list(set(files_)-set(train_))
    val_ = np.random.choice(files_, size=num_val, replace=False)

    files_ = list(set(files_)-set(val_))
    tes_ = np.random.choice(files_, size=num_tes, replace=False)


    for i in range(len(train_)):
        label_path = train_[i]
        os.makedirs('./train', exist_ok=True)
        os.makedirs('./train/labels', exist_ok=True)
        os.makedirs('./train/mask_imgs', exist_ok=True)
        os.makedirs('./train/mask_features', exist_ok=True)
        os.makedirs('./train/pile_imgs', exist_ok=True)
        os.makedirs('./train/pile_features', exist_ok=True)
        # os.rename(_files[i], './all/'+_files[i])
        shutil.copyfile(label_path, './train/'+label_path)

        pile_feature_path = str(np.char.replace(label_path, 'labels', 'pile_features'))
        shutil.copyfile(pile_feature_path, './train/'+pile_feature_path)
        pile_img_path = str(np.char.replace(label_path, 'labels', 'pile_imgs'))
        pile_img_path = str(np.char.replace(pile_img_path, 'npy', 'jpg'))
        shutil.copyfile(pile_img_path, './train/'+pile_img_path)

        mask_feature_path = str(np.char.replace(label_path, 'labels', 'mask_features'))
        shutil.copyfile(mask_feature_path, './train/'+mask_feature_path)
        mask_img_path = str(np.char.replace(label_path, 'labels', 'mask_imgs'))
        mask_img_path = str(np.char.replace(mask_img_path, 'npy', 'jpg'))
        shutil.copyfile(mask_img_path, './train/'+mask_img_path)

    for i in range(len(val_)):
        label_path = val_[i]
        os.makedirs('./val', exist_ok=True)
        os.makedirs('./val/labels', exist_ok=True)
        os.makedirs('./val/mask_imgs', exist_ok=True)
        os.makedirs('./val/mask_features', exist_ok=True)
        os.makedirs('./val/pile_imgs', exist_ok=True)
        os.makedirs('./val/pile_features', exist_ok=True)
        # os.rename(_files[i], './all/'+_files[i])
        shutil.copyfile(label_path, './val/'+label_path)

        pile_feature_path = str(np.char.replace(label_path, 'labels', 'pile_features'))
        shutil.copyfile(pile_feature_path, './val/'+pile_feature_path)
        pile_img_path = str(np.char.replace(label_path, 'labels', 'pile_imgs'))
        pile_img_path = str(np.char.replace(pile_img_path, 'npy', 'jpg'))
        shutil.copyfile(pile_img_path, './val/'+pile_img_path)

        mask_feature_path = str(np.char.replace(label_path, 'labels', 'mask_features'))
        shutil.copyfile(mask_feature_path, './val/'+mask_feature_path)
        mask_img_path = str(np.char.replace(label_path, 'labels', 'mask_imgs'))
        mask_img_path = str(np.char.replace(mask_img_path, 'npy', 'jpg'))
        shutil.copyfile(mask_img_path, './val/'+mask_img_path)

    for i in range(len(tes_)):
        label_path = tes_[i]
        os.makedirs('./tes', exist_ok=True)
        os.makedirs('./tes/labels', exist_ok=True)
        os.makedirs('./tes/mask_imgs', exist_ok=True)
        os.makedirs('./tes/mask_features', exist_ok=True)
        os.makedirs('./tes/pile_imgs', exist_ok=True)
        os.makedirs('./tes/pile_features', exist_ok=True)
        # os.rename(_files[i], './all/'+_files[i])
        shutil.copyfile(label_path, './tes/'+label_path)

        pile_feature_path = str(np.char.replace(label_path, 'labels', 'pile_features'))
        shutil.copyfile(pile_feature_path, './tes/'+pile_feature_path)
        pile_img_path = str(np.char.replace(label_path, 'labels', 'pile_imgs'))
        pile_img_path = str(np.char.replace(pile_img_path, 'npy', 'jpg'))
        shutil.copyfile(pile_img_path, './tes/'+pile_img_path)

        mask_feature_path = str(np.char.replace(label_path, 'labels', 'mask_features'))
        shutil.copyfile(mask_feature_path, './tes/'+mask_feature_path)
        mask_img_path = str(np.char.replace(label_path, 'labels', 'mask_imgs'))
        mask_img_path = str(np.char.replace(mask_img_path, 'npy', 'jpg'))
        shutil.copyfile(mask_img_path, './tes/'+mask_img_path)

    return _files, train_, val_, tes_


# Load the training dataset
pile_files, train_, val_, tes_= split_train_val_tes(file_path='./labels/*', ratio_=[0.8,0.2,0.0])