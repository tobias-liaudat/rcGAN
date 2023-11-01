import glob
import os
import numpy as np

src_train_path = "/share/gpu0/jjwhit/kappa_cosmos_simulations/kappa_train/*.npy"
src_test_path = "/share/gpu0/jjwhit/kappa_cosmos_simulations/kappa_test/*.npy"
src_val_path = "/share/gpu0/jjwhit/kappa_cosmos_simulations/kappa_val/*.npy"
all_training = glob.glob(src_train_path)
all_test = glob.glob(src_test_path)
all_val = glob.glob(src_val_path)

#Define the destination folder
dst_path = "/share/gpu0/jjwhit/kappa_cosmos_simulations/cropped_dataset/"
center_size = 300 #Size of COSMOS MAPS

   
if not os.path.exists(dst_path):
   # Create a new directory because it does not exist
   os.makedirs(dst_path)
   print("The new directory has been made!")

dst_train_path = dst_path + 'kappa_train/'
dst_test_path = dst_path + 'kappa_test/'
dst_val_path = dst_path + 'kappa_val/'

if not os.path.exists(dst_train_path):
    os.makedirs(dst_train_path)
if not os.path.exists(dst_test_path):
    os.makedirs(dst_test_path)
if not os.path.exists(dst_val_path):
    os.makedirs(dst_val_path)



img_number = 1
#Takes kappa.npy files and crops them - test
for fname in all_test:
    print('Processing test file n', img_number)
    #Load file
    kappa = np.load(fname, allow_pickle=True)
    kappa[kappa>0.7]=0.7
    #Crop convergence map
    center_start = (1024 - center_size) // 2
    center_end =  center_start + center_size

    kappa_cropped = kappa[center_start:center_end,  center_start:center_end]
    #Save cropped file
    save_path = '{:s}{:s}{:05d}{:s}'.format(dst_test_path, "cropped_sim_", img_number, ".npy")
    
    np.save(save_path, kappa_cropped, allow_pickle=True)
    
    img_number +=1


img_number = 1
#Takes kappa.npy files and crops them - train
for fname in all_training:
    print('Processing file n', img_number)
    kappa = np.load(fname, allow_pickle=True)
    kappa[kappa>0.7]=0.7

    center_start = (1024 - center_size) // 2
    center_end =  center_start + center_size
    kappa_cropped = kappa[center_start:center_end,  center_start:center_end]
    save_path = '{:s}{:s}{:05d}{:s}'.format(dst_train_path, "kappa_run_", img_number, ".npy")
    
    np.save(save_path, kappa_cropped, allow_pickle=True)
    
    img_number +=1


img_num = 1
#Takes kappa.npy files and crops them - val
for fname in all_val:
    print('Processing file n', img_number)
    kappa = np.load(fname, allow_pickle=True)
    kappa[kappa>0.7]=0.7

    center_start = (1024 - center_size) // 2
    center_end =  center_start + center_size
    kappa_cropped = kappa[center_start:center_end,  center_start:center_end]
    save_path = '{:s}{:s}{:05d}{:s}'.format(dst_val_path, "kappa_run_", img_number, ".npy")
    
    np.save(save_path, kappa_cropped, allow_pickle=True)
    
    img_number +=1

