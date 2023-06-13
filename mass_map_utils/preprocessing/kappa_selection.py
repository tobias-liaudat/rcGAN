import glob
import os
import numpy as np


#Define the source folder
#kappa2 - z = 0.858
src_path = "/disk/xray99/jdm/kappaTNG/kappaTNG-Hydro/LP*/run*/kappa20.dat"
all_files = glob.glob(src_path)

#Define the destination folder
dst_path = os.path.join("/disk/xray99/tl3/mass_map_dataset/kappa_selection/kappa20/")


   
if not os.path.exists(dst_path):
   # Create a new directory because it does not exist
   os.makedirs(dst_path)
   print("The new directory has been made!")


img_number = 1
ng = 1024

for fname in all_files:
    print('Processing file n', img_number)

    with open(fname, 'rb') as f:


        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)

        kappa = kappa.reshape((ng,ng))
    
        save_path = '{:s}{:s}{:05d}{:s}'.format(dst_path, "kappa_run_", img_number, ".npy")
        np.save(save_path, kappa, allow_pickle=True)
        
        img_number +=1
