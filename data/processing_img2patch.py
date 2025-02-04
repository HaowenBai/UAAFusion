import sys
import os
sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from utils import image_read

def image_save(path,img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image/255., [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

path_ir=...
path_vi=...
path_mask=...


patchsize=256   
stride=256    
path_save=r"data/trainingdata"

patch_num=0
file_name_list=os.listdir(path_ir) 
for img_name in tqdm(file_name_list):
    img1=image_read(os.path.join(path_ir,img_name),mode='GRAY') 
    img2=image_read(os.path.join(path_vi,img_name),mode='GRAY')
    mask=image_read(os.path.join(path_mask,img_name),mode='GRAY')


    H,W=img1.shape
    p_H_num=(H-patchsize)//stride + 1
    p_W_num=(W-patchsize)//stride + 1

    for k in range(p_H_num*p_W_num):
        a0=k//p_W_num 
        a1=k-a0*p_W_num 

        img1_patch=img1[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize]
        img2_patch=img2[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize]
        mask_patch=mask[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize]
        if not (is_low_contrast(img1_patch) or is_low_contrast(img2_patch)):
            os.makedirs(os.path.join(path_save,str(patch_num)),exist_ok=True)

            image_save(os.path.join(path_save,str(patch_num),"img1.png"), img1_patch)
            image_save(os.path.join(path_save,str(patch_num),"img2.png"), img2_patch)
            image_save(os.path.join(path_save,str(patch_num),"mask.png"), mask_patch)

            patch_num+=1

print('training set, # samples %d\n' % (patch_num))     