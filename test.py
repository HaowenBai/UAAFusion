import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from nets.SS_Net import SS_Net
from nets.UAAFusion import UAAFusion
from utils import *

path_ir = r"test_cases/ir"
path_vi = r"test_cases/vi"
path_result = r"test_result"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_f = UAAFusion(5,5).to(device)
model_s = SS_Net(14+1).to(device)

model_f.load_state_dict(torch.load(r"models/UAAFusion.pth")['model_f'])
model_s.load_state_dict(torch.load(r"models/UAAFusion.pth")['model_s'])
model_f.eval()
model_s.eval()

with torch.no_grad():
    for imgname in tqdm(os.listdir(path_ir)):
        img1=image_read(os.path.join(path_ir, imgname))
        img2=image_read(os.path.join(path_vi, imgname))

        img1_YCrCb=cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
        img2_YCrCb=cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)

        if is_grayscale(img1):
            if is_grayscale(img2):
                CrCb=None
            else:
                CrCb=img2_YCrCb[:,:,1:]
        else:
            if is_grayscale(img2): 
                CrCb=img1_YCrCb[:,:,1:]
            else:
                CrCb=fuse_CrCb(img1_YCrCb[:,:,1:],img2_YCrCb[:,:,1:])

        img1=img1_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
        img2=img2_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
        img1 = ((torch.FloatTensor(img1))).to(device)
        img2 = ((torch.FloatTensor(img2))).to(device)

        data_Fuse,_,_,_,_=model_f(img1,img2,model_s)
        data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
        fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
        img_save(fused_image, imgname, path_result, CrCb)

        

