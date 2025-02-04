import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.utils.data as Data
from utils import image_read

class FMB_training(Data.Dataset):
    def __init__(self, file_path=r"data/trainingdata"):
        self.file_path = file_path
        
    def __len__(self):
        return len(os.listdir(self.file_path))
    
    def __getitem__(self, index):
        patch_path=os.path.join(self.file_path,str(index))
        img1=image_read(os.path.join(patch_path,"img1.png"),"GRAY")[None,...]/255.
        img2=image_read(os.path.join(patch_path,"img2.png"),"GRAY")[None,...]/255.
        segmask=image_read(os.path.join(patch_path,"mask.png"),"GRAY").astype(int)
        segmask_onehot  = np.eye(15)[segmask.reshape([-1])].reshape(256,256,15)
        segmask = torch.Tensor(segmask).long()

        return  torch.Tensor(img1),torch.Tensor(img2),segmask,segmask_onehot,index
