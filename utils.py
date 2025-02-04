import numpy as np
import cv2
import os
from skimage.io import imsave
import torch.nn.functional as F
import torch
import torch.nn as nn

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB','GRAY','YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image, imagename, savepath, CrCb=None):
    temp = np.squeeze(image)
    path1 = os.path.join(savepath, 'RGB')
    path2 = os.path.join(savepath, 'Gray')
    if not os.path.exists(path2):
        os.makedirs(path2)
    imsave(os.path.join(path2, imagename), temp) # If error, make sure imageio==2.26.0 is installed.

    if CrCb is not None:
        assert len(CrCb.shape) == 3 and CrCb.shape[2] == 2, "CrCb error"
        temp_RGB = cv2.cvtColor(np.concatenate((temp[..., np.newaxis], CrCb), axis=2), cv2.COLOR_YCrCb2RGB)
        if not os.path.exists(path1):
            os.makedirs(path1)
        temp_RGB[temp_RGB<0]=0
        temp_RGB[temp_RGB>255]=255
        imsave(os.path.join(path1, imagename), temp_RGB)

def fuse_CrCb(CrCb1,CrCb2):
    assert len(CrCb1.shape) == 3 and CrCb1.shape[2] == 2, "CrCb error"
    assert len(CrCb2.shape) == 3 and CrCb2.shape[2] == 2, "CrCb error"
    Cf=(CrCb1*np.abs(CrCb1-0.5)+CrCb2*np.abs(CrCb2-0.5))/(np.abs(CrCb1-0.5)+np.abs(CrCb2-0.5)+1e-4)
    return Cf

def is_grayscale(image):
    return np.all(image[:,:,0] == image[:,:,1]) and np.all(image[:,:,1] == image[:,:,2])

def grad(input):
    weightx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    weighty = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    weightx = weightx.repeat(1, input.shape[1], 1, 1).to(input.device)
    weighty = weighty.repeat(1, input.shape[1], 1, 1).to(input.device)

    sobelx=F.conv2d(input, weightx, padding=1)
    sobely=F.conv2d(input, weighty, padding=1)
    return torch.abs(sobelx)+torch.abs(sobely)

def CE_Loss(inputs, target, cls_weights=torch.ones([15]), num_classes=0):
    cls_weights=cls_weights.cuda()
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss