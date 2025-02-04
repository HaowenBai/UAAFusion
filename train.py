# -*- coding: utf-8 -*-
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import time
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)
from torch.utils.data import DataLoader


from nets.SS_Net import SS_Net
from nets.UAAFusion import UAAFusion
from utils import CE_Loss, grad
from data.FMB_training import FMB_training

num_epochs = 40
lr = 1e-4
step_size=10
gamma=0.5
weight_decay = 0
batch_size=8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']


blocknom=5
attnum=5
coeff_grad=1
coeff_seg=0.1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_f=UAAFusion(blocknom,attnum).to(device)
model_s=SS_Net(14+1).to(device)
model_s.load_state_dict(torch.load(r"models/SS_pretrain.pth"))

optimizer_f = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=step_size, gamma=gamma)

optimizer_s = torch.optim.Adam(model_s.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=step_size, gamma=gamma)

trainloader = DataLoader(FMB_training(),batch_size=batch_size, shuffle=True, num_workers=0)

path_exp=os.path.join("exp",time.strftime("%m_%d_%H_%M", time.localtime()))
os.makedirs(path_exp,exist_ok=True)
os.makedirs(os.path.join(path_exp,"model"),exist_ok=True)


# Train

for epoch in range(num_epochs):
    ''' train '''
    model_s.train()
    model_f.train()
    for i, (img_i, img_v, mask_s, mask_oh,index) in enumerate(trainloader):
        img_i, img_v, mask_s,mask_oh = img_i.cuda(), img_v.cuda(), mask_s.cuda(),mask_oh.cuda()
        optimizer_f.zero_grad()
        optimizer_s.zero_grad()

        output_f,w_i,w_v,map_i_list,map_v_list=model_f(img_i,img_v,model_s,mask_oh.permute(0, 3, 1, 2)) 
        
        output_s=model_s(output_f)
        output_s_i=model_s(img_i)
        output_s_v=model_s(img_v)

        loss_int_i=F.mse_loss(w_i*(img_i-output_f),torch.zeros_like(img_i))
        loss_int_v=F.mse_loss(w_v*(img_v-output_f),torch.zeros_like(img_i))
        loss_grad=F.l1_loss(grad(output_f),torch.max(grad(img_i),grad(img_v)))
        loss_seg=(CE_Loss(output_s, mask_s)+CE_Loss(output_s_i, mask_s)+CE_Loss(output_s_v, mask_s))/3

        loss_total=loss_int_i+loss_int_v+coeff_grad*loss_grad+coeff_seg*loss_seg

        loss_total.backward()
        optimizer_f.step()
        optimizer_s.step()

        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        print(
            "[Epoch %d/%d] [Batch %d/%d] [loss_total: %f] [loss_I: %f] [loss_V: %f] [loss_grad: %f] [loss_seg: %f]"
            % (
                epoch+1,
                num_epochs,
                i,
                len(trainloader),
                loss_total.item(),
                loss_int_i.item(),
                loss_int_v.item(),
                loss_grad.item(),
                loss_seg.item(),
            )
        )
                
    scheduler_f.step()  
    scheduler_s.step()

    checkpoint = {'model_f': model_f.state_dict(),
        'model_s': model_s.state_dict(),}
    
    torch.save(checkpoint, os.path.join(path_exp,"model", 'ckpt_%s.pth' % (str(epoch+1))))



            