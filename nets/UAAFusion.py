import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvUnit(nn.Module):
    def __init__(self,in_ch,mid_ch,out_ch,kernel_size=3):
        super(ConvUnit, self).__init__()
        p = kernel_size//2
        self.conv_unit = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=kernel_size, stride=1, padding=p,bias=True,padding_mode="reflect"),
                nn.PReLU(),
                nn.Conv2d(mid_ch, out_ch, kernel_size=kernel_size, stride=1, padding=p, bias=True,padding_mode="reflect"), 
            )
    def forward(self, input):
        return self.conv_unit(input)   
    
class AUIF_prox(nn.Module):
    def __init__(self):
        super(AUIF_prox, self).__init__()

        self.conv1=ConvUnit(1,32,32)
        self.conv2=ConvUnit(96,32,32)
        self.conv3=ConvUnit(32,32,32)
        self.conv4=nn.Sequential(ConvUnit(32,16,1),nn.Sigmoid())
        self.lstm=ConvLSTM(32,32)
        self.conv5=ConvUnit(96,32,32)
        self.conv6=ConvUnit(32,32,32)

    def forward(self, f,att,h,c,l_m,s_m):
        f=self.conv1(f) 
        l_m=self.conv2(torch.cat((f,l_m,s_m),dim=1))
        f_out=self.conv3(l_m)
        s_m=l_m+f_out*F.sigmoid(att)
        f_out=self.conv4(s_m)
        # memory
        h,c=self.lstm(self.conv5(torch.cat((f,l_m,s_m),dim=1)),h,c)
        l_m=self.conv6(h)
        return f_out,h,c,l_m,s_m
           
class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel=3):
        
        super().__init__()
        pad_x = 1
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,padding_mode="reflect")
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,padding_mode="reflect")
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,padding_mode="reflect")
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,padding_mode="reflect")

        pad_h = 1
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,padding_mode="reflect")
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,padding_mode="reflect")
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,padding_mode="reflect")
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,padding_mode="reflect")

    def forward(self, x, h, c):      
        
        if h is None and c is None:
            i = torch.sigmoid(self.conv_xi(x))
            o = torch.sigmoid(self.conv_xo(x))
            j = torch.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = torch.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = torch.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * torch.tanh(c)
    
        return h, c        
    
class UAAFusion(nn.Module):
    def __init__(self,block_num=5,attnum=10):
        super(UAAFusion, self).__init__()

        self.blockflow = nn.ModuleList([AUIF_prox() for i in range(block_num)])    
        self.alpha = nn.Parameter(0.001*torch.ones(block_num))

        self.attnum=attnum
        self.init_conv_ir=ConvUnit(1,16,16)
        self.init_conv_vi=ConvUnit(1,16,16)
        self.init_conv_f0= nn.Sequential(ConvUnit(32,32,1),nn.Sigmoid())
        self.alpha = nn.Parameter(0.001*torch.ones(block_num))
    
    def forward(self,i,v,Segmodel,mask=None):
        [h, c] = [None, None]
        l_m=torch.cat((self.init_conv_ir(i),self.init_conv_vi(v)),dim=1)
        s_m=l_m
        f=(i+v)/2                
        f_list=[]
        att=torch.zeros_like(f)
        
        if mask is None:
            mask=Segmodel(f)
        w_i,w_v,map_i_list,map_v_list=self.cal_weight(i,v,mask,Segmodel)
        
        for k in range(len(self.blockflow)):
            f_list.append(f.data)
            grad=self.cal_grad_single(f,Segmodel,mask)
            att=self.cal_att(f_list,att,grad)
            f-self.alpha[k]*(f-(w_i*i+w_v*v))
            f,h,c,l_m,s_m=self.blockflow[k](f,att,h,c,l_m,s_m)

        return f,w_i,w_v,map_i_list,map_v_list


    def cal_weight(self,i,v,mask,Segmodel):
        grad_map_i=self.cal_weight_IG(i,mask,Segmodel)
        grad_map_v=self.cal_weight_IG(v,mask,Segmodel)
        w_i=torch.sum(grad_map_i,1,keepdim=True)
        w_v=torch.sum(grad_map_v,1,keepdim=True)
        w_i,w_v=(w_i+1e-8)/(w_i+w_v+2e-8),(w_v+1e-8)/(w_i+w_v+2e-8)
        return w_i,w_v,grad_map_i,grad_map_v


    def cal_weight_IG(self,img,mask,Segmodel):
        K=self.attnum
        grad_map=torch.zeros_like(img).repeat(1,15,1,1)
        for k in range(K):
            img_temp=img*(k+1)/K
            img_temp.requires_grad_(True)
            with torch.enable_grad():
                out=Segmodel(img_temp)
                for j in range(15):
                    index = torch.where(torch.argmax(mask, axis=1)==j)
                    indicat=torch.zeros_like(img)
                    indicat[index[0],:,index[1],index[2]]=1
                    if index[0].shape[0]!=0:
                        grad_temp=torch.autograd.grad(torch.mean(out[index[0],j,index[1],index[2]]),img_temp,retain_graph=True)[0]
                        grad_map[:,j:j+1,:,:]+=torch.sum(grad_temp *img * indicat)/torch.sum(indicat) * indicat

        grad_map=torch.relu(grad_map)/K            
        return grad_map
    
    def cal_grad_single(self,img,Segmodel,mask):
        img=img.data
        img.requires_grad_(True)
        grad=torch.zeros_like(img)
        with torch.enable_grad():
            SS_out=Segmodel(img)
            grad+=torch.autograd.grad(torch.mean(mask*SS_out),img,retain_graph=True)[0]
        return grad

    def cal_att(self,img_list,att,grad):
        if len(img_list)>1:
            att=(att*(len(img_list)-2)+grad*(img_list[-1]-img_list[-2]))/(len(img_list)-1)
        return att

def unit_test(model,segmodel):
    import numpy as np
    x = torch.tensor(np.random.rand(3,1,640,480).astype(np.float32)).cuda()
    z = torch.tensor(np.random.rand(3,15,640,480).astype(np.float32)).cuda()
    model.cuda()
    y = model(x,x,segmodel, z)[0]
    print('output shape:', y.shape)

if __name__ == "__main__":
    from SS_Net import SS_Net
    import sys
    import os
    sys.path.append(os.getcwd())
    model = UAAFusion().cuda()
    segmodel=SS_Net(15).cuda()
    unit_test(model,segmodel)