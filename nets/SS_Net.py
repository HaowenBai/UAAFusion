import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.MobileNetv2 import MobileNetV2_Backbone


class SS_Net(nn.Module):
    def __init__(self, num_classes,downsample_factor=16):
        super(SS_Net, self).__init__()
        from functools import partial
        self.feature_extractor=MobileNetV2_Backbone()
        low_level_channels = 24
        
        self.total_idx  = len(self.feature_extractor.features)
        self.down_idx   = [2, 4, 7, 14]
        if downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.feature_extractor.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+320, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)  
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features = self.feature_extractor.features[:4](x)
        x = self.feature_extractor.features[4:](low_level_features)
        low_level_features = self.shortcut_conv(low_level_features)    
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return nn.Sigmoid()(x)


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

def unit_test(model):
    import numpy as np
    x = torch.tensor(np.random.rand(3,1,640,480).astype(np.float32)).cuda()
    model.cuda()
    y = model(x)
    print('output shape:', y.shape)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    model = SS_Net(9)
    for i, layer in enumerate(model.feature_extractor.features):
        print(i, layer)
    unit_test(model)
