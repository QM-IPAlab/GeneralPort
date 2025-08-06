import os
import sys
import pickle as pkl
from pathlib import Path
from typing import List, Union
import pdb
# new_path = '/gpfs/home/a/acw799/concept-fusion/examples'
# os.chdir(new_path)

import numpy as np
import torch
import torch.nn as nn
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up, DoubleConv, OutConv
from cliport.models.core.clip import build_model, load_clip, tokenize
from cliport.models.core.fusion import FusionConvLat
import torch.nn.functional as F

# from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from PIL import Image, ImageEnhance
from typing_extensions import Literal
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import open_clip

np.random.seed(3)

class pretrain(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.output_dim = output_dim
        self.bilinear = True
        self.batchnorm = self.cfg['train']['batchnorm']

        os.makedirs(cfg['conceptfusion'].save_dir, exist_ok=True)
        self.desired_height = cfg['conceptfusion']['desired_height']
        self.desired_width = cfg['conceptfusion']['desired_width']
        self.preprocess = preprocess

        self._build_model()

        
    def _build_model(self):
        self.conv1 = DoubleConv(in_channels=1, out_channels=64)

        self.conv2d = nn.Sequential(
            nn.Conv2d(64, self.output_dim, kernel_size=1)
        )
        
    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def forward(self, x, l, _sim):
        x = self.preprocess(x, dist='clip')
        img = x[:,:3]  # B C W H
        depth = x[:,3:]
        # pdb.set_trace()
        in_type = img.dtype  # in_shape = x.shape
        in_shape = img.shape
        img= img.half()
        load_img_height = img.shape[-1] 
        load_img_weight = img.shape[-2]
        # pdb.set_trace()
        # if load_img_height > self.desired_height:
        top = (load_img_height - self.desired_height) // 2
        bottom = load_img_height - self.desired_height - top
        left = (load_img_weight - self.desired_width) // 2
        right = load_img_weight - self.desired_width - left

        img = img[:, :, left:left + self.desired_width, bottom:bottom+self.desired_height]  # B C W H
        depth = depth[:, :, left:left + self.desired_width, bottom:bottom+self.desired_height]

        input_color = img - img.min()  # 平移到 0
        input_color = input_color / input_color.max()  # 归一化到 [0, 1]
        input_color = (input_color * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 范围并转为 uint8
        input_color = input_color[0].permute(2, 1, 0)    # H W C

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')

        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H


        x = self.conv1(sim)
        x = self.conv2d(x)

        heatmap = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return heatmap # one-hot is used for evaluating the prescious of conceptfuion

class pretrain_kernel(pretrain):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
    
    def forward(self, x, l, sim):
        # pdb.set_trace()
        img = self.preprocess(x, dist='clip')
        img = img[:,:3]  # B C W H
        in_type = x.dtype  # in_shape = x.shape
        in_shape = img.shape

        x = self.conv1(sim)
        x = self.conv2d(x)

        output = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return output
