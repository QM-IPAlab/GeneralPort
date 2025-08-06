import os
from re import X
import sys
import pickle as pkl
from pathlib import Path
from typing import List, Union
import pdb
import logging
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
from PIL import Image
from typing_extensions import Literal
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import open_clip

np.random.seed(3)
class FiLM(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(text_dim, visual_dim)
        self.beta_fc = nn.Linear(text_dim, visual_dim)

    def forward(self, text_embedding, visual_embedding):
        """
        text_embedding: [1, 1024] -> text embedding vector
        visual_embedding: [256, 1280] -> visual embedding vector
        """
        gamma = self.gamma_fc(text_embedding)  # [1, 512] -> [1, 512]
        beta = self.beta_fc(text_embedding)    # [1, 512] -> [1, 512]

        gamma = gamma.expand(visual_embedding.shape)  # [256, 512]
        beta = beta.expand(visual_embedding.shape)    # [256, 512]

        modulated_embedding = gamma * visual_embedding + beta  # [256, 512]

        return modulated_embedding
    
class sam2clip_wo_pixel(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.output_dim = output_dim
        self.bilinear = True
        self.batchnorm = self.cfg['train']['batchnorm']

        sam2_checkpoint = "/home/a/acw799/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        # sam2 = build_sam2(cfg['sam2']['model_cfg'], cfg['sam2']['checkpoint'], device=device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        for param in self.mask_generator.predictor.model.parameters():
            param.requires_grad = False
        # pdb.set_trace()
        for name, param in self.mask_generator.predictor._transforms.named_parameters():
            print(f"{name}:requires_grad={param.requires_grad}")
        # pdb.set_trace()
        for param in self.mask_generator.predictor._transforms.parameters():
            param.requires_grad = False

        self.mask_generator_2 = SAM2AutomaticMaskGenerator(
                                                            model=sam2,
                                                            points_per_side=64,
                                                            points_per_batch=128,
                                                            pred_iou_thresh=0.7,
                                                            stability_score_thresh=0.92,
                                                            stability_score_offset=0.7,
                                                            crop_n_layers=1,
                                                            box_nms_thresh=0.7,
                                                            crop_n_points_downscale_factor=2,
                                                            min_mask_region_area=25.0,
                                                            use_m2m=True,
                                                        )
        for param in self.mask_generator_2.predictor.model.parameters():
            param.requires_grad = False
        # pdb.set_trace()
        for name, param in self.mask_generator_2.predictor._transforms.named_parameters():
            print(f"{name}:requires_grad={param.requires_grad}")
        # pdb.set_trace()
        for param in self.mask_generator_2.predictor._transforms.parameters():
            param.requires_grad = False
        os.makedirs(cfg['conceptfusion'].save_dir, exist_ok=True)
        self.desired_height = cfg['conceptfusion']['desired_height']
        self.desired_width = cfg['conceptfusion']['desired_width']
        self.preprocess = preprocess

        self._load_clip()
        self._build_model()

        self.visual_encoder = self.clip.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
    def _load_clip(self):
        # pdb.set_trace()
        open_clip_model = "ViT-H-14"
        open_clip_pretrained_dataset = "laion2b_s32b_b79k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False
        # pdb.set_trace()
        # for name, param in self.clip.named_parameters():
        #     print(f"{name}:requires_grad={param.requires_grad}")
        # pdb.set_trace()
    
    def _build_model(self):

        self.film_layer = FiLM(text_dim=1024, visual_dim=1280)
        self.conv_film =  nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=1),
        )

        self.conv1 = DoubleConv(in_channels=1, out_channels=64)
        self.sim_fusion = FusionConvLat(input_dim=64+64, output_dim=64)

        self.conv2d = nn.Sequential(
            nn.Conv2d(64, self.output_dim, kernel_size=1)
        )

        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=1),
        )

        self.up =  nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1280, 64, kernel_size=1),
        )

        self.fusion = FusionConvLat(input_dim=128, output_dim=64)

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1)
        )
    
    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def extract_patch_features(self, input_image, kernel=False):
        x = self.visual_encoder.conv1(input_image)  # (batch_size, hidden_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch_size, num_patches, hidden_dim)

        x = torch.cat([self._expand_token(self.visual_encoder.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = torch.cat([vision_model.class_embedding.expand(x.shape[0], -1, -1), x], dim=1)  # CLS token
        x = x + self.visual_encoder.positional_embedding  # (batch_size, num_patches+1, hidden_dim)  

        x = self.visual_encoder.ln_pre(x) 

        if kernel:
            x = self.visual_encoder.transformer(x) 
            return x[:,1:,:]
        
        final_x = self.visual_encoder.transformer(x) 
        visual_feat = self.visual_encoder.ln_post(final_x)

        x_list = []
        for i, resblock in enumerate(self.visual_encoder.transformer.resblocks):
            x = resblock(x)
            if (i+1) % 8 == 0:
                x_list.append(x)

        text_feat = text_feat[0]
        visual_feat = visual_feat[0][1:]
        x = self.film_layer(text_feat, visual_feat) # [256, 1280]
        x3 = x.view(16, 16, 1280).unsqueeze(0).permute(0,3,1,2)

        x0 = x_list[0][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x0 = x0.permute(0, 3, 1, 2)
        x1 = x_list[1][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x_list[2][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x2 = x2.permute(0, 3, 1, 2)

        x0 = self.up(x0)  # 1, 64, 32, 32
        x0 = self.up1(x0) # 1, 64, 64, 64
        x0 = self.up1(x0)
        x1 = self.up(x1)  # 1, 64, 32, 32
        x1 = self.up1(x1) # 1, 64, 64, 64
        x2 = self.up(x2)  # 1, 64, 32, 32
        x3 = self.up(x3)

        x = self.fusion(x3, x2)  # 32
        x = self.up1(x)
        x = self.fusion(x, x1)   # 64 
        x = self.up1(x)
        x = self.fusion(x, x0)   # 128
        x = self.up1(x)          # 256
        x = self.conv(x)

        x =  F.interpolate(x, size=(224, 224), mode='bilinear')
        # patch_features = x[:, 1:, :].view(16, 16, 1280)  # (batch_size, num_patches, hidden_dim)
        return x   # [1, 224, 224, 64]

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

        input_color = img - img.min()  
        input_color = input_color / input_color.max()  
        input_color = (input_color * 255).clamp(0, 255).byte()  
        input_color = input_color[0].permute(2, 1, 0)    # H W C
        
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        text = tokenizer(l)
        textfeat = self.clip.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        img = input_color.cpu().numpy()
        with torch.cuda.amp.autocast():
            _img = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
        with torch.no_grad():
            # pdb.set_trace()
            pixel_feat = self.extract_patch_features(_img.cuda())  # [1, 64, 224, 224]       

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')
        # one_hot = np.pad(one_hot, ((top, bottom), (left, right)), mode='constant')

        pixel_feat = F.interpolate(pixel_feat, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H
        # one_hot = torch.tensor(one_hot, dtype=in_type, device='cuda')
        # one_hot = one_hot.view(1, 1, one_hot.shape[0], one_hot.shape[1])
        # one_hot = one_hot.permute(0, 1, 3, 2)

        x = self.conv1(sim)
        x = self.sim_fusion(x, pixel_feat)
        x = self.conv2d(x)

        heatmap = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return heatmap # one-hot is used for evaluating the prescious of conceptfuion


class sam2clip_wo_pixel_kernel(sam2clip_wo_pixel):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_model(self):
        # self.proj = nn.Sequential(
        #     nn.linear(1024, 512)
        # )
        self.conv1d = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.lat_fusion1 = FusionConvLat(input_dim=64+1024, output_dim=512)
        # self.up1 = Up(512, 64, self.bilinear)   # 32
        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True)
        )   # 64
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True)
        )   # 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, self.output_dim, kernel_size=1)
        )

    def extract_kernel_features(self, input_image):

        x = self.visual_encoder.conv1(input_image)  # (batch_size, hidden_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch_size, num_patches, hidden_dim)

        x = torch.cat([self._expand_token(self.visual_encoder.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = torch.cat([vision_model.class_embedding.expand(x.shape[0], -1, -1), x], dim=1)  # CLS token
        x = x + self.visual_encoder.positional_embedding  # (batch_size, num_patches+1, hidden_dim)  

        x = self.visual_encoder.ln_pre(x) 

        x = self.visual_encoder.transformer(x) 
        return x[:,1:,:]

    def similarity_map(self, input_tensor):
        img = input_tensor.cpu().numpy()  

        [H, W, C] = img.shape
        with torch.cuda.amp.autocast():
            _img = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            global_feat = self.clip.encode_image(_img.cuda())
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
        global_feat = global_feat.half().cuda()
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]

        with torch.no_grad():
            patch_feat = self.extract_kernel_features(_img.cuda())

        # patch_feat = self.proj(patch_feat)
        patch_feat = patch_feat.permute(0,2,1)  # [1, 1280, 256]
        patch_feat = self.conv1d(patch_feat)   # [1, 1024, 256]
        patch_feat = patch_feat.permute(0,2,1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        _sim = cosine_similarity(patch_feat, global_feat)  # [1, 256]
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        _sim = _sim.view(1, 16, 16).unsqueeze(0)

        patch_feat = patch_feat.permute(0,2,1).view(1, 1024, 16, 16)
        x = self.conv1(_sim)
        x = self.up1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x
    
    def forward(self, x, l):
        # pdb.set_trace()
        img = self.preprocess(x, dist='clip')
        img = img[:,:3]  # B C W H
        in_type = x.dtype  # in_shape = x.shape
        in_shape = img.shape

        input_tensor = img - img.min()  
        input_tensor = input_tensor / input_tensor.max()  
        input_tensor = (input_tensor * 255).clamp(0, 255).byte() 
        
        # pdb.set_trace()
        sim_list = []
        for i in range(len(input_tensor)):
            sim = self.similarity_map(input_tensor[i].permute(2,1,0))   # H W C --> 1 H W
            # pdb.set_trace()
            # _sim = torch.tensor(sim, dtype=in_type, device='cuda')
            sim_list.append(sim.to(in_type)) 
        sim_kernel = torch.stack(sim_list) # B, 1, H, W
        sim_kernel = sim_kernel.squeeze(1)
        sim_kernel = sim_kernel.permute(0, 1, 3, 2) # B, 1, W, H

        output = F.interpolate(sim_kernel, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return output
