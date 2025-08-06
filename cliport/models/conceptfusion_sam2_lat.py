import os
from re import X
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

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            # import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


class ConceptfusionSam2Lat(nn.Module):
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
        self.conv2d = nn.Sequential(
            nn.Conv2d(32, self.output_dim, kernel_size=1)
        )

        self.down_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.output_dim, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(True))
        ])

        self.lat_fusion4 = FusionConvLat(input_dim=64+32, output_dim=32)
        self.lat_fusion3 = FusionConvLat(input_dim=128+64, output_dim=64)
        self.lat_fusion2 = FusionConvLat(input_dim=256+128, output_dim=128)
        self.lat_fusion1 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.fusion3 = FusionConvLat(input_dim=64+32, output_dim=32)
        self.fusion2 = FusionConvLat(input_dim=128+64, output_dim=64)
        self.fusion1 = FusionConvLat(input_dim=256+128, output_dim=128)
        
    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def forward(self, x, lat, l, _sim):
        # pdb.set_trace()
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

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')

        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H

        _sim = []
        x = sim
        for layer in self.down_layers:
            x = layer(x)
            _sim.append(x)

        x = self.lat_fusion1(_sim[-1], lat[-5])
        x = self.up(x)
        
        x = self.fusion1(x, self.lat_fusion2(_sim[-2], lat[-4]))
        x = self.up(x)

        x = self.fusion2(x, self.lat_fusion3(_sim[-3], lat[-3]))
        x = self.up(x)

        x = self.fusion3(x, self.lat_fusion4(_sim[-4], lat[-2]))

        x = self.conv2d(x)   # (1, 1, 320, 160)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return x # one-hot is used for evaluating the prescious of conceptfuion



class ConceptfusionSam2Lat_kernel(ConceptfusionSam2Lat):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
        
        self._build_kernel()

    def _build_kernel(self):
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
        self.lat_fusion = FusionConvLat(input_dim=64+1024, output_dim=256)

        self.down = nn.Conv2d(256, 256, kernel_size=1, stride=2)

        # self.up1 = Up(512, 64, self.bilinear)   # 32
        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(True)
        )   # 64
        self.up2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(True)
        )   # 64
        self.up3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(True)
        )   # 64
        
    def extract_patch_features(self, input_image, kernel=False):
        """
        extract patch feature from CLIP visual encoder
        Args:
            vision_model: CLIP visual encoder (ViT)
            input_image: pretrained input image (Tensor, shape=(1, 3, H, W))。
        Returns:
            patch_features: extracted patch feature (Tensor, shape=(batch_size, num_patches, hidden_dim))。
        """
        # 1. patch tokens
        x = self.visual_encoder.conv1(input_image)  # (batch_size, hidden_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch_size, num_patches, hidden_dim)

        # 2. positional embedding
        x = torch.cat([self._expand_token(self.visual_encoder.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = torch.cat([vision_model.class_embedding.expand(x.shape[0], -1, -1), x], dim=1)  #  CLS token
        x = x + self.visual_encoder.positional_embedding  # (batch_size, num_patches+1, hidden_dim)  

        # 3.  Transformer 
        x = self.visual_encoder.ln_pre(x)  

        if kernel:
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
            patch_feat = self.extract_patch_features(_img.cuda(), kernel=True)  # [1, 256, 1280] B C H W
        # patch_feat = self.proj(patch_feat)
        patch_feat = patch_feat.permute(0,2,1)  # [1, 1280, 256]
        patch_feat = self.conv1d(patch_feat)   # [1, 1024, 256]
        patch_feat = patch_feat.permute(0,2,1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        _sim = cosine_similarity(patch_feat, global_feat)  # [1, 256]
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        _sim = _sim.view(1, 16, 16).unsqueeze(0)
        
        # pdb.set_trace()
        # kernel_sim = _sim[0].permute(1,2,0)
        # kernel_sim = (kernel_sim - kernel_sim.min()) / (kernel_sim.max() - kernel_sim.min() + 1e-12)
        # kernel_sim = kernel_sim.detach().cpu().numpy()
        # plt.imshow(kernel_sim)
        # plt.savefig("/home/a/acw799/cliport/cliport/visualization/sim_kernel1.png")
        sim = []
        patch_feat = patch_feat.permute(0,2,1).view(1, 1024, 16, 16)
        x = self.conv1(_sim)
        x1 = self.lat_fusion(x, patch_feat) 
        x1 = self.down(x1) # [1, 512, 8, 8]
        sim.append(x1)
        x2 = self.up1(x1)  # [1, 256, 16, 16]
        sim.append(x2)
        x3 = self.up2(x2)  # [1, 128, 32, 32]
        sim.append(x3)
        x4 = self.up3(x3)  # [1, 64, 64, 64]
        sim.append(x4)

        return x1, x2, x3, x4
    
    def forward(self, x, lat, l):
        # pdb.set_trace()
        img = self.preprocess(x, dist='clip')
        img = img[:,:3]  # B C W H
        in_type = x.dtype  # in_shape = x.shape
        in_shape = img.shape

        input_tensor = img - img.min() 
        input_tensor = input_tensor / input_tensor.max()  
        input_tensor = (input_tensor * 255).clamp(0, 255).byte()  
        
        x1_list = []
        x2_list = []
        x3_list = []
        x4_list = []
        _sim = []
        for i in range(len(input_tensor)):
            x1, x2, x3, x4 = self.similarity_map(input_tensor[i].permute(2,1,0))   # H W C --> 1 H W
            # pdb.set_trace()
            # _sim = torch.tensor(sim, dtype=in_type, device='cuda')
            x1_list.append(x1.to(in_type)) 
            x2_list.append(x2.to(in_type))
            x3_list.append(x3.to(in_type))
            x4_list.append(x4.to(in_type))
        # pdb.set_trace()
        x4 = torch.stack(x4_list) # B, 1, H, W
        x4 = x4.squeeze(1)
        x4 = x4.permute(0, 1, 3, 2) # B, 1, W, H
        _sim.append(x4)

        x3 = torch.stack(x3_list) # B, 1, H, W
        x3 = x3.squeeze(1)
        x3 = x3.permute(0, 1, 3, 2) # B, 1, W, H
        _sim.append(x3)

        x2 = torch.stack(x2_list) # B, 1, H, W
        x2 = x2.squeeze(1)
        x2 = x2.permute(0, 1, 3, 2) # B, 1, W, H
        _sim.append(x2)

        x1 = torch.stack(x1_list) # B, 1, H, W
        x1 = x1.squeeze(1)
        x1 = x1.permute(0, 1, 3, 2) # B, 1, W, H
        _sim.append(x1)

        # pdb.set_trace()
        x = self.lat_fusion1(_sim[-1], lat[-5])  # [36, 512, 8, 8]
        x = self.up(x)  # [36, 256, 16, 16]
        
        x = self.fusion1(x, self.lat_fusion2(_sim[-2], lat[-4]))  # [36, 256, 16, 16]
        x = self.up(x)  # [36, 128, 32, 32]

        x = self.fusion2(x, self.lat_fusion3(_sim[-3], lat[-3]))  # [36, 128, 32, 32]
        x = self.up(x)  # [36, 64, 64, 64]

        x = self.fusion3(x, self.lat_fusion4(_sim[-4], lat[-2]))  # [36, 64, 64, 64]
        # [36, 32, 64, 64]
        x = self.conv2d(x)   # [36, 1, 64, 64]

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')  # [36, 1, 64, 64]
  
        return x
