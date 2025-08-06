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


class conceptfusion_sam2(nn.Module):
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

        self._load_clip()
        self._build_model()

        self.visual_encoder = self.clip.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
    def _load_clip(self):
        # pdb.set_trace()
        open_clip_model = "ViT-H-14"
        open_clip_pretrained_dataset = "laion2b_s32b_b79k"
        # open_clip_model = "ViT-B-16"
        # open_clip_pretrained_dataset = "laion2b_s34b_b88k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False
        
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

        input_color = img - img.min() 
        input_color = input_color / input_color.max() 
        input_color = (input_color * 255).clamp(0, 255).byte()  
        input_color = input_color[0].permute(2, 1, 0)    # H W C

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')

        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H


        x = self.conv1(sim)
        x = self.conv2d(x)

        heatmap = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return heatmap # one-hot is used for evaluating the prescious of conceptfuion



class conceptfusion_sam2_1(nn.Module):
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

        self._load_clip()
        self._build_model()

        self.visual_encoder = self.clip.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
    def _load_clip(self):
        # pdb.set_trace()
        # open_clip_model = "ViT-H-14"
        # open_clip_pretrained_dataset = "laion2b_s32b_b79k"
        open_clip_model = "ViT-B-16"
        open_clip_pretrained_dataset = "laion2b_s34b_b88k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False
        
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

        input_color = img - img.min() 
        input_color = input_color / input_color.max()  
        input_color = (input_color * 255).clamp(0, 255).byte()  
        input_color = input_color[0].permute(2, 1, 0)    # H W C

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')

        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H


        x = self.conv1(sim)
        x = self.conv2d(x)

        heatmap = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return heatmap # one-hot is used for evaluating the prescious of conceptfuion


class conceptfusion_sam2_kernel(conceptfusion_sam2):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
        # self._load_clip()
        # self.visual_encoder = self.clip.visual
        # for param in self.visual_encoder.parameters():
        #     param.requires_grad = False
    
        self._load_clip()

        self.visual_encoder = self.clip.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
    def _load_clip(self):
        # pdb.set_trace()
        open_clip_model = "ViT-H-14"
        open_clip_pretrained_dataset = "laion2b_s32b_b79k"
        # open_clip_model = "ViT-B-16"
        # open_clip_pretrained_dataset = "laion2b_s34b_b88k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False

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
            nn.Conv2d(512, 64, kernel_size=1),
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
    def extract_patch_features(self, input_image, kernel=False):

        x = self.visual_encoder.conv1(input_image)  # (batch_size, hidden_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch_size, num_patches, hidden_dim)

        x = torch.cat([self._expand_token(self.visual_encoder.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = torch.cat([vision_model.class_embedding.expand(x.shape[0], -1, -1), x], dim=1)  # 添加 CLS token
        x = x + self.visual_encoder.positional_embedding  # (batch_size, num_patches+1, hidden_dim)  

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
        # pdb.set_trace()
        patch_feat = patch_feat.permute(0,2,1)  # [1, 1280, 256]
        patch_feat = self.conv1d(patch_feat)   # [1, 1024, 256]
        patch_feat = patch_feat.permute(0,2,1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        _sim = cosine_similarity(patch_feat, global_feat)  # [1, 256]
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        _sim = _sim.view(1, 16, 16).unsqueeze(0)
        

        patch_feat = patch_feat.permute(0,2,1).view(1, 1024, 16, 16)
        x = self.conv1(_sim)
        x = self.lat_fusion1(x, patch_feat)
        x = self.up1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x
    
    def forward(self, x, l, sim):
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

        # pdb.set_trace()
        output = F.interpolate(sim_kernel, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return output

class conceptfusion_sam2_kernel_sim(conceptfusion_sam2):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
        self._load_sam2_o1()

    def _load_sam2_o1(self):
        sam2_checkpoint = "~/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        for param in self.mask_generator.predictor.model.parameters():
            param.requires_grad = False
        # pdb.set_trace()
        for name, param in self.mask_generator.predictor._transforms.named_parameters():
            print(f"{name}:requires_grad={param.requires_grad}")
        # pdb.set_trace()
        for param in self.mask_generator.predictor._transforms.parameters():
            param.requires_grad = False
        
    def _load_clip(self):
        # pdb.set_trace()
        open_clip_model = "ViT-H-14"
        open_clip_pretrained_dataset = "laion2b_s32b_b79k"
        # open_clip_model = "ViT-B-16"
        # open_clip_pretrained_dataset = "laion2b_s34b_b88k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False
        # unlock bias terms and LN terms.
        i = 0
        for name, param in self.clip.transformer.named_parameters():
            if "c_proj.bias" in name:
                param.requires_grad = True
                i = i + 1
            if i == 3:
                break
        j = 0
        for name, param in self.clip.visual.named_parameters():
            if 'ln' in name:
                param.requires_grad = True
                j = j + 1
            if j == 6:
                break

    def similarity_map(self, input_color, input_depth, l):
        img = input_color.cpu().numpy()  
        dep = input_depth[:,:,0].cpu().numpy()
        with torch.no_grad():
            # masks_original = self.mask_generator_2.generate(img)
            masks_original = self.mask_generator.generate(img)
        # cur_mask = masks_original[0]['segmentation']
        img_w, img_h, _ = img.shape
        # pdb.set_trace()
        l = 'A photo of ' + l + '.'
        
        # pdb.set_trace()
        masks = [] 
        target_rgb = 0
        idx = 0
        # pdb.set_trace()
        for mask_data in masks_original:
            # pdb.set_trace()
            mask = mask_data["segmentation"]  
            pixel_count = np.sum(mask > 0)    
            masked_image = dep[mask] 
            target_count = np.sum(masked_image <= target_rgb)
            total_count = np.sum(mask)

            _x, _y, _w, _h = tuple(int(x) for x in mask_data["bbox"])
            if _w < 1 or _h < 1:   # to avoid division by zero in clip_preprocess
                continue
            if _w >= img_w - 1 or _h >= img_h - 1:  # to avoid the whole crop or some boundary shadow
                continue
            if target_count == total_count:  # to avoid shadow
                continue
            masks.append(mask_data)
            if pixel_count >= 70: 
                masks.append(mask_data)

        out = (img - img.min()) / (img.max() - img.min())
        plt.imshow(out)
        plt.savefig("~/cliport/cliport/visualization/kernel_image.png")
        # plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.savefig("cliport/visualization/kernel_masks.png") 

        print(f"The number of original masks: {len(masks_original)}")
        print(f"The number of filtered masks: {len(masks)}")   

        with torch.cuda.amp.autocast():
            _img = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            # pdb.set_trace()
            global_feat = self.clip.encode_image(_img.cuda())  # [1, 1024]
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
        global_feat = global_feat.half().cuda()
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for maskidx in range(len(masks)):
            # pdb.set_trace()
            _x, _y, _w, _h = tuple(int(x) for x in masks[maskidx]["bbox"]) # xywh bounding box
            # print(f"x,y,w,h: {_x, _y, _w, _h}") 
            seg = masks[maskidx]["segmentation"]
            nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]["segmentation"]))
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]
            img_roi = Image.fromarray(img_roi)
            img_roi = self.clip_preprocess(img_roi).unsqueeze(0).cuda()   #
            roifeat = self.clip.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            roifeat = roifeat.half().cuda()
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0).half()
        outfeat = torch.zeros(img_w, img_h, feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [img_w, img_h], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0] # --> H, W, feat_dim

        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        text1 = tokenizer(l)
        textfeat1 = self.clip.encode_text(text1.cuda())
        textfeat1 = torch.nn.functional.normalize(textfeat1, dim=-1)
        textfeat1 = textfeat1.unsqueeze(0)

        _simfunc = torch.nn.CosineSimilarity(dim=-1)
        _sim = _simfunc(outfeat.float().cuda(), textfeat1)  # H, W
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        sim1 = _sim.detach().cpu().numpy()
        # pdb.set_trace()
        plt.imshow(sim1)
        plt.savefig('cliport/visualization/kernel_masks.png')

        return sim1
    
    def forward(self, x, l, softmax=False):  
        img = self.preprocess(x, dist='clip')
        img = img[:,:3]  # B C W H
        depth = x[:,3:]
        img = img.half()
        in_type = x.dtype  # in_shape = x.shape
        in_shape = img.shape

        input_tensor = img - img.min()  
        input_tensor = input_tensor / input_tensor.max()  
        input_tensor = (input_tensor * 255).clamp(0, 255).byte() 
        
        # pdb.set_trace()
        sim_list = []
        for i in range(len(input_tensor)):
            input_color = img - img.min() 
            input_color = input_color / input_color.max() 
            input_color = (input_color * 255).clamp(0, 255).byte() 
            input_color = input_color[0].permute(2, 1, 0)    # H W C

            input_depth = depth / depth.max() 
            input_depth = (input_depth * 255).clamp(0, 255).byte()  
            input_depth = input_depth[0].permute(2, 1, 0)    # H W C
            
            sim = self.similarity_map(input_color, input_depth, l)   # H W C --> 1 H W
            # pdb.set_trace()
            # _sim = torch.tensor(sim, dtype=in_type, device='cuda')
            sim_list.append(torch.from_numpy(sim)) 
        sim_kernel = torch.stack(sim_list) # B, 1, W, H
        sim_kernel = sim_kernel.unsqueeze(1)
        sim_kernel = sim_kernel.cuda()
        
        # pdb.set_trace()
        x = self.conv1(sim_kernel)
        x = self.conv2d(x)

        output = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return output

class conceptfusion_sam2_kernel_real(conceptfusion_sam2_kernel):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _load_clip(self):
        # pdb.set_trace()
        open_clip_model = "ViT-B-16"
        open_clip_pretrained_dataset = "laion2b_s34b_b88k"
        model, compose, preprocess = open_clip.create_model_and_transforms(open_clip_model, open_clip_pretrained_dataset)
        model.cuda()
        model.eval()
        self.clip = model
        self.clip_preprocess = preprocess
        del model
        for param in self.clip.parameters():
            param.requires_grad = False

    def _build_model(self):
        # self.proj = nn.Sequential(
        #     nn.linear(1024, 512)
        # )
        self.conv1d = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.lat_fusion1 = FusionConvLat(input_dim=64+512, output_dim=512)
        # self.up1 = Up(512, 64, self.bilinear)   # 32
        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 64, kernel_size=1),
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
        # pdb.set_trace()
        patch_feat = patch_feat.permute(0,2,1)  # [1, 1280, 256]
        patch_feat = self.conv1d(patch_feat)   # [1, 1024, 256]
        patch_feat = patch_feat.permute(0,2,1)

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        _sim = cosine_similarity(patch_feat, global_feat)  # [1, 256]
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        _sim = _sim.view(1, 14, 14).unsqueeze(0)

        patch_feat = patch_feat.permute(0,2,1).view(1, 512, 14, 14)
        x = self.conv1(_sim)
        x = self.lat_fusion1(x, patch_feat)
        x = self.up1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x