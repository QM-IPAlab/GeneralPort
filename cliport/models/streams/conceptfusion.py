import os
import re
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


class conceptfusion(nn.Module):
    def __init__(self, cfg, device, preprocess):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.bilinear = True
        self.batchnorm = self.cfg['train']['batchnorm']

        self._load_sam2()

        os.makedirs(cfg['conceptfusion'].save_dir, exist_ok=True)
        self.desired_height = cfg['conceptfusion']['desired_height']
        self.desired_width = cfg['conceptfusion']['desired_width']
        self.preprocess = preprocess

        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        self.pattern = r'\b(' + '|'.join(self.keywords) + r')\b'

        self._load_clip()

        self.visual_encoder = self.clip.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
    
    def _load_sam2(self):
        sam2_checkpoint = "/home/a/acw799/sam2/checkpoints/sam2.1_hiera_large.pt"
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

    def instructions(self, l1, l2, name):
        if 'hanoi' in name:
            if 'the middle of the stand' in l2:
                l2 = l2[:11] + 'a medium brown square, darker than light brown but not as dark as chocolate brown.'
            elif 'darker brown side' in l2:
                l2 = l2[:11] + 'the darker brown square.'
            elif 'lighter brown side' in l2:
                l2 = l2[:11] + 'the white square.'
        if 'kits' in name:
            l2 = l2[:-1] + ', which is closed to black.'

        return l1, l2


    def similarity_map(self, input_color, input_depth, l1, l2, name):
        img = input_color.cpu().numpy()  
        dep = input_depth[:,:,0].cpu().numpy()
        with torch.no_grad():
            # masks_original = self.mask_generator_2.generate(img)
            masks_original = self.mask_generator.generate(img)
        # cur_mask = masks_original[0]['segmentation']

        # pdb.set_trace()
        l1 = 'A photo of ' + l1 + '.'
        l2 = 'A photo of ' + l2 + '.'
        l1, l2 = self.instructions(l1, l2, name)
        
        masks = []  
        target_rgb = 0
        idx = 0
        if 'packing-boxes-pairs' in name:
            target_rgb = 0
        if 'pyramid' in name:
            target_rgb = 0
        if 'kits' in name:
            l1 = l1[:-1] + ' whose color is important.'
            l2 = l2[:-1] + ' whose color is much darker.'
        for mask_data in masks_original:
            # pdb.set_trace()
            mask = mask_data["segmentation"]  
            pixel_count = np.sum(mask > 0)   
            masked_image = dep[mask] 
            target_count = np.sum(masked_image <= target_rgb)
            total_count = np.sum(mask)

            _x, _y, _w, _h = tuple(int(x) for x in mask_data["bbox"])
            if _w < 2 or _h < 2:   # to avoid division by zero in clip_preprocess
                continue
            if total_count > 0.8 * 160 * 320:
                continue
            if 'hanoi' in name:
                if target_count > 0.3 * total_count:
                    continue
                if pixel_count >= 70:
                    masks.append(mask_data)
                    continue
            if 'google' in name:
                if pixel_count < 70 and target_count > 0.3 * total_count:
                    continue
                if pixel_count >= 70: 
                    masks.append(mask_data)
                    continue
            if 'bowl' in name:    
                if target_count > 0.3 * total_count:
                    continue
            if pixel_count >= 70:  
                masks.append(mask_data)


        # pdb.set_trace()
        out = (img - img.min()) / (img.max() - img.min())
        plt.imshow(out)
        plt.savefig("~/cliport/cliport/visualization/google_image.png")
        # plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.savefig("cliport/visualization/google_masks.png") 

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
        outfeat = torch.zeros(self.desired_height, self.desired_width, feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [self.desired_height, self.desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0] # --> H, W, feat_dim

        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        text1 = tokenizer(l1)
        textfeat1 = self.clip.encode_text(text1.cuda())
        textfeat1 = torch.nn.functional.normalize(textfeat1, dim=-1)
        textfeat1 = textfeat1.unsqueeze(0)

        _simfunc = torch.nn.CosineSimilarity(dim=-1)
        _sim = _simfunc(outfeat.float().cuda(), textfeat1)  # H, W
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        sim1 = _sim.detach().cpu().numpy()

        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        text2 = tokenizer(l2)
        textfeat2 = self.clip.encode_text(text2.cuda())
        textfeat2 = torch.nn.functional.normalize(textfeat2, dim=-1)
        textfeat2 = textfeat2.unsqueeze(0)

        _sim = _simfunc(outfeat.float().cuda(), textfeat2)  # H, W
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        sim2 = _sim.detach().cpu().numpy()

        plt.imshow(sim1)
        plt.savefig("~/cliport/cliport/visualization/google_sim1.png")
        plt.imshow(sim2)
        plt.savefig("~/cliport/cliport/visualization/google_sim2.png")

        return sim1, sim2  
    
    def forward(self, inp_img, l, name, softmax=False):
        sentence = re.sub(r'^\S+\s*', '', l)
        match = re.search(self.pattern, sentence)
        if match:
            start_index = match.start()
            end_index = match.end()
            pick_l = sentence[:start_index].strip()
            place_l = sentence[end_index:].strip()
            
        in_shape = (1,) + inp_img.shape
        in_data = inp_img.reshape(in_shape)
        in_tensor = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # pdb.set_trace()
        in_tensor = in_tensor.permute(0, 3, 1, 2)  # [B 6 W H]
        
        x = self.preprocess(in_tensor, dist='clip')
        img = x[:,:3]  # B C W H
        depth = x[:,3:]
        in_type = img.dtype  # in_shape = x.shape
        in_shape = img.shape
        img = img.half()
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

        input_depth = depth / depth.max()  
        input_depth = (input_depth * 255).clamp(0, 255).byte()  
        input_depth = input_depth[0].permute(2, 1, 0)    # H W C
        
        sim1, sim2 = self.similarity_map(input_color, input_depth, pick_l, place_l, name)

        return sim1, sim2
    
