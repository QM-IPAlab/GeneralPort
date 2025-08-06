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


class conceptfusion_clip_large(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.output_dim = output_dim
        self.bilinear = True
        self.batchnorm = self.cfg['train']['batchnorm']

        sam2_checkpoint = "~/sam2/checkpoints/sam2.1_hiera_large.pt"
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
        """
        提取 CLIP 模型中每个 patch 的特征。
        Args:
            vision_model: CLIP 的视觉编码器 (ViT)。
            input_image: 经过预处理的输入图像 (Tensor, shape=(1, 3, H, W))。
        Returns:
            patch_features: 提取的 patch 特征 (Tensor, shape=(batch_size, num_patches, hidden_dim))。
        """
        # 1. 图像转换为 patch tokens
        x = self.visual_encoder.conv1(input_image)  # (batch_size, hidden_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch_size, num_patches, hidden_dim)

        # 2. 添加位置编码
        x = torch.cat([self._expand_token(self.visual_encoder.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = torch.cat([vision_model.class_embedding.expand(x.shape[0], -1, -1), x], dim=1)  # 添加 CLS token
        x = x + self.visual_encoder.positional_embedding  # (batch_size, num_patches+1, hidden_dim)  

        # 3. 通过 Transformer 模块
        x = self.visual_encoder.ln_pre(x)  # 预归一化

        if kernel:
            x = self.visual_encoder.transformer(x) 
            return x[:,1:,:]

        x_list = []
        for i, resblock in enumerate(self.visual_encoder.transformer.resblocks):
            x = resblock(x)
            if (i+1) % 8 == 0:
                x_list.append(x)
        x0 = x_list[0][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x0 = x0.permute(0, 3, 1, 2)
        x1 = x_list[1][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x_list[2][:, 1:, :].view(16, 16, 1280).unsqueeze(0)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x_list[3][:, 1:, :].view(16, 16, 1280).unsqueeze(0) 
        x3 = x3.permute(0, 3, 1, 2)  # [1, 1280, 16, 16]

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
        # 4. 提取 patch tokens (去掉 CLS token)
        # patch_features = x[:, 1:, :].view(16, 16, 1280)  # (batch_size, num_patches, hidden_dim)
        return x   # [1, 224, 224, 64]

    def get_one_hot(self, input_color, input_depth, l):
        img = input_color.cpu().numpy()  # 转为numpy，并加载到cpu上
        dep = input_depth[:,:,0].cpu().numpy()
        # 提取图像中所有的mask
        with torch.no_grad():
            masks_original = self.mask_generator_2.generate(img)
        # cur_mask = masks_original[0]['segmentation']
        
        # pdb.set_trace()
        masks = []  # 用于存储过滤后的掩码
        target_rgb = 10
        idx = 0
        for mask_data in masks_original:
            # pdb.set_trace()
            mask = mask_data["segmentation"]  # 提取掩码 (2D numpy 数组)
            pixel_count = np.sum(mask > 0)    # 计算掩码中非零像素的数量
            masked_image = dep[mask] 
            # filter_dep = np.any(masked_image <= 9)  # np.all(), np.any()判断函数
            target_count = np.sum(masked_image <= target_rgb)
            total_count = np.sum(mask)

            _x, _y, _w, _h = tuple(int(x) for x in mask_data["bbox"])
            if _w < 2 or _h < 2:   # to avoid division by zero in clip_preprocess
                continue
            if target_count > 0.3 * total_count:
                # 过滤桌面和阴影masks
                # print(target_count)
                continue
            if pixel_count >= 70: 
                # 过滤小于 70 像素的掩码
                masks.append(mask_data)
                mask_overlay = np.zeros_like(img, dtype=np.uint8) # mask_overlay = np.zeros_like(dep, dtype=np.uint8)

                # 为当前掩码区域添加颜色（例如红色）
                mask_overlay[mask > 0] = [255, 0, 0]  # 红色区域
                
                # 混合原图和当前掩码
                blended = cv2.addWeighted(img, 0.8, mask_overlay, 0.5, 0)  # blended = cv2.addWeighted(dep, 0.8, mask_overlay, 0.5, 0)
                save_path = f"cliport/visualization/a{idx + 1:03d}.png"  # save_path = f"cliport/visualization/dep_filter.png" 
                # 显示当前掩码覆盖图
                plt.figure(figsize=(8, 8))
                plt.imshow(blended)
                plt.savefig(save_path)  
                idx = idx + 1

        # if len(masks)==0:
        #     print(l)
        #     out = (img - img.min()) / (img.max() - img.min())
        #     plt.imshow(out)
        #     plt.savefig("/home/a/acw799/cliport/cliport/visualization/wrong_image.png")

        #     plt.figure(figsize=(20, 20))
        #     plt.imshow(img)
        #     show_anns(mask1)
        #     plt.savefig("cliport/visualization/wrong_image_masks.png")

        # pdb.set_trace()
        # # 将img整理成可视化状态
        # out = (img - img.min()) / (img.max() - img.min())
        # # 生成原图
        # plt.imshow(out)
        # plt.savefig("/home/a/acw799/cliport/cliport/visualization/initial_image.png")
        # plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.savefig("cliport/visualization/image_masks.png") 

        print(f"The number of original masks: {len(masks_original)}")
        print(f"The number of filtered masks: {len(masks)}") 

        if len(masks) == 0:
            # pdb.set_trace()
            for idx, mask_data in enumerate(masks_original):
                mask = mask_data["segmentation"]  # 获取当前掩码
                mask_overlay = np.zeros_like(img, dtype=np.uint8)

                # 为当前掩码区域添加颜色（例如红色）
                mask_overlay[mask > 0] = [255, 0, 0]  # 红色区域
                
                # 混合原图和当前掩码
                blended = cv2.addWeighted(img, 0.8, mask_overlay, 0.5, 0)
                save_path = f"cliport/visualization/{idx + 1:05d}.png"
                # 显示当前掩码覆盖图
                plt.figure(figsize=(8, 8))
                plt.imshow(blended)
                plt.savefig(save_path) 

        # pdb.set_trace()

        # 查看每一个mask
        # for idx, mask_data in enumerate(masks):
        #     mask = mask_data["segmentation"]  # 获取当前掩码
        #     mask_overlay = np.zeros_like(img, dtype=np.uint8)

        #     # 为当前掩码区域添加颜色（例如红色）
        #     mask_overlay[mask > 0] = [255, 0, 0]  # 红色区域
            
        #     # 混合原图和当前掩码
        #     blended = cv2.addWeighted(img, 0.8, mask_overlay, 0.5, 0)
        #     save_path = f"cliport/visualization/{idx + 1:05d}.png"
        #     # 显示当前掩码覆盖图
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(blended)
        #     plt.savefig(save_path)   

        with torch.cuda.amp.autocast():
            _img = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            # pdb.set_trace()
            global_feat = self.clip.encode_image(_img.cuda())  # [1, 1024]
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
        global_feat = global_feat.half().cuda()
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        with torch.no_grad():
            # pdb.set_trace()
            pixel_feat = self.extract_patch_features(_img.cuda())  # [1, 64, 224, 224]

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
        text = tokenizer(l)
        textfeat = self.clip.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        _simfunc = torch.nn.CosineSimilarity(dim=-1)
        _sim = _simfunc(outfeat.float().cuda(), textfeat)  # H, W
        _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        _sim = _sim.detach().cpu().numpy()

        # pdb.set_trace()
        # # # 生成相似度图
        # plt.imshow(_sim)
        # plt.savefig("/home/a/acw799/cliport/cliport/visualization/sim.png")

        # get the mask positions of the maximum of simlarity map
        max_similarity_value = np.max(_sim)
        mask_positions = np.where(_sim == max_similarity_value)
        # pdb.set_trace()
        # get the centre coordinate
        y_indices, x_indices = mask_positions
        total_similarity = np.sum(_sim[mask_positions])
        centroid_x = np.sum(x_indices * _sim[mask_positions]) / total_similarity
        centroid_y = np.sum(y_indices * _sim[mask_positions]) / total_similarity  
        
        # pdb.set_trace()
        one_hot = np.zeros_like(_sim)  # H W

        centroid_x = int(np.round(centroid_x))
        centroid_y = int(np.round(centroid_y))
        centroid_x = np.clip(centroid_x, 0, one_hot.shape[0] - 1)
        centroid_y = np.clip(centroid_y, 0, one_hot.shape[1] - 1)
        one_hot[centroid_x, centroid_y] = 1.0

        return one_hot, _sim, pixel_feat  #, patch_feat

    def forward(self, x, l, _sim):
        x = self.preprocess(x, dist='clip')
        img = x[:,:3]  # B C W H
        depth = x[:,3:]
        # pdb.set_trace()
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

        input_color = img - img.min()  # 平移到 0
        input_color = input_color / input_color.max()  # 归一化到 [0, 1]
        input_color = (input_color * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 范围并转为 uint8
        input_color = input_color[0].permute(2, 1, 0)    # H W C
        
        # one_hot, _sim, pixel_feat = self.get_one_hot(input_color, input_depth, l)  # H W
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

class conceptfusion_large_place(conceptfusion_clip_large):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
    def forward(self, x, l):
        x = self.preprocess(x, dist='clip')
        img = x[:,:3]  # B C W H
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

        input_tensor = img - img.min()  # 平移到 0
        input_tensor = input_tensor / input_tensor.max()  # 归一化到 [0, 1]
        input_tensor = (input_tensor * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 范围并转为 uint8
        
        # pdb.set_trace()
        input_tensor = input_tensor[0].permute(2, 1, 0)    # H W C
        one_hot, _sim = self.get_one_hot(input_tensor, l)  # H W

        _sim = np.pad(_sim, ((top, bottom), (left, right)), mode='constant')
        one_hot = np.pad(one_hot, ((top, bottom), (left, right)), mode='constant')

        # pdb.set_trace()
        _sim = torch.tensor(_sim, dtype=in_type, device='cuda')   
        sim = _sim.view(1, 1, _sim.shape[0], _sim.shape[1])    # B C H W
        sim = sim.permute(0, 1, 3, 2)                          # [1, 1, 320, 160]  B C W H
        one_hot = torch.tensor(one_hot, dtype=in_type, device='cuda')
        one_hot = one_hot.view(1, 1, one_hot.shape[0], one_hot.shape[1])
        one_hot = one_hot.permute(0, 1, 3, 2)

        x = self.conv1(sim)
        x = self.sim_fusion(x, one_hot)

        x = self.conv2d(x)

        heatmap = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
  
        return heatmap

class conceptfusion_large_kernel(conceptfusion_clip_large):
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
        img = input_tensor.cpu().numpy()  # 转为numpy，并加载到cpu上

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
        # # 生成相似度图
        # kernel_sim = _sim[0].permute(1,2,0)
        # kernel_sim = (kernel_sim - kernel_sim.min()) / (kernel_sim.max() - kernel_sim.min() + 1e-12)
        # kernel_sim = kernel_sim.detach().cpu().numpy()
        # plt.imshow(kernel_sim)
        # plt.savefig("/home/a/acw799/cliport/cliport/visualization/sim_kernel1.png")

        patch_feat = patch_feat.permute(0,2,1).view(1, 1024, 16, 16)
        x = self.conv1(_sim)
        x = self.lat_fusion1(x, patch_feat)
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

        input_tensor = img - img.min()  # 平移到 0
        input_tensor = input_tensor / input_tensor.max()  # 归一化到 [0, 1]
        input_tensor = (input_tensor * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 范围并转为 uint8
        
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
