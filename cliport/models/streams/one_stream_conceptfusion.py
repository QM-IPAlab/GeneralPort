import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pdb
import matplotlib.pyplot as plt

import cliport.models as models
from cliport.utils import utils 
from cliport.models.core.transport import Transport

class OneStreamAttenConceptFusion(nn.Module):
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)
        self.keywords = {'on', 'in', 'to', 'into', 'from'}

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, l, sim):
        # pdb.set_trace()
        sentence = re.sub(r'^\S+\s*', '', l)
        pattern = r'\b(' + '|'.join(self.keywords) + r')\b'
        match = re.search(pattern, sentence)
        if match:
            start_index = match.start()
            l = sentence[:start_index].strip()
        x = self.attn_stream_one(x, l, sim)
        return x

    def init_forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]  has been to rotated

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        # pdb.set_trace()  # visualize the in_tens to check whether be rotated
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]  
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)
        # pdb.set_trace()

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal) # [B 1 W H]
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)
        # pdb.set_trace()

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]  # remove padding

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
    
    def forward(self, inp_img, lang_goal, sim, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        in_shape = (1,) + inp_img.shape
        in_data = inp_img.reshape(in_shape)
        in_tensor = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # pdb.set_trace()
        in_tensor = in_tensor.permute(0, 3, 1, 2)  # [B 6 W H]
        logits = self.attend(in_tensor, lang_goal, sim)

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
    

    
class OneStreamTransportConceptFusion(nn.Module):
    """Transport (a.k.a) Place module with language features fused at the bottleneck"""

    def __init__(self, key_stream_fcn, query_stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        super().__init__()
        self.iters = 0
        self.key_stream_fcn = key_stream_fcn
        self.query_stream_fcn = query_stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        self.pattern = r'\b(' + '|'.join(self.keywords) + r')\b'

    def _build_nets(self):
        key_stream_fcn, _ = self.key_stream_fcn
        query_stream_fcn, _ = self.query_stream_fcn
        key_stream_model = models.names[key_stream_fcn]
        query_stream_model = models.names[query_stream_fcn]

        self.key_stream_one = key_stream_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = query_stream_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

        print(f"Transport FCN: {key_stream_model}, {query_stream_model}")
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        # input feature map
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def transport(self, in_tensor, crop, l, sim_crop, sim):
        sentence = re.sub(r'^\S+\s*', '', l)
        match = re.search(self.pattern, sentence)
        if match:
            start_index = match.start()
            end_index = match.end()
            query_l = sentence[:start_index].strip()
            key_l = sentence[end_index:].strip()
            
        logits = self.key_stream_one(in_tensor, key_l, sim)
        kernel = self.query_stream_one(crop, query_l, sim_crop)

        return logits, kernel
    
    def forward(self, inp_img, p, lang_goal, sim_crop, sim, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        # pdb.set_trace()
        in_sim_crop = sim_crop.reshape(320,160,1)
        in_sim_crop = np.pad(in_sim_crop, self.padding, mode='constant')
        in_sim = in_sim_crop
        sim_shape = (1,) + in_sim.shape
        in_sim = in_sim.reshape(sim_shape)
        in_sim = torch.from_numpy(in_sim).to(dtype=torch.float, device=self.device)
        in_sim = in_sim.permute(0, 3, 1, 2)

        sim_crop = in_sim.repeat(self.n_rotations, 1, 1, 1)
        sim_crop = self.rotator(sim_crop, pivot=pv)
        sim_crop = torch.cat(sim_crop, dim=0)
        sim_crop = sim_crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop, lang_goal, sim_crop, sim)

        return self.correlate(logits, kernel, softmax)
    
class OneStreamAttenUnetr(nn.Module):
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)
        self.keywords = {'on', 'in', 'to', 'into', 'from'}

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, l):
        # pdb.set_trace()
        sentence = re.sub(r'^\S+\s*', '', l)
        pattern = r'\b(' + '|'.join(self.keywords) + r')\b'
        match = re.search(pattern, sentence)
        if match:
            start_index = match.start()
            l = sentence[:start_index].strip()
        x = self.attn_stream_one(x, l)
        return x

    def init_forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]  has been to rotated

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        # pdb.set_trace()  # visualize the in_tens to check whether be rotated
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]  
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)
        # pdb.set_trace()

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal) # [B 1 W H]
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)
        # pdb.set_trace()

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]  

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
    
    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        in_shape = (1,) + inp_img.shape
        in_data = inp_img.reshape(in_shape)
        in_tensor = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # pdb.set_trace()
        in_tensor = in_tensor.permute(0, 3, 1, 2)  # [B 6 W H]
        logits = self.attend(in_tensor, lang_goal)

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
    
class OneStreamTransportUnetr(nn.Module):
    """Transport (a.k.a) Place module with language features fused at the bottleneck"""

    def __init__(self, key_stream_fcn, query_stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        super().__init__()
        self.iters = 0
        self.key_stream_fcn = key_stream_fcn
        self.query_stream_fcn = query_stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        self.pattern = r'\b(' + '|'.join(self.keywords) + r')\b'

    def _build_nets(self):
        key_stream_fcn, _ = self.key_stream_fcn
        query_stream_fcn, _ = self.query_stream_fcn
        key_stream_model = models.names[key_stream_fcn]
        query_stream_model = models.names[query_stream_fcn]
        # pdb.set_trace()
        self.key_stream_one = key_stream_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = query_stream_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

        print(f"Transport FCN: {key_stream_model}, {query_stream_model}")
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def transport(self, in_tensor, crop, l):
        sentence = re.sub(r'^\S+\s*', '', l)
        match = re.search(self.pattern, sentence)
        if match:
            start_index = match.start()
            end_index = match.end()
            query_l = sentence[:start_index].strip()
            key_l = sentence[end_index:].strip()
            
        logits = self.key_stream_one(in_tensor, key_l)
        kernel = self.query_stream_one(crop, query_l)

        return logits, kernel
    
    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        # pdb.set_trace()
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        # pdb.set_trace()

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # pdb.set_trace()
        logits, kernel = self.transport(in_tensor, crop, lang_goal)
        # pdb.set_trace()

        return self.correlate(logits, kernel, softmax)