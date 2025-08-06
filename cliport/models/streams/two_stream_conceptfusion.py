import pdb
from cliport import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from cliport.utils import utils
import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.streams.two_stream_attention import TwoStreamAttention


class TwoStreamAttenConceptfusion(nn.Module):
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
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
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)
        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l, sim):
        sentence = re.sub(r'^\S+\s*', '', l)
        pattern = r'\b(' + '|'.join(self.keywords) + r')\b'
        match = re.search(pattern, sentence)
        if match:
            start_index = match.start()
            l = sentence[:start_index].strip()
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l, sim)
        x = self.fusion(x1, x2)
        return x

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
    

class TwoStreamTransportConceptfusion(nn.Module):
    def __init__(self, key_stream_fcn, query_stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        super().__init__()
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
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

        self.sim_padding = np.zeros((2, 2), dtype=int)
        self.sim_padding[:2, :] = self.pad_size

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

        self.keywords = {'on', 'in', 'to', 'into', 'from'}
        self.pattern = r'\b(' + '|'.join(self.keywords) + r')\b'

    def _build_nets(self):
        key_stream_one_fcn, key_stream_two_fcn = self.key_stream_fcn
        query_stream_one_fcn, query_stream_two_fcn = self.query_stream_fcn
        key_one_model = models.names[key_stream_one_fcn]
        key_two_model = models.names[key_stream_two_fcn]
        query_one_model = models.names[query_stream_one_fcn]
        query_two_model = models.names[query_stream_two_fcn]

        self.key_stream_one = key_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.key_stream_two = key_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.query_stream_one = query_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.query_stream_two = query_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=1)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=1)

        print(f"Transport Place FCN - Stream One: {key_stream_one_fcn}, Stream Two: {key_stream_two_fcn}, Stream Fusion: {self.fusion_type}")
        print(f"Transport Kernel FCN - Stream One: {query_stream_one_fcn}, Stream Two: {query_stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l, sim):
        sentence = re.sub(r'^\S+\s*', '', l)
        match = re.search(self.pattern, sentence)
        if match:
            start_index = match.start()
            end_index = match.end()
            query_l = sentence[:start_index].strip()
            key_l = sentence[end_index:].strip()

        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, key_l, sim)
        logits = self.fusion_key(key_out_one, key_out_two)  # [1, 1, 320, 160]

        sim = np.pad(sim, self.sim_padding, mode='constant')
        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, query_l)
        kernel = self.fusion_query(query_out_one, query_out_two)  # [36, 1, 64, 64]

        return logits, kernel
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        # 两个输入应该是各自的feature map
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output
    
    def forward(self, inp_img, p, lang_goal, sim, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        # input_data = inp_img
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)
        # pdb.set_trace()
        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop, lang_goal, sim)

        return self.correlate(logits, kernel, softmax)