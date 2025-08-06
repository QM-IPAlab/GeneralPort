import os
from cliport.models.streams.clipfit_for_real import clipfit_real
import cv2
from cliport.agents.transporter import TransporterAgent
import numpy as np
import pdb

from cliport.utils import utils
import cliport.utils.visual_utils as vu
from cliport.models.streams.one_stream_conceptfusion import OneStreamAttenConceptFusion
from cliport.models.streams.one_stream_conceptfusion import OneStreamTransportConceptFusion
from cliport.models.streams.clipfit import clipfit


from cliport.agents.clip_conceptfusion import ConceptFusionAgent


class CLIPFitAgent(ConceptFusionAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def _build_model(self):
        atten_stream_fcn = 'conceptfusion_sam2'   # pick map
        key_stream_fcn = 'conceptfusion_sam2'     # place map
        query_stream_fcn = 'conceptfusion_sam2_kernel'   # crop
        self.attention = OneStreamAttenConceptFusion(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportConceptFusion(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = clipfit(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )

class CLIPFitAgent1(ConceptFusionAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def _build_model(self):
        atten_stream_fcn = 'conceptfusion_sam2_1'   # pick map
        key_stream_fcn = 'conceptfusion_sam2'     # place map
        query_stream_fcn = 'conceptfusion_sam2_kernel'   # crop
        self.attention = OneStreamAttenConceptFusion(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportConceptFusion(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = clipfit(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )

class CLIPFitAllSimAgent(ConceptFusionAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def _build_model(self):
        atten_stream_fcn = 'conceptfusion_sam2'   # pick map
        key_stream_fcn = 'conceptfusion_sam2'     # place map
        query_stream_fcn = 'conceptfusion_sam2_kernel_sim'   # crop
        self.attention = OneStreamAttenConceptFusion(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportConceptFusion(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = clipfit(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )

class PretrainAgent(ConceptFusionAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def _build_model(self):
        atten_stream_fcn = 'pretrain'   # pick map
        key_stream_fcn = 'pretrain'     # place map
        query_stream_fcn = 'pretrain_kernel'   # crop
        self.attention = OneStreamAttenConceptFusion(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportConceptFusion(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = clipfit(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )


class CLIPFitRealAgent(ConceptFusionAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
    
    def _build_model(self):
        atten_stream_fcn = 'conceptfusion_sam2'   # pick map
        key_stream_fcn = 'conceptfusion_sam2'     # place map
        query_stream_fcn = 'conceptfusion_sam2_kernel_real'   # crop
        self.attention = OneStreamAttenConceptFusion(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportConceptFusion(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = clipfit_real(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )