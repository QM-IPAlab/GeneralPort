from cliport.agents.transporter import TransporterAgent
from cliport.agents.clip_conceptfusion import ConceptFusionAgent
from cliport.models.streams.conceptfusion import conceptfusion
from cliport.utils import utils
from cliport.models.streams.two_stream_conceptfusion import TwoStreamAttenConceptfusion, TwoStreamTransportConceptfusion

class TransFusionSam2(ConceptFusionAgent):
    
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        atten_stream_fcn = 'ConceptfusionSam2Lat'   # pick map
        key_stream_fcn = 'ConceptfusionSam2Lat'     # place map
        query_stream_fcn = 'ConceptfusionSam2Lat_kernel'   # crop

        self.attention = TwoStreamAttenConceptfusion(
            stream_fcn=(stream_one_fcn, atten_stream_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportConceptfusion(
            key_stream_fcn=(stream_one_fcn, key_stream_fcn),
            query_stream_fcn=(stream_one_fcn, query_stream_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.conceptfusion = conceptfusion(
            cfg=self.cfg, 
            device=self.device_type, 
            preprocess=utils.preprocess,
        )


