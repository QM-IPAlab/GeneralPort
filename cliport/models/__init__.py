from cliport.models.resnet import ResNet43_8s
from cliport.models.clip_wo_skip import CLIPWithoutSkipConnections

from cliport.models.rn50_bert_unet import RN50BertUNet
from cliport.models.rn50_bert_lingunet import RN50BertLingUNet
from cliport.models.rn50_bert_lingunet_lat import RN50BertLingUNetLat
from cliport.models.untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from cliport.models.clip_unet import CLIPUNet
from cliport.models.clip_lingunet import CLIPLingUNet, CLIPLingUNetLessLayers

from cliport.models.resnet_lang import ResNet43_8s_lang

from cliport.models.resnet_lat import ResNet45_10s
from cliport.models.clip_unet_lat import CLIPUNetLat
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.clip_film_lingunet_lat import CLIPFilmLingUNet

from cliport.models.clip_cocoop import cocoop
from cliport.models.clip_coop import coop
from cliport.models.coop import coop_rn50
from cliport.models.cocoop import cocoop_rn50

from cliport.models.conceptfusion_sam2 import conceptfusion_sam2, conceptfusion_sam2_1, conceptfusion_sam2_kernel, conceptfusion_sam2_kernel_sim, conceptfusion_sam2_kernel_real
from cliport.models.conceptfusion_clip_large import conceptfusion_clip_large, conceptfusion_large_place, conceptfusion_large_kernel
from cliport.models.Unetr import Unetr, Unetr_kernel
from cliport.models.sam2clip_wo_pixel import sam2clip_wo_pixel, sam2clip_wo_pixel_kernel

from cliport.models.conceptfusion_sam2_lat import ConceptfusionSam2Lat, ConceptfusionSam2Lat_kernel

from cliport.models.pretrain import pretrain, pretrain_kernel


names = {
    # resnet
    'plain_resnet': ResNet43_8s,
    'plain_resnet_lang': ResNet43_8s_lang,

    # without skip-connections
    'clip_woskip': CLIPWithoutSkipConnections,

    # unet
    'clip_unet': CLIPUNet,
    'rn50_bert_unet': RN50BertUNet,

    # lingunet
    'clip_lingunet': CLIPLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,
    'clip_lingunet_lesslayers': CLIPLingUNetLessLayers,

    # lateral connections
    'plain_resnet_lat': ResNet45_10s,
    'clip_unet_lat': CLIPUNetLat,
    'clip_lingunet_lat': CLIPLingUNetLat,
    'clip_film_lingunet_lat': CLIPFilmLingUNet,
    'rn50_bert_lingunet_lat': RN50BertLingUNetLat,

    # cocoop
    'cocoop': cocoop,
    'coop': coop,
    'coop_rn50': coop_rn50,
    'cocoop_rn50': cocoop_rn50,

    # cliport without decoder
    'conceptfusion_sam2': conceptfusion_sam2,
    'conceptfusion_sam2_1': conceptfusion_sam2_1,
    'conceptfusion_sam2_kernel': conceptfusion_sam2_kernel,
    'conceptfusion_sam2_kernel_sim': conceptfusion_sam2_kernel_sim,
    'conceptfusion_clip_large': conceptfusion_clip_large,
    'conceptfusion_large_place': conceptfusion_large_place,
    'conceptfusion_large_kernel': conceptfusion_large_kernel,

    # sam2 + clip
    'sam2clip_wo': sam2clip_wo_pixel,
    'sam2clip_wo_kernel': sam2clip_wo_pixel_kernel,

    # clip without sam
    'Unetr': Unetr,
    'Unetr_kernel': Unetr_kernel,

    # clip + tranportnet + sam2
    'ConceptfusionSam2Lat': ConceptfusionSam2Lat,
    'ConceptfusionSam2Lat_kernel': ConceptfusionSam2Lat_kernel,
    'conceptfusion_sam2_kernel_real': conceptfusion_sam2_kernel_real,

    # pretrain
    'pretrain': pretrain,
    'pretrain_kernel': pretrain_kernel,

}
