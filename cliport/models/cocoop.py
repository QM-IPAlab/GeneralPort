import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from cliport.models.coop import coop_rn50
import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat

class cocoop_rn50(coop_rn50):
    """ CLIP RN50 with U-Net skip connections and lateral connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_decoder(self):
        # prompt learning
        self.meta_net = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 8),
            nn.ReLU(True),
            nn.Linear(self.input_dim // 8, 512)
        )

        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)
        self.lat_fusion3 = FusionConvLat(input_dim=256+128, output_dim=128)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128+64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64+32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32+16, output_dim=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def encode_text(self, x, context):
        # pdb.set_trace()
        # with torch.no_grad():
        tokens = tokenize([x]).to(self.device)
        x = self.clip_rn50.token_embedding(tokens).type(self.dtype)
        n_ctx = context.shape[1]
        x = torch.cat([context, x[:, n_ctx :, :]], dim=1)
        text_feat, text_emb = self.clip_rn50.encode_text_for_prompt_learning(x, tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask
    
    def initial_encode_text(self, x):
        # pdb.set_trace()
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, lat, l):
        # pdb.set_trace()
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype  # fp32
        in_shape = x.shape
        x = x[:,:3]  # select RGB  [1, 3, 320, 320]
        # pdb.set_trace()
        x, im = self.encode_image(x)  # [1, 2048, 10, 10], [1, 32, 160, 160]// [1, 2048, 12, 7], [1, 32, 192, 112] // [36, 2048, 2, 2], [36, 32, 32, 32]
        x = x.to(in_type)  # [1, 2048, 10, 10] // [36, 2048, 2, 2]
        # pdb.set_trace()

        l = self.ctx_init + " " + l
        if x.shape[0] == 1:
            x_reshape = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        else:
            # pdb.set_trace()
            x_reshape = x.reshape(1, x.shape[1], x.shape[0] * x.shape[2] * x.shape[3])

        x_reshape = x_reshape.permute(0, 2, 1)
        bias = self.meta_net(x_reshape)
        bias = torch.mean(bias, dim=1).type(self.dtype)

        context = self.ctx + bias
        # pdb.set_trace()
        # if torch.allclose(self.save_ctx, context.data, atol=1e-8):
        #     print("no updating")
        # else:
        #     self.save_ctx = context.data
        #     print("Updated!")
        initial_l_enc, initial_l_emb, initial_l_mask = self.initial_encode_text(l)

        l_enc, l_emb, l_mask = self.encode_text(l, context)  # [1, 1024], [1, 77, 512], [1, 77]
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)  # (1, 1024)
        # pdb.set_trace()

        assert x.shape[1] == self.input_dim   # 2048
        x = self.conv1(x)  # [1, 1024, 10, 10]

        # fuse all clip, unet skip and transporter
        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)   # check the size of l_input
        x = self.up1(x, im[-2])          # [1, 512, 20, 20]
        x = self.lat_fusion1(x, lat[-6]) # [1, 512, 20, 20]

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])
        x = self.lat_fusion2(x, lat[-5]) # [1, 256, 40, 40]

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])
        x = self.lat_fusion3(x, lat[-4]) # [1, 128, 80, 80]

        # only fuse clip and transporter, lat from transporter
        x = self.layer1(x)               # [1, 64, 160, 160]
        x = self.lat_fusion4(x, lat[-3]) # [1, 64, 160, 160]

        x = self.layer2(x)               # [1, 32, 320, 320]
        x = self.lat_fusion5(x, lat[-2]) # [1, 32, 320, 320]

        x = self.layer3(x)               # [1, 16, 640, 640]
        x = self.lat_fusion6(x, lat[-1]) # [1, 16, 640, 640]

        x = self.conv2(x)                # [1, 1, 640, 640]              

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear') # [1, 1, 320, 320]
        return x
    