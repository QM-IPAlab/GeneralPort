import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat

class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_ctx = cfg['train']['coop']['n_ctx']
        ctx_init = cfg['train']['coop']['ctx_init']
        dtype = clip_model.dtype
        ctx_dim = 512
        device = clip_model.device()

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(1, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        context = ctx

        return context


class coop_rn50(CLIPLingUNetLat):
    """ CLIP RN50 with U-Net skip connections and lateral connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(CLIPLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess
        self.ctx_init = cfg['train']['coop']['ctx_init']
        n_ctx = cfg['train']['coop']['n_ctx']

        self._load_clip()
        self._build_decoder()
        self.dtype =  self.clip_rn50.dtype
        # self.prompt_learner = PromptLearner(cfg, self.clip_rn50)
        ctx_dim = 512

        # pdb.set_trace()
        # if self.ctx_init:
        #     ctx_init = self.ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = tokenize(ctx_init).to(self.device)
        #     with torch.no_grad():
        #         embedding = self.clip_rn50.token_embedding(prompt).type(self.dtype)
        #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        # self.register_buffer("token_prefix", embedding[:, :1, :])
        # self.save_ctx = self.ctx.data

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        self.clip_rn50.requires_grad = False
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
        del model

    def encode_text(self, x, context):
        # pdb.set_trace()
        # with torch.no_grad():

        tokens = tokenize([x]).to(self.device)
        x = self.clip_rn50.token_embedding(tokens).type(self.dtype)
        x = torch.nan_to_num(x, nan=0.0)

        context = torch.cat([x[:, :1, :], context], dim=1)
        n_ctx = context.shape[1]
        x = torch.cat([context, x[:, n_ctx :, :]], dim=1)

        text_feat, text_emb = self.clip_rn50.encode_text_for_prompt_learning(x, tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, lat, l):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype  # fp32
        in_shape = x.shape
        x = x[:,:3]  # select RGB  [1, 3, 320, 320]
        x, im = self.encode_image(x)  # [1, 2048, 10, 10], [1, 32, 160, 160]// [1, 2048, 12, 7], [1, 32, 192, 112] // [36, 2048, 2, 2], [36, 32, 32, 32]
        x = x.to(in_type)  # [1, 2048, 10, 10]

        l = self.ctx_init + " " + l
        # pdb.set_trace()
        context = self.ctx
        context = context.unsqueeze(0).expand(1, -1, -1)
        # context = self.prompt_learner()

        l_enc, l_emb, l_mask = self.encode_text(l, context)  # [1, 1024], [1, 77, 512], [1, 77]
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)  # (1, 1024) tensor([[nan, nan, nan,  ..., nan, nan, nan]], device='cuda:0'

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
    