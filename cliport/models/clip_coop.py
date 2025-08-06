import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb

from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up, DoubleConv
from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat

from cliport.models.core import clip
from cliport.models.core.clip import build_model, load_clip, tokenize
from cliport.utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def uppatching(features, patch_size, img_size, output_size):

    batch_size, num_patches, embed_dim = features.shape
    grid_size = int(img_size / patch_size)  # Number of patches per dimension

    # Reshape to 2D grid: [batch_size, grid_size, grid_size, embed_dim]
    features_2d = features.view(batch_size, grid_size, grid_size, embed_dim)

    # Permute to bring embed_dim to channels dimension: [batch_size, embed_dim, grid_size, grid_size]
    features_2d = features_2d.permute(0, 3, 1, 2)

    # Upsample the feature map
    # upsampled_features = F.interpolate(features_2d, scale_factor=upsampling_factor, mode='bilinear', align_corners=False)
    upsampled_features = F.interpolate(features_2d, size=(output_size[0], output_size[1]), mode='bilinear', align_corners=False)

    return upsampled_features


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):
        with torch.no_grad():
            x = prompts + self.positional_embedding.type(self.dtype) # (1, 77, 512)+(77, 512)=(1, 77, 512)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            text_emb = self.ln_final(x).type(self.dtype)  # (1, 77, 512)

            text_feat = x[torch.arange(text_emb.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # (1, 512)
            text_mask = torch.where(tokenized_prompts==0, tokenized_prompts, 1) # [1, 77]

        return text_feat, text_emb, text_mask

class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, device):
        super().__init__()

        # pdb.set_trace()
        self.clip_model = clip_model
        n_ctx = cfg['train']['cocoop']['n_ctx']
        ctx_init = cfg['train']['cocoop']['ctx_init']  # context initialization
        self.dtype = clip_model.dtype  # fp16
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        vis_dim = clip_model.visual.output_dim         # 512
        clip_imsize = clip_model.visual.input_resolution # 224
        self.device = next(clip_model.parameters()).device
        self.clip_model = clip_model

        # context vectors
        if ctx_init: 
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx_init = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)   # 上下文向量 [4, 512]

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim , vis_dim // 8)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 8, ctx_dim))
        ]))

        if cfg['train']['cocoop']['prec'] == "fp16":
            self.meta_net.half()

        # pdb.set_trace()
        self.n_ctx = n_ctx

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS [1, 1, 512]
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])

    def tokenized_prompts(self, instruction):
        # input initial instruction sentence
        # ouput the prompts token in current time
        # looking for the end-of-token

        # pdb.set_trace()
        ins_lens = len(_tokenizer.encode(instruction))
        prompts = self.ctx_init + " " + instruction + "."
        tokenized_prompts = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS [1, 1, 512]
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])

        self.current_ins_lens = ins_lens 

        return tokenized_prompts
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        dim0 = ctx.shape[0]
        assert dim0 == 1
        # print(dim0)
        # if dim0 > 1:
        #     prefix = prefix.repeat(dim0, 1, 1)
        #     suffix = suffix.repeat(dim0, 1, 1)
        # print(prefix.shape, ctx.shape, suffix.shape)
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1
        )

        return prompts   # [1, 77, 512]

    def forward(self, im_features):  # [1, 512]    
        prefix = self.token_prefix   # [1, 1, 512]
        suffix = self.token_suffix   # [1, 72, 512]
        ctx = self.ctx                     # (n_ctx, ctx_dim) [4, 512]
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)       [1, 1, 512]
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)       [1, 4, 512]
        ctx = ctx + bias
        # ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)   [1, 4, 512]
        ctx = ctx.mean(dim=0, keepdim=True)

        # ctx = ctx_shifted.unsqueeze(0)     # [1, 1, 4, 512]
        prompt = self.construct_prompts(ctx, prefix, suffix)  # (batch, n_tkn, ctx_dim)   [1, 77, 512]

        return prompt

class coop(CLIPLingUNetLat):
    # CLIP ViT-B/16 without U-net connection
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)
        # self.prompt_learner = PromptLearner(cfg, self.clip_model, device)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)

    def _load_clip(self):
        model, _ = load_clip("ViT-B/16", device=self.device)
        self.clip_model = build_model(model.state_dict()).to(self.device)
        # pdb.set_trace()
        del model

    def encode_image(self, img):  #[1, 3, 224, 224]
        with torch.no_grad(): 
            cls, emb = self.clip_model.visual(img)
        return cls, emb # [1, 512], [1, 196, 512]
    
    def _build_decoder(self):
        # language
        # self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 512
        # self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim // 4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        # self.up1 = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     DoubleConv(1024, 512)
        # ) 
        # self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)

        self.up2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DoubleConv(512, 256)
        ) 
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            DoubleConv(256, 128)
        )
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

    def forward(self, x, lat, l):
        # pdb.set_trace()
        x = self.preprocess(x, dist='clip')
        # pdb.set_trace()
        in_type = x.dtype  # fp32
        in_shape = x.shape
        x = x[:,:3]  # select RGB  [1, 3, 320, 320]
        x = x.half()
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # [1, 3, 224, 224]
        cls, im_emb = self.encode_image(x)  # [1, 512, 14, 14]
        im_emb = im_emb.to(in_type)
        x = uppatching(im_emb, 16, 224, [in_shape[-2] // 16, in_shape[-1] // 16])  # [1, 512, 20, 20]

        # tokenized_prompts = self.prompt_learner.tokenized_prompts(l)
        # prompt_instruction = self.prompt_learner(cls)  # embedding of the whole [X][X][X][X][instruction_embedding]
        tokenized_prompts = clip.tokenize(l).to(self.device)
        instruction = self.clip_model.token_embedding(tokenized_prompts).type(in_type)
        l_enc, l_emb, l_mask = self.text_encoder(instruction, tokenized_prompts)

        # l_enc, l_emb, l_mask = self.text_encoder(prompt_instruction, tokenized_prompts) # [1, 1024], [1, 77, 512], [1, 77]
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        # pdb.set_trace()
        assert x.shape[1] == self.input_dim // 4
        # with torch.no_grad():
        x = self.conv1(x)  # [1, 512, 20, 20]

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2) # (1, 512, 20, 20)
        x = self.up2(x)                  # [1, 256, 40, 40]
        x = self.lat_fusion2(x, lat[-5]) # [1, 256, 40, 40]

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3) # [1, 256, 40, 40]
        x = self.up3(x)                  # [1, 128, 80, 80]
        x = self.lat_fusion3(x, lat[-4]) # [1, 128, 80, 80]

        x = self.layer1(x)               # [1, 64, 160, 160]
        x = self.lat_fusion4(x, lat[-3]) # [1, 64, 160, 160]

        x = self.layer2(x)               # [1, 32, 320, 320]
        x = self.lat_fusion5(x, lat[-2]) # [1, 32, 320, 320]

        x = self.layer3(x)               # [1, 16, 640, 640]
        x = self.lat_fusion6(x, lat[-1]) # [1, 16, 640, 640]

        x = self.conv2(x)                # [1, 1, 640, 640]              

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear') # [1, 1, 320, 320]
        return x

