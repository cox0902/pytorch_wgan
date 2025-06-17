from typing import *

import math
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision

from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange



def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SPT(nn.Module):
    def __init__(self, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim,
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # print(x.shape)  # (32, 256, 512)
 
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size) 
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Decoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, dim, num_head, num_layers, norm=None):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_head, batch_first = True)
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class Config:

    vocab_size = 90
    image_size = 256
    patch_size = 16 
    dim = 256  # 512 
    num_layer = 6
    num_head = 8 
    mlp_dim = 1024 
    dropout = 0.1 
    emb_dropout = 0.1


class Embedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.tok_emb = TokenEmbedding(Config.vocab_size, Config.dim)
        self.reg_emb = nn.Linear(4, Config.dim)
        self.positional_encoding = PositionalEncoding(Config.dim, dropout=Config.dropout)
        # self.dropout = nn.Dropout(Config.emb_dropout)

    def forward(self, code, rect):
        return self.positional_encoding(self.tok_emb(code) + self.reg_emb(rect))
        # return self.dropout(self.tok_emb(code) + self.reg_emb(rect))


class MaskEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.tok_emb = TokenEmbedding(Config.vocab_size, Config.dim)
        self.reg_emb = nn.Linear(16 * 16, Config.dim)
        self.positional_encoding = PositionalEncoding(Config.dim, dropout=Config.dropout)
        # self.dropout = nn.Dropout(Config.emb_dropout)

    def forward(self, code, mask):
        return self.positional_encoding(self.tok_emb(code) + self.reg_emb(mask.view(-1, 16 * 16)))
        # return self.dropout(self.tok_emb(code) + self.reg_emb(mask.view(-1, 256 * 256)))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum(q, k, "b i d, b j d -> b i j") * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, "b i j, b j d -> b i d")
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)

    

class Generator(nn.Module):

    def __init__(self, vit, emb):
        super().__init__()
        self.encoder = vit
        # self.decoder = Decoder(
        #     dim=Config.dim, 
        #     num_head=Config.num_head, 
        #     num_layers=Config.num_layer
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.dim, 
            nhead=Config.num_head, 
            batch_first=True
        )
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, Config.num_layer)
        self.projector = CrossAttention(Config.dim, Config.dim, dim_head=Config.dim, heads=Config.num_head, 
                                        dropout=Config.dropout)
        self.emb = emb
        self.output = nn.Sequential(
            nn.Linear(Config.dim, 4)
        )

    def forward(self, image, code, rect):
        memory = self.encoder(image)
        # print(memory.shape)
        embs = self.emb(code, rect.sigmoid())
        # outs = self.decoder(embs, memory, tgt_key_padding_mask=(code == 0))
        outs = self.txt_encoder(embs, src_key_padding_mask=(code == 0))
        outs = self.projector(outs, memory)
        outs = self.output(outs).sigmoid()
        # print(outs.shape, code.shape)
        mask = code <= 7
        mask = mask.unsqueeze(-1)
        outs = outs.masked_fill(mask, 0)
        # print(outs)
        # print(rect)
        # assert False
        return outs


class MaskGenerator(nn.Module):

    def __init__(self, vit, emb):
        super().__init__()
        self.encoder = vit
        self.decoder = Decoder(
            dim=Config.dim, 
            num_head=Config.num_head, 
            num_layers=Config.num_layer
        )
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=Config.dim, 
        #     nhead=Config.num_head, 
        #     batch_first=True
        # )
        # self.txt_encoder = nn.TransformerEncoder(encoder_layer, Config.num_layer)
        # self.projector = CrossAttention(Config.dim, Config.dim, dim_head=Config.dim, heads=Config.num_head, 
        #                                 dropout=Config.dropout)
        self.emb = emb
        # self.output = nn.Sequential(
        #     nn.Linear(Config.dim, 4)
        # )
        self.generator = nn.Sequential(
            nn.Linear(Config.dim, 16 * 16),
            nn.Tanh()
        )

    def forward(self, image, code, rect):
        memory = self.encoder(image)
        embs = self.emb(code, rect.sigmoid())
        outs = self.decoder(embs, memory, tgt_key_padding_mask=(code == 0))
        # outs = self.txt_encoder(embs, src_key_padding_mask=(code == 0))
        # outs = self.projector(outs, memory)
        # outs = self.output(outs).sigmoid()
        outs = self.generator(outs)
        # print(outs.shape, code.shape)
        mask = code <= 7
        mask = mask.unsqueeze(-1)
        outs = outs.masked_fill(mask, 0)
        # print(outs)
        # print(rect)
        # assert False
        return outs


class Discriminator(nn.Module):

    def __init__(self, vit, emb):
        super().__init__()
        self.encoder = vit
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.dim, 
            nhead=Config.num_head, 
            batch_first=True
        )
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, Config.num_layer)
        self.projector = CrossAttention(Config.dim, Config.dim, dim_head=Config.dim, heads=Config.num_head, 
                                        dropout=Config.dropout)
        self.emb = emb
        # self.classifier = nn.Sequential(
        #     nn.Linear(Config.dim * 2, Config.dim),
        #     nn.ReLU(),
        #     nn.Dropout(Config.dropout),
        #     nn.Linear(Config.dim, 1),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(Config.dim, 1),
            # nn.Tanh()
        )

    def forward(self, image, code, rect):
        image_features = self.encoder(image)  # [:, 0, :]
        # print(image_features.shape)
        embs = self.emb(code, rect)
        text_features = self.txt_encoder(embs, src_key_padding_mask=(code == 0))
        outs = self.projector(text_features, image_features)
        # combined = torch.cat([text_features, image_features], dim=1)
        return self.classifier(outs[:, 0, :])


class MaskDiscriminator(nn.Module):

    def __init__(self, vit, emb):
        super().__init__()
        self.encoder = vit
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.dim, 
            nhead=Config.num_head, 
            batch_first=True
        )
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, Config.num_layer)
        self.projector = CrossAttention(Config.dim, Config.dim, dim_head=Config.dim, heads=Config.num_head, 
                                        dropout=Config.dropout)
        self.emb = emb
        self.classifier = nn.Sequential(
            nn.Linear(Config.dim, Config.dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.dim // 2, 1),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(Config.dim, 1),
        # )

    def forward(self, image, code, rect):
        image_features = self.encoder(image)  # [:, 0, :]
        embs = self.emb(code, rect)
        text_features = self.txt_encoder(embs, src_key_padding_mask=(code == 0))
        outs = self.projector(text_features, image_features)
        # combined = torch.cat([text_features, image_features], dim=1)
        return self.classifier(outs[:, 0, :])
    

def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk

    # return intsctk, unionk
    return intsctk / (unionk + eps)


def create_vit():
    encoder = ViT(
        image_size=Config.image_size,
        patch_size=Config.patch_size,
        dim=Config.dim,
        depth=Config.num_layer,
        heads=Config.num_head,
        mlp_dim=Config.mlp_dim,
        dropout=Config.dropout,
        emb_dropout=Config.emb_dropout
    )
    return encoder


def create_res():
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # pretrained ImageNet ResNet-50
    modules = list(resnet.children())[:-2]

    resnet = nn.Sequential(
        *modules,
        nn.AdaptiveAvgPool2d((16, 16)),
        Rearrange("b c w h -> b c (w h)")
        )
    return resnet


def create_emb():
    return Embedding()


def create_model(args):
    if args.vis == "vit":
        vit_g = create_vit()
        vit_d = vit_g if args.share_vis else create_vit()
    else:
        vit_g = create_res()
        vit_d = vit_g if args.share_vis else create_res()


    emb_g = create_emb()
    emb_d = emb_g if args.share_emb else create_emb()

    return Generator(vit_g, emb_g), Discriminator(vit_d, emb_d)


def count_trainale_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(f"Trainable parameters: {params:,}")
    return params


def evaluate(model, valid_loader, device):
    with torch.no_grad():
        total_ious, count_ious = 0, 0
        for batch in valid_loader:
            images = batch["image"].to(device)
            codes = batch["code"].to(device)
            code_lens = batch["code_len"].long()
            rects = batch["rect"].to(device)

            z = torch.randn(codes.size()[0], codes.size()[1], 4).to(device)
            samples = model(images, codes, z)

            t_label = pack_padded_sequence(codes, code_lens, batch_first=True, enforce_sorted=False).data
            t_rects = pack_padded_sequence(rects, code_lens, batch_first=True, enforce_sorted=False).data
            p_rects = pack_padded_sequence(samples, code_lens, batch_first=True, enforce_sorted=False).data

            mask = t_label > 7
            t_rects = t_rects[mask]
            p_rects = p_rects[mask]
            # print(code_lens.sum(), t_rects.shape)

            p_rects = torchvision.ops.box_convert(p_rects, "cxcywh", "xyxy")
            t_rects = torchvision.ops.box_convert(t_rects, "cxcywh", "xyxy")

            ious = compute_iou(t_rects, p_rects)
            # print(ious.shape)
            total_ious += ious.sum().item()
            count_ious += ious.size(0)

        print(f"  mean iou = {total_ious / count_ious}")
