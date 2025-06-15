from typing import *

import os
import math
import copy
import time as t

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch import autograd
import torchvision
from torchvision import utils

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


SAVE_PER_TIMES = 100


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
    dim = 512 
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

    def forward(self, code, rect):
        return self.positional_encoding(self.tok_emb(code) + self.reg_emb(rect))


class Generator(nn.Module):

    def __init__(self, vit, emb):
        super().__init__()
        self.encoder = vit
        self.decoder = Decoder(
            dim=Config.dim, 
            num_head=Config.num_head, 
            num_layers=Config.num_layer
        )
        self.emb = emb
        self.output = nn.Linear(Config.dim, 4)

    def forward(self, image, code, rect):
        memory = self.encoder(image)
        embs = self.emb(code, rect)
        outs = self.decoder(embs, memory, tgt_key_padding_mask=(code == 0))
        return self.output(outs).sigmoid()


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
        self.emb = emb
        self.classifier = nn.Sequential(
            nn.Linear(Config.dim * 2, Config.dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.dim, 1),
        )

    def forward(self, image, code, rect):
        image_features = self.encoder(image)[:, 0, :]
        embs = self.emb(code, rect)
        text_features = self.txt_encoder(embs, src_key_padding_mask=(code == 0))[:, 0, :]
        combined = torch.cat([text_features, image_features], dim=1)
        return self.classifier(combined)


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

def create_emb():
    return Embedding()


def count_trainale_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(f"Trainable parameters: {params:,}")
    return params


class WGAN_GP(object):

    def __init__(self, args):
        print("WGAN_GradientPenalty init model.")

        vit_g = create_vit()
        vit_d = vit_g if args.share_vit else create_vit()

        emb_g = create_emb()
        emb_d = emb_g if args.share_emb else create_emb()

        self.G = Generator(vit_g, emb_g)
        self.D = Discriminator(vit_d, emb_d)

        g_params = count_trainale_parameters(self.G)
        d_params = count_trainale_parameters(self.D)
        print(f"Trainable parameters: G={g_params:,}, D={d_params:,}")

        self.check_cuda()  # Check if cuda is available

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = args.batch_size

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10

    def check_cuda(self):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("Device: {}".format(self.device))
        self.D.to(self.device)
        self.G.to(self.device)

    def train(self, train_loader, valid_loader):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                items = self.data.__next__()
                images = items["image"]
                codes = items["code"]
                rects = items["rect"]

                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images, codes, rects)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(self.batch_size, codes.size()[1], 4).to(self.device)

                fake_rects = self.G(images, codes, z)
                d_loss_fake = self.D(images, codes, fake_rects)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                if self.cuda:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        gradient_penalty = self.calculate_gradient_penalty(images, codes, rects.data, fake_rects.data)
                        gradient_penalty.backward()
                else:
                    gradient_penalty = self.calculate_gradient_penalty(images, codes, rects.data, fake_rects.data)
                    gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            while True:
                items = self.data.__next__()
                images = items["image"] 
                codes = items["code"]

                if images.size(0) != self.batch_size:
                    continue

                # train generator
                # compute loss with fake images
                z = torch.randn(self.batch_size, codes.size()[1], 4).to(self.device)
                fake_rects = self.G(images, codes, z)
                g_loss = self.D(images, codes, fake_rects)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
                break

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(125):
                #     samples  = self.data.__next__()
                # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                # #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                # #
                # # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                # if not os.path.exists('training_result_images/'):
                #     os.makedirs('training_result_images/')

                # Valid
                with torch.no_grad():
                    total_ious, count_ious = 0, 0
                    for batch in valid_loader:
                        self.to_device(batch, ["code_len"])
                        images = batch["image"]
                        codes = batch["code"]
                        code_lens = batch["code_len"].long()
                        rects = batch["rect"]

                        z = torch.randn(codes.size()[0], codes.size()[1], 4).to(self.device)
                        samples = self.G(images, codes, z)

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

                # Testing
                time = t.time() - self.t_begin

                info = OrderedDict({
                    'Generator iter': g_iter,
                    'Time': time,
                    'Wasserstein distance': Wasserstein_D.data.cpu(),
                    'Loss D': d_loss.data.cpu(),
                    'Loss G': g_cost.data.cpu(),
                    'Loss D Real': d_loss_real.data.cpu(),
                    'Loss D Fake': d_loss_fake.data.cpu()

                })

                print(" ".join([f"{tag}: {value}" for tag, value in info.items()]))

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, images, codes, real_rects, fake_rects):
        eta = torch.FloatTensor(self.batch_size, 1, 1).uniform_(0, 1).to(self.device)
        eta = eta.expand(self.batch_size, real_rects.size(1), real_rects.size(2))

        interpolated = eta * real_rects + ((1 - eta) * fake_rects)

        # define it to calculate gradient
        interpolated.requires_grad = True

        # calculate probability of interpolated examples
        prob_interpolated = self.D(images, codes, interpolated)

        ones = torch.ones(prob_interpolated.size()).to(self.device)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, 
                                  inputs=interpolated,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def to_device(self, batch, ignores = []):
        for k, v in batch.items():
            if k in ignores:
                continue
            batch[k] = v.to(self.device)

    def get_infinite_batches(self, data_loader):
        while True:
            for _, batch in enumerate(data_loader):
                self.to_device(batch, ["code_len"])
                yield batch

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")