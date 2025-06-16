from typing import *

import os

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

from .components import create_vit, create_emb, Generator, Discriminator, count_trainale_parameters


SAVE_PER_TIMES = 1


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
        self.normal_train = args.normal_train

    def check_cuda(self):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("Device: {}".format(self.device))
        self.D.to(self.device)
        self.G.to(self.device)

    def train(self, train_loader, validator):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        # self.data = self.get_infinite_batches(train_loader)

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
            # for d_iter in range(self.critic_iter):
            for d_iter, items in enumerate(train_loader):
                self.D.zero_grad()

                # items = self.data.__next__()
                self.to_device(items, ["code_len"])
                images = items["image"]
                codes = items["code"]
                rects = items["rect"]

                # Check for batch to have full batch_size
                # if (images.size()[0] != self.batch_size):
                #     continue

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images, codes, rects)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(codes.size()[0], codes.size()[1], 4).to(self.device)

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
                print(f'  Discriminator iteration: {d_iter}/{len(train_loader)}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # while True:
            for s_iter, items in enumerate(train_loader):
                # items = self.data.__next__()
                self.to_device(items)
                images = items["image"] 
                codes = items["code"]

                # if images.size(0) != self.batch_size:
                #     continue

                # train generator
                # compute loss with fake images
                z = torch.randn(codes.size()[0], codes.size()[1], 4).to(self.device)
                fake_rects = self.G(images, codes, z)
                g_loss = self.D(images, codes, fake_rects)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                print(f'Generator iteration: {s_iter}/{len(train_loader)}, g_loss: {g_loss}')
                # break

            if self.normal_train:
                for s_iter, items in enumerate(train_loader):
                    # items = self.data.__next__()
                    self.to_device(items, ["code_len"])
                    images = items["image"] 
                    codes = items["code"]
                    rects = items["rect"]
                    code_lens = items["code_len"].long()

                    # train generator
                    # compute loss with fake images
                    z = torch.randn(codes.size()[0], codes.size()[1], 4).to(self.device)
                    fake_rects = self.G(images, codes, z)

                    t_label = pack_padded_sequence(codes, code_lens, batch_first=True, enforce_sorted=False).data
                    t_rects = pack_padded_sequence(rects, code_lens, batch_first=True, enforce_sorted=False).data
                    p_rects = pack_padded_sequence(fake_rects, code_lens, batch_first=True, enforce_sorted=False).data

                    mask = t_label > 7
                    t_rects = t_rects[mask]
                    p_rects = p_rects[mask]
                    # print(code_lens.sum(), t_rects.shape)

                    p_rects = torchvision.ops.box_convert(p_rects, "cxcywh", "xyxy")
                    t_rects = torchvision.ops.box_convert(t_rects, "cxcywh", "xyxy")
                
                    n_loss = torchvision.ops.generalized_box_iou_loss(p_rects, t_rects, reduction="mean")
                    n_loss.backward()
                    self.g_optimizer.step()
                    print(f'Normal iteration: {s_iter}/{len(train_loader)}, n_loss: {n_loss}')

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
                validator(model=self.G, device=self.device)

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

                print(", ".join([f"{tag}: {value}" for tag, value in info.items()]))

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