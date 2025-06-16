from typing import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from torchvision import utils

from .components import create_model, count_trainale_parameters


SAVE_PER_TIMES = 1


class WGAN_CP(object):
    def __init__(self, args):
        print("WGAN_CP init model.")

        self.G, self.D = create_model(args)

        g_params = count_trainale_parameters(self.G)
        d_params = count_trainale_parameters(self.D)
        print(f"Trainable parameters: G={g_params:,}, D={d_params:,}")
        
        # check if cuda is available
        self.check_cuda()

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = args.batch_size
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)

        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5


    def check_cuda(self):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("Device: {}".format(self.device))
        self.D.to(self.device)
        self.G.to(self.device)

    def train(self, train_loader, validator):
        self.t_begin = t.time()
        #self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.FloatTensor([1]).to(self.device)
        mone = one * -1

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                items = self.data.__next__()
                
                images = items["image"].to(self.device)
                codes = items["code"].to(self.device)
                rects = items["rect"].to(self.device)

                # Check for batch to have full batch_size
                # if (images.size()[0] != self.batch_size):
                #     continue

                z = torch.rand((self.batch_size, codes.size()[1], 4)).to(self.device)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images, codes, rects)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                # Train with fake images
                fake_rects = self.G(images, codes, z)
                d_loss_fake = self.D(images, codes, fake_rects)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(mone)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.detach().cpu().item()}, loss_real: {d_loss_real.detach().cpu().item()}')


            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            items = self.data.__next__()
            images = items["image"].to(self.device)
            codes = items["code"].to(self.device)
            rects = items["rect"].to(self.device)
            
            # Train generator
            # Compute loss with fake images
            z = torch.randn(self.batch_size, codes.size()[1], 4).to(self.device)
            fake_rects = self.G(images, codes, z)
            g_loss = self.D(images, codes, fake_rects)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.detach().cpu().item()}')

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(10):
                #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                #
                # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                validator(model=self.G, device=self.device)

                # Testing
                time = t.time() - self.t_begin
                #print("Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                time = t.time() - self.t_begin

                info = OrderedDict({
                    'Generator iter': g_iter,
                    'Time': time,
                    'Wasserstein distance': Wasserstein_D.detach().cpu().item(),
                    'Loss D': d_loss.detach().cpu().item(),
                    'Loss G': g_cost.detach().cpu().item(),
                    'Loss D Real': d_loss_real.detach().cpu().item(),
                    'Loss D Fake': d_loss_fake.detach().cpu().item()

                })

                print(", ".join([f"{tag}: {value}" for tag, value in info.items()]))

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

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

    def get_infinite_batches(self, data_loader):
        while True:
            for i, images in enumerate(data_loader):
                yield images


    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between two noise (z1, z2).
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