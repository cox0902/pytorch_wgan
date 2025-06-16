from typing import *

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--test-only', action="store_true", default=False)
    parser.add_argument('--model', type=str, default="GP", help='name')

    parser.add_argument('--image-path', required=True, help='path to image')
    parser.add_argument('--code-path', required=True, help='path to code')
    parser.add_argument('--split-path', required=True, help='path to split')

    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='The size of batch')

    parser.add_argument('--share-vit', action="store_true", default=False)
    parser.add_argument('--share-emb', action="store_true", default=False)
    parser.add_argument('--normal-train', action="store_true", default=False)

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')

    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    return args