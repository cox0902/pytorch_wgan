from typing import *

from functools import partial

from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.components import evaluate
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP


def main(args):

    if args.model == "CP":
        model = WGAN_CP(args)
    else:
        model = WGAN_GP(args)

    # Load datasets to train and test loaders
    train_loader, valid_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if not args.test_only:
        model.train(train_loader, partial(evaluate, valid_loader=valid_loader))

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        for i in range(50):
           model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)