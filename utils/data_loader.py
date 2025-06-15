from typing import *

import numpy as np

import torch.utils.data as data_utils

from .dataset import ImageCodeDataset


def get_data_loader(args):

    split = np.load(args.split_path)
    split_train = split["train"]
    split_valid = split["valid"]
    split_test = split["test"]

    train_dataset = ImageCodeDataset(args.image_path, 
                                     args.code_path, 
                                     split_train)
    train_dataset.summary("Train:")

    valid_dataset = ImageCodeDataset(args.image_path, 
                                     args.code_path, 
                                     split_valid)
    valid_dataset.summary("Valid:")

    test_dataset = ImageCodeDataset(args.image_path, 
                                    args.code_path, 
                                    split_test)
    test_dataset.summary("Test:")
    
    # Check if everything is ok with loading datasets
    assert train_dataset
    assert valid_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader