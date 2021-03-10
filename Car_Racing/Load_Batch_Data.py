#!/usr/bin/env python3

"""
Author      : David Valencia
Date        : March / 2021
Description :

"""

import torch
import numpy as np
from torchvision import datasets, transforms


def load_data_generator(batch_size=32, mode='train'):
    input_path = f'./Preprocessed_Data/{mode}/Input_images/'
    input_images_folder = datasets.ImageFolder(input_path, transform=transforms.Compose([transforms.ToTensor()]))
    input_img_data_loader = torch.utils.data.DataLoader(input_images_folder, batch_size=batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True)

    target_path = f'./Preprocessed_Data/{mode}/Target_images/'
    target_images_folder = datasets.ImageFolder(target_path, transform=transforms.Compose([transforms.ToTensor()]))
    target_img_data_loader = torch.utils.data.DataLoader(target_images_folder, batch_size=batch_size,
                                                         shuffle=False, num_workers=8, pin_memory=True)

    action_path = f'./Preprocessed_Data/{mode}/data_actions.npy'
    actions_file = torch.from_numpy(np.load(action_path))
    input_act_data_loader = torch.utils.data.DataLoader(actions_file, batch_size=batch_size,
                                                        shuffle=False, num_workers=8, pin_memory=True)

    return input_img_data_loader, target_img_data_loader, input_act_data_loader
