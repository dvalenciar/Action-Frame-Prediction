#!/usr/bin/env python3

"""
Author      : David Valencia
Date        : 10/ Feb/2021
Description :
              ***Note: What is new here is that, A function sorts each  preprocessed image
              ***READY FOR IMAGES REGENERATION*** in the correct folder as input_image, target_image and action

              ---> Also de normalization is respect to 255
              ---> No NPZ files for images are generated here
              ---> Data set contains around 100 episodes, manually checked
              ---> From Image Size to (192,256,3)
              ---> Action is one-hot encoded here

              Here created the function that will help in the training process
              Load and Preprocess data (image and actions) Using PIL

              This library helps to preprocess the data(resize, normalization, sort, etc)
              A list with the images/actions (inputs) and images(target) in the correct order and format
              is created and write in folders.

              Result = Preprocessed_Data folder with train, test folder, each with input, target and actions ready
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import image as mat_img


def load_actions(data_directory):
    episode_directories = sorted([data_directory + p for p in os.listdir(data_directory)])
    number_episodes = len(episode_directories)
    inputs_action = []
    for i in tqdm(range(number_episodes)):
        episode = episode_directories[i]  # path of each episode e.g. /DataSet/train/000000
        action_file = episode + "/action.txt"  # actions file of each episode e.g. train/000000/action.txt
        action_vectors = np.loadtxt(action_file)  # read each action.txt file
        n_action = len(action_vectors)  # number of actions in each action.txt e.g. 100
        for a in range(n_action):
            single_action  = action_vectors[a]  # action as a list
            one_hot_vector = encoding_action(single_action)  # action one-hot encoded
            inputs_action.append(one_hot_vector)
    return inputs_action


def encoding_action(action_vector):
    joints = [0, 1, 2]
    num_unique_actions = 2
    one_hot_vector = np.zeros(num_unique_actions * len(joints))
    for joint_index in range(len(joints)):
        joint = joints[joint_index]
        if action_vector[joint] < 0:
            one_hot_vector[joint_index * num_unique_actions] = 1
            return one_hot_vector
        if action_vector[joint] > 0:
            one_hot_vector[joint_index * num_unique_actions + 1] = 1
            return one_hot_vector


def load_data_input_target_pil(data_directory):
    resize_height = 192
    resize_width  = 256

    episode_directories = sorted([data_directory + p for p in os.listdir(data_directory)])
    number_episodes = len(episode_directories)
    data_input = []
    data_target = []

    for i in tqdm(range(number_episodes)):
        episode = episode_directories[i]
        frames = sorted([f for f in os.listdir(episode) if f.endswith(".png")])

        for f in range(len(frames)):
            img = Image.open(episode + "/" + frames[f])
            img = img.resize((resize_width, resize_height))  # resize the images (640x480)==>(192x256)
            # img.thumbnail((256, 256))
            img = np.asarray(img)  # when using PIL convert to array is necessary to work with images
            img = img.astype('float32')  # convert from integers to floats 32
            #pix_max = img.max()  # the highest pixel in each image (around 138~144)
            #img = img / pix_max  # normalize the image [0~1]
            img = img / 255.0  # normalize the image [0~1]
            frames[f] = img
        frames_input = frames[0:-1]
        frames_target = frames[1:]
        data_input.extend(frames_input)  # should be extend because adding each frame to the list and extending the list
        data_target.extend(frames_target)
    # keep in mind that data_input and data_target are list type
    return data_input, data_target


def save_new_data(input_images, target_images, input_actions, typo):

    print(f"Writing the {typo} images, please wait....")
    print("\n")

    # -------------------------------------
    i = 0
    for img in tqdm(input_images):
        mat_img.imsave(f'./Preprocessed_Data/{typo}/Input_images/images_folder/{i:08d}.png', img)
        i += 1
    #-------------------------------------
    j = 0
    for img in tqdm(target_images):
        mat_img.imsave(f'./Preprocessed_Data/{typo}/Target_images/images_folder/{j:08d}.png', img)
        j += 1
    # -------------------------------------
    np.save(f'./Preprocessed_Data/{typo}/data_actions', input_actions)
    file = open(f'./Preprocessed_Data/{typo}/data_actions.txt', 'w')
    for action in tqdm(input_actions):
        file.write(f'{action}\n')
    file.close()
    #-------------------------------------


if __name__ == '__main__':

    mode = 'Train'

    if mode == 'Train':
        print("Resizing and normalizing train images, please wait...")

        data_set_path = './DataSet/train/'

        input_frames, target_frames = load_data_input_target_pil(data_set_path)
        input_action = np.float32(load_actions(data_set_path))
        save_new_data(input_frames, target_frames, input_action, "train")

    elif mode == 'Test':
        print("Resizing and normalizing test images, please wait...")

        data_set_path = './DataSet/test/'

        input_frames, target_frames = load_data_input_target_pil(data_set_path)
        input_action = np.float32(load_actions(data_set_path))
        save_new_data(input_frames, target_frames, input_action, "test")

    else:
        print("Select a correct mode to generated the train o test data preprocessed file")


