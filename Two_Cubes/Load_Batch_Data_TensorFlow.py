#!/usr/bin/env python3
"""
Author: David Valencia
Date: 10 - Feb - 2021

Description:
                    ***    custom data generation for data-generator in Keras   ***

             This scrip shows how to load data with a custom data generation function
             This avoid run out of memory when loading a big data set
             The actions are also included here
             This is using tensorflow-keras
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def generated_batch_actions(file, batch_size):
    actions = np.load(file)
    while True:
        for i in range(0, len(actions), batch_size):
            yield actions[i:i + batch_size]


def single_generator_multiple(batch_size=32, mode='test'):
    # This return a single "Tensor" witch includes 3 batches one for inp other for target other for actions

    path_actions = f'./Preprocessed_Data/{mode}/data_actions.npy'
    gen_act = generated_batch_actions(path_actions, batch_size)

    generator = ImageDataGenerator(rescale=1. / 255)

    path1 = f'./Preprocessed_Data/{mode}/Input_images/'
    gen1 = generator.flow_from_directory(path1, shuffle=False, class_mode=None, batch_size=batch_size, target_size=(192, 256))

    path2 = f'./Preprocessed_Data/{mode}/Target_images/'
    gen2 = generator.flow_from_directory(path2, shuffle=False, class_mode=None, batch_size=batch_size, target_size=[192, 256])

    while True:
        x = gen1.next()
        y = gen2.next()
        a = next(gen_act)
        yield x, a, y


def triple_generator_multiple(batch_size=32, mode='test'):
    # This return a three "Tensors" separately one for inp other for target other for actions

    path_actions = f'./Preprocessed_Data/{mode}/data_actions.npy'
    gen_actions = generated_batch_actions(path_actions, batch_size)

    generator = ImageDataGenerator(rescale=1. / 255)

    path1 = f'./Preprocessed_Data/{mode}/Input_images/'
    gen_inputs = generator.flow_from_directory(path1, shuffle=False, class_mode=None, batch_size=batch_size, target_size=[192, 256])

    path2 = f'./Preprocessed_Data/{mode}/Target_images/'
    gen_targets = generator.flow_from_directory(path2, shuffle=False, class_mode=None, batch_size=batch_size, target_size=[192, 256])

    return gen_inputs, gen_targets, gen_actions


'''
steps_epochs = 332
data_train = single_generator_multiple(batch_size=32, mode='train')
for _ in range(0, 2):
    print("--------------------")
    for step, data in zip(range(steps_epochs), data_train):
        input_img, input_actions, target_images = data
        print(input_img.shape, input_actions.shape, target_images.shape, step)
'''


'''

img_input, img_target, vector_actions = triple_generator_multiple(batch_size=32, mode='test')
steps_epoch = len(img_input)
for j in range(1, 6):
    print("------------------")
    for step, img_in, img_t, act in zip(range(1, steps_epoch + 1), img_input, img_target, vector_actions):
        print(img_in.shape, img_t.shape, act.shape, step)
'''







