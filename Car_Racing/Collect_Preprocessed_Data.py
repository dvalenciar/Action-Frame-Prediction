#!/usr/bin/env python3

"""
Author      : David Valencia
Date        : March / 2021
Description :

"""

import os
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing
from PIL import Image
from tqdm import tqdm
from matplotlib import image as mat_img


def crop_normalize_observation(observation):
    crop_frame = Image.fromarray(observation[:83, :, :], mode='RGB')
    img_resize = crop_frame.resize((64, 64), Image.BILINEAR)
    img = np.asarray(img_resize)
    img = img.astype('float32')  # convert from integers to floats 32
    img = img / 255.0  # normalize the image [0~1]
    return img


def generated_action():
    steer = np.random.uniform(low=-1.0, high=1.0)
    throttle = np.random.uniform(low=0.0, high=1.0)
    break_pedal = np.random.uniform(low=0.0, high=0.1)

    action_vector = [steer, throttle, break_pedal]
    return np.float32(action_vector)


def generated_frames(n_episodes, n_steps):
    input_frame  = []
    target_frame = []
    input_action = []

    env = CarRacing()
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        position = np.random.randint(len(env.track))  # to start the car' position in a different place every episode
        env.car = Car(env.world, *env.track[position][1:4])  # Here * is passing values, no multiplication
        state = crop_normalize_observation(state)

        for steps in range(n_steps):
            #action = env.action_space.sample()
            action = generated_action()
            new_state, reward, done, info = env.step(action)
            env.viewer.window.dispatch_events()  # Avoiding possibles corrupt environment observations
            new_state = crop_normalize_observation(new_state)

            if steps >= 50:
                # with > 50; I discard the first 50 states since they are useless, no info there
                input_frame.append(state)
                target_frame.append(new_state)
                input_action.append(action)
            state = new_state
    env.close()
    return input_frame, target_frame, input_action


def save_new_data(input_images, target_images, input_actions, typo):
    print(f"Writing the {typo} images, please wait....")
    print("\n")

    # -------------------------------------
    fig_dir = f'./Preprocessed_Data/{typo}/Input_images/images_folder'
    os.makedirs(fig_dir, exist_ok=True)  # Create the folder to save images
    i = 0
    for img in tqdm(input_images):
        mat_img.imsave(f'./{fig_dir}/{i:08d}.png', img)
        i += 1

    #-------------------------------------
    fig_dir_target = f'./Preprocessed_Data/{typo}/Target_images/images_folder'
    os.makedirs(fig_dir_target, exist_ok=True)  # Create the folder to save images
    j = 0
    for img in tqdm(target_images):
        mat_img.imsave(f'./{fig_dir_target}/{j:08d}.png', img)
        j += 1
    # -------------------------------------
    np.save(f'./Preprocessed_Data/{typo}/data_actions', input_actions)
    file = open(f'./Preprocessed_Data/{typo}/data_actions.txt', 'w')
    for action in tqdm(input_actions):
        file.write(f'{action}\n')
    file.close()
    #-------------------------------------


if __name__ == '__main__':

    mode = 'None'

    if mode == 'Train':
        print("Collecting training images, please wait...")

        N_STEPS = 350  # 350
        N_EPISODES = 100  # 100

        data_input_frame, data_target_frame, data_input_action = generated_frames(n_episodes=N_EPISODES, n_steps=N_STEPS)
        save_new_data(data_input_frame, data_target_frame, data_input_action, "train")

    elif mode == 'Test':
        print("Collecting testing images, please wait...")

        N_STEPS = 350
        N_EPISODES = 10

        data_input_frame, data_target_frame, data_input_action = generated_frames(n_episodes=N_EPISODES, n_steps=N_STEPS)
        save_new_data(data_input_frame, data_target_frame, data_input_action, "test")

