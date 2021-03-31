#!/usr/bin/env python3
"""
Author: David Valencia
Date: 28 - Feb - 2021

Description: *** Evaluate Intro_VAE with Pytorch ***
                 Input Frame + Actions = Target Frame
                 The goal here is to see what the prediction looks like with the test data.
                 Two kinds of predictions are made
                 1) single step prediction
                 2) multiple step prediction
                 Data load on batches from Load_Batch_Data_Pytorch.py, mode=test load the test folder

                 Images_Result_Testing Folder will be created automatically to save Curves, Single Prediction and
                 Multiple Step Prediction

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Frame_Prediction_Intro_VAE_V4_pytorch import IntroVAE
from Load_Batch_Data_Pytorch import load_data_generator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr


def transport_shape(y_prediction, t_target):
    # reshape from (32,3,192,256) to (32,192,256,3)
    y_prediction = np.transpose(y_prediction, (0, 2, 3, 1))
    t_target = np.transpose(t_target, (0, 2, 3, 1))
    return y_prediction, t_target


def calculate_mse_ssim_psnr_batch(y_prediction, t_target):
    # calculate mse and ssim of each bath of images
    y_prediction, t_target = transport_shape(y_prediction, t_target)

    ave_ssim  = []
    ave_psnr  = []
    ave_mse   = []
    for img1, img2 in zip(y_prediction, t_target):
        ssim_single = ssim(img2, img1, multichannel=True)
        psnr_single = psnr(img2, img1)
        mse_single = mse(img2, img1)

        ave_ssim.append(ssim_single)
        ave_psnr.append(psnr_single)
        ave_mse.append(mse_single)

    ssim_batch = np.mean(ave_ssim)
    psnr_batch = np.mean(ave_psnr)
    mse_batch = np.mean(ave_mse)

    return mse_batch, ssim_batch, psnr_batch


def mse_ssim_sequence(y_prediction, t_target, sequence_len):

    y_prediction, t_target = transport_shape(y_prediction, t_target)

    ave_ssim = []
    ave_mse  = []

    for img1, img2 in zip(y_prediction, t_target):
        ssim_single = ssim(img2, img1, multichannel=True)
        mse_single  = mse(img2, img1)
        ave_ssim.append(ssim_single)
        ave_mse.append(mse_single)

    sequence_plot = range(1, sequence_len + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('MSE and SSIM of Multiple Step Prediction')

    ax1.plot(sequence_plot, ave_mse, label='MSE')
    ax1.set_xlabel('Steps Sequence Prediction')
    ax1.legend()
    ax1.grid()

    ax2.plot(sequence_plot, ave_ssim, label='SSIM')
    ax2.set_xlabel('Steps Sequence Prediction')
    ax2.legend()
    ax2.grid()
    plt.savefig(f'./Images_Result_Testing/Curve_MSE_SSIM.png')
    #plt.show()
    plt.close()


def plot_samples(y_prediction, t_target, mode, num_img):
    fig_dir = './Images_Result_Testing'
    os.makedirs(fig_dir, exist_ok=True)  # Create the folder to save images after testing

    y_prediction, t_target = transport_shape(y_prediction, t_target)
    #choose_images = [0, 68, 20, 105, 160, 208, 170]
    choose_images = [200, 25, 122, 75, 160, 230, 170]

    c = 1
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle(f'{mode} Predictions ', fontweight="bold", fontsize=17)

    for i in choose_images:
        #  the target frame
        ax = fig.add_subplot(2, num_img, c)
        plt.imshow(t_target[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # the predicted frame
        ax = fig.add_subplot(2, num_img, c + num_img)
        plt.imshow(y_prediction[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        c = c + 1

    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.025, hspace=0.0)
    fig.savefig(f'./{fig_dir}/Sample_Result_{mode}.png')
    #plt.show()
    plt.close()


def plot_samples_sequence(y_prediction, t_target, mode, num_img):

    y_prediction, t_target = transport_shape(y_prediction, t_target)

    c = 1
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle(f'{mode} Predictions ', fontweight="bold", fontsize=17)

    for i in range(num_img):
        #  the target frame
        ax = fig.add_subplot(2, num_img, c)
        plt.imshow(t_target[i])
        ax.set_title(f"Target {i+1} ", fontsize=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # the predicted frame
        ax = fig.add_subplot(2, num_img, c + num_img)
        plt.imshow(y_prediction[i])
        ax.set_title(f"Predictions {i+1} ", fontsize=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        c = c + 1

    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.025, hspace=0.0)
    fig.savefig(f'./Images_Result_Testing/Sample_Result_{mode}.png')
    plt.show()
    plt.close()


def evaluate_model_single(model, data_input_test, data_target_test, data_actions_test):

    print("single-step prediction")

    avr_mse  = []  # average MSE
    avr_ssim = []  # average SSIM
    avr_psnr = []  # average PSNR

    with torch.no_grad():
        for idx, ((img_in, _), (img_t, _), act) in enumerate(zip(data_input_test, data_target_test, data_actions_test), 1):
            img_predicted, _, _, _ = model(img_in, act)
            print(idx)
            mse_batch, ssim_batch, psnr_batch = calculate_mse_ssim_psnr_batch(img_predicted.numpy(), img_t.numpy())
            avr_mse.append(mse_batch)
            avr_ssim.append(ssim_batch)
            avr_psnr.append(psnr_batch)
            if idx == 1:  # this just take the first bath to plot
                plot_samples(img_predicted.numpy(), img_t.numpy(), 'Single_Step', 7)

        mse_data_test = np.mean(avr_mse)
        ssmi_data_test = np.mean(avr_ssim)
        psnr_data_test = np.mean(avr_psnr)

        print("Average MSE of the whole testing dataset:", mse_data_test)
        print("Average SSIM of the whole testing dataset:", ssmi_data_test)
        print("Average PSNR of the whole testing dataset:", psnr_data_test)


def evaluate_model_sequence(model, data_input_test, data_target_test, data_actions_test, long_sequence):

    print("multiple step prediction")

    x_inp, _ = next(iter(data_input_test))
    y_tar, _ = next(iter(data_target_test))
    a_inp = next(iter(data_actions_test))

    x_inp = x_inp[60:]  # to start from frame 60
    a_inp = a_inp[60:]
    y_tar = y_tar[60:]

    with torch.no_grad():
        image_input_true  = x_inp[0].unsqueeze(0)  # reshape from [3,192,256] to [1,3,192,256]
        predictions = []
        for n in range(long_sequence):
            if n < 1:
                input_img = image_input_true
            action = a_inp[n].unsqueeze(0)
            img_predicted, _, _, _ = model(input_img, action)
            predictions.extend(img_predicted.numpy())
            input_img = img_predicted

    mse_ssim_sequence(predictions, y_tar.numpy(), long_sequence)
    plot_samples_sequence(predictions, y_tar.numpy(), 'Multiple_Steps', 15)


def evaluation_mode():

    z_dim = 32
    batch_size = 256
    long_sequence = 100  # How many steps for using their own predictions as input

    # ======== Load Testing Data ===================== #
    data_input_test, data_target_test, data_actions_test = load_data_generator(batch_size=batch_size, mode='test')

    # ======== Load trained model ===================== #
    model = IntroVAE(z_dim)

    model_saved_path = './Model_Saved/model_intro_vae_completed_after_10000_epochs.pt'
    model.load_state_dict(torch.load(model_saved_path, map_location=torch.device('cpu')))

    #model_saved_path = './Model_Saved/checkpoint_7600_epochs.pt'
    #model.load_state_dict(torch.load(model_saved_path, map_location=torch.device('cpu'))['model_state_dict'])

    # ======== Evaluate single-step prediction ===================== #
    evaluate_model_single(model, data_input_test, data_target_test, data_actions_test)

    # ======== Evaluate multiple-step prediction ===================== #
    evaluate_model_sequence(model, data_input_test, data_target_test, data_actions_test, long_sequence)


if __name__ == '__main__':

    evaluation_mode()
