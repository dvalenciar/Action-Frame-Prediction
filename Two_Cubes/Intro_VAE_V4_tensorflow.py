#!/usr/bin/env python3
"""
Author: David Valencia
Date: 10 - Feb - 2021

Description: Version 4 of intro_vae for frame prediction.

             What is new here is:
             *** ---> Load the data with a custom data generation <--- ***
             and 100 % Manually following the same ideas as the paper.
"""
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda, Concatenate

from Load_Batch_Data_TensorFlow import triple_generator_multiple


# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#

def build_network_variational(latent_size=32):

    image_size = (192, 256, 3)
    action_space = 6

    # define inputs encoder
    layer_input_frame = Input(shape=image_size, name='layer_input_frame')

    # ================================================= Encoding =================================================== #
    conv_1 = Conv2D(filters=32,  kernel_size=5, strides=2, padding='same', activation='relu')(layer_input_frame)
    conv_2 = Conv2D(filters=64,  kernel_size=5, strides=2, padding='same', activation='relu')(conv_1)
    conv_3 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(conv_2)
    conv_4 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(conv_3)

    # ============================================== Middle Layers ================================================== #
    conv_shape = K.int_shape(conv_4)  # I will need this later (12, 16, 256)

    # flatten output matrix for further processing with dense layers
    layer_flatten = Flatten()(conv_4)
    layer_dense_1 = Dense(256)(layer_flatten)

    layer_mean    = Dense(latent_size, name='layer_mean')(layer_dense_1)  # variation layer
    layer_log_var = Dense(latent_size, name='layer_log_var')(layer_dense_1)  # log var layer
    layer_sigma   = Lambda(convert_to_sigma, name='sigma')(layer_log_var)  # sigma layer

    # sampling z
    layer_sampling = Lambda(sampling, name='z')([layer_mean, layer_sigma])

    # ================================================= Decoder ==================================================== #
    # define inputs decoder
    inp_decoder = Input(shape=(latent_size,),  name='layer_input_decoder')
    inp_action  = Input(shape=(action_space,), name='layer_input_action')

    layer_concatenate = Concatenate(axis=-1)([inp_decoder, inp_action])  # add actions to z vector (None, 38)

    layer_dense_2 = Dense(1024)(layer_concatenate)
    layer_dense_3 = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3])(layer_dense_2)
    layer_unflatten = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(layer_dense_3)

    d_conv1 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')(layer_unflatten)
    d_conv2 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(d_conv1)
    d_conv3 = Conv2DTranspose(filters=64,  kernel_size=4, strides=2, padding='same', activation='relu')(d_conv2)
    d_conv4 = Conv2DTranspose(filters=32,  kernel_size=4, strides=2, padding='same', activation='relu')(d_conv3)
    d_output = Conv2DTranspose(filters=3,  kernel_size=3, strides=1, padding='same', activation='sigmoid')(d_conv4)

    # ------
    # ------
    encoder_model = Model(layer_input_frame, [layer_mean, layer_log_var, layer_sampling], name="Encoder_Model")
    decoder_model = Model([inp_decoder, inp_action], d_output, name="Decoder_Model")

    encoder_model.summary()
    decoder_model.summary()

    return encoder_model, decoder_model


class Intro_VAE(tf.keras.Model):

    def __init__(self, enc, dec, z_dim):
        super(Intro_VAE, self).__init__()

        self.encoder = enc
        self.decoder = dec
        self.z_dim = z_dim

        self.optimizer_enc = tf.keras.optimizers.Adam(0.00002)
        self.optimizer_dec = tf.keras.optimizers.Adam(0.00002)

        self.m = 43.0  # after alpha zero
        self.beta = 1.0
        self.alpha = 0.25  # fix to 0.25

    def reconstruction_loss(self, target, prediction):
        l_recon = 0.5 * K.sum(K.square(target - prediction), axis=[1, 2, 3])
        return K.mean(l_recon)  # reconstruction_loss of the batch

    def kl_divergence_calculation(self, mean, log_var):
        l_reg = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
        return K.mean(l_reg)  # KL loss of the batch

    def calculate_encoder_loss(self, x, y, a, zp, ap):

        mean_z, log_var_z, z = self.encoder(x)  # z --> Enc(x)

        xr = self.decoder((z, a))
        xp = self.decoder((zp, ap))

        mean_r_ng, log_var_r_ng, _ = self.encoder(K.stop_gradient(xr))  # no 100% sure here
        mean_p_ng, log_var_p_ng, _ = self.encoder(K.stop_gradient(xp))  # no 100% sure here

        l_ae = self.reconstruction_loss(y, xr)  # L_AE Reconstruction Error

        l_reg_z = self.kl_divergence_calculation(mean_z, log_var_z)
        l_reg_zr_ng = self.kl_divergence_calculation(mean_r_ng, log_var_r_ng)
        l_reg_zpp_ng = self.kl_divergence_calculation(mean_p_ng, log_var_p_ng)

        enc_adv_l = K.maximum(0., self.m - l_reg_zr_ng) + K.maximum(0., self.m - l_reg_zpp_ng)
        encoder_loss = l_reg_z + self.alpha * enc_adv_l + self.beta * l_ae

        return encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng

    def calculate_decoder_loss(self, x, y, a, zp, ap):

        mean_z, log_var_z, z = self.encoder(x)  # z --> Enc(x)

        xr = self.decoder((z, a))
        xp = self.decoder((zp, ap))

        mean_r, log_var_r, _ = self.encoder(xr)
        mean_p, log_var_p, _ = self.encoder(xp)

        l_ae = self.reconstruction_loss(y, xr)  # L_AE Reconstruction Error

        l_reg_zr  = self.kl_divergence_calculation(mean_r, log_var_r)
        l_reg_zpp = self.kl_divergence_calculation(mean_p, log_var_p)

        dec_adv_l = l_reg_zr + l_reg_zpp
        decoder_loss = self.alpha * dec_adv_l + self.beta * l_ae

        return decoder_loss, l_ae, l_reg_zr, l_reg_zpp

    def train_manual(self, x, y, a):

        z_p = K.random_normal(shape=(x.shape[0], self.z_dim), mean=0.0, stddev=1.0)  # z --> Samples from prior N(0,I)
        a_p = tf.zeros([x.shape[0], 6], dtype=tf.float32)

        with tf.GradientTape() as enc_tape:
            encoder_loss, l_ae_en, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng = self.calculate_encoder_loss(x, y, a, z_p, a_p)

        enc_gradients = enc_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        self.optimizer_enc.apply_gradients(zip(enc_gradients, self.encoder.trainable_variables))

        with tf.GradientTape() as dec_tape:
            decoder_loss, l_ae_de, l_reg_zr, l_reg_zpp = self.calculate_decoder_loss(x, y, a, z_p, a_p)

        dec_gradients = dec_tape.gradient(decoder_loss, self.decoder.trainable_variables)
        self.optimizer_dec.apply_gradients(zip(dec_gradients, self.decoder.trainable_variables))

        return encoder_loss, decoder_loss, l_ae_en, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, l_reg_zr, l_reg_zpp


# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
# Helpers Functions

def sampling(args):

    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0., stddev=1.)
    return z_mean + z_sigma * epsilon


def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)


def plot_images_samples(target, predictions, directory, epoch):

    print("saving results samples")

    c = 1
    n_img = 16
    fig = plt.figure(figsize=(20, 5))

    for i in range(n_img):
        #  the target frame
        ax = fig.add_subplot(2, n_img, c)
        plt.imshow(target[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # the predicted frame
        ax = fig.add_subplot(2, n_img, c + n_img)
        plt.imshow(predictions[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        c = c + 1

    plt.savefig(f'./{directory}/sample_epoch_{epoch}.png')
    #plt.show()
    plt.close()


def plot_curves(loss_kl_z_values, loss_kl_zr_values, loss_kl_zp_values, loss_kl_zr_ng_values, loss_kl_zp_ng_values,
                loss_rec_values, loss_enc_values, loss_dec_values, fig_dir):

    print("Plotting Curves")

    epoch_plot = range(1, len(loss_kl_z_values) + 1)  # x axis

    plt.plot(epoch_plot, loss_kl_z_values, label='KLD_Z')
    plt.plot(epoch_plot, loss_kl_zp_values, label='KLD_Zp')
    plt.plot(epoch_plot, loss_kl_zr_values, label='KLD_Zr')

    plt.plot(epoch_plot, loss_kl_zr_ng_values, label='KLD_Zr_ng')
    plt.plot(epoch_plot, loss_kl_zp_ng_values, label='KLD_Zp_ng')

    plt.plot(epoch_plot, loss_rec_values, label='Reconst Loss')
    plt.plot(epoch_plot, loss_enc_values, label='Encoder Loss')
    plt.plot(epoch_plot, loss_dec_values, label='Decoder Loss')

    plt.title('Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{fig_dir}/Training_Curves.png')
    #plt.show()
    plt.close()


# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
# Train Functions

def intro_vae_frame_prediction_main(batch_size=32, n_epochs=10, save_per=100, latent_space_size=32):

    # ---------- Load Data with data generator -------- #
    img_input, img_target, vector_actions = triple_generator_multiple(batch_size=batch_size, mode='train')
    steps_epoch = len(img_input)

    # -------------- Helpers parameters --------------#
    loss_kl_z_values = []
    loss_kl_zr_values = []
    loss_kl_zp_values = []
    loss_kl_zr_ng_values = []
    loss_kl_zp_ng_values = []
    loss_rec_values = []
    loss_enc_values = []
    loss_dec_values = []

    fig_dir = './Images_Result_TF'
    os.makedirs(fig_dir, exist_ok=True)  # Create the folder to save images
    model_dir = './Model_Saved_TF'
    os.makedirs(model_dir, exist_ok=True)  # Create the folder to save model weights

    # --------------Build  and configure model --------------#

    encoder, decoder = build_network_variational(latent_size=latent_space_size)
    model = Intro_VAE(encoder, decoder, latent_space_size)

    #  ------------------- train_model ----------------------#
    for epoch in range(1, n_epochs + 1):
        loss_kl_z_batch  = []
        loss_kl_zr_batch = []
        loss_kl_zp_batch = []
        loss_kl_zr_np_batch = []
        loss_kl_zp_np_batch = []
        loss_rec_batch = []
        loss_enc_batch = []
        loss_dec_batch = []

        start_time = time.time()  # star time for each epoch

        for step, img_in, img_t, act in zip(range(1, steps_epoch + 1), img_input, img_target, vector_actions):

            encoder_loss, decoder_loss, l_ae_en, l_reg_z, \
            l_reg_zr_ng, l_reg_zpp_ng, l_reg_zr, l_reg_zpp = model.train_manual(img_in, img_t, act)

            # ========= show info per bach ==================
            info  = f" Epoch:[{epoch}/{n_epochs}] - Batch:[{step}/{steps_epoch}] -"
            info += f" Enc_Loss:{encoder_loss:4f} - Dec_Loss:{decoder_loss:4f} - Rec_Loss:{l_ae_en:4f} -"
            info += f" KL_z:{l_reg_z:4f} - KL_zr:{l_reg_zr:4f} - KL_zp:{l_reg_zpp:4f}-"
            info += f" KL_zr_ng:{l_reg_zr_ng:4f} - KL_zp_ng:{l_reg_zpp_ng:4f}"
            print(info)

            # Save Model checkpoint and plot some samples
            if epoch % save_per == 0 and step == steps_epoch-1:
                _, _, zt = model.encoder.predict(img_in)
                img_predicted = model.decoder.predict((zt, act))
                plot_images_samples(img_t, img_predicted, fig_dir, epoch)
                model.save_weights(f"./{model_dir}/checkpoint_{epoch:08d}.h5")

            loss_kl_z_batch.append(l_reg_z.numpy())
            loss_kl_zr_batch.append(l_reg_zr.numpy())
            loss_kl_zp_batch.append(l_reg_zpp.numpy())
            loss_kl_zr_np_batch.append(l_reg_zr_ng.numpy())
            loss_kl_zp_np_batch.append(l_reg_zpp_ng.numpy())
            loss_rec_batch.append(l_ae_en.numpy())
            loss_enc_batch.append(encoder_loss.numpy())
            loss_dec_batch.append(decoder_loss.numpy())
            #break

        # ========= show info per epoch ==================
        end_time = time.time() - start_time  # time per epoch
        start_time = 0

        kl_z_total  = np.mean(loss_kl_z_batch)
        kl_zr_total = np.mean(loss_kl_zr_batch)
        kl_zp_total = np.mean(loss_kl_zp_batch)
        kl_zr_ng_total = np.mean(loss_kl_zr_np_batch)
        kl_zp_ng_total = np.mean(loss_kl_zp_np_batch)
        rec_loss_total = np.mean(loss_rec_batch)
        enc_loss_total = np.mean(loss_enc_batch)
        dec_loss_total = np.mean(loss_dec_batch)

        info_epoch = f"\n==> Epoch:[{epoch}/{n_epochs}] - Time/epoch:{end_time:4.4f} -"
        info_epoch += f" Enc Loss:{enc_loss_total:4.4f} - Dec Loss:{dec_loss_total:4.4f} - Rec Loss:{rec_loss_total:4.4f} -"
        info_epoch += f" KL_z:{kl_z_total:4.4f} - KL_zr:{kl_zr_total:4.4f} - KL_zp:{kl_zp_total:4.4f} -"
        info_epoch += f" KL_zr_ng:{kl_zr_ng_total:4.4f} - KL_zp_ng:{kl_zp_ng_total:4.4f} <==\n"
        print(info_epoch)

        loss_kl_z_values.append(kl_z_total)
        loss_kl_zr_values.append(kl_zr_total)
        loss_kl_zp_values.append(kl_zp_total)
        loss_kl_zr_ng_values.append(kl_zr_ng_total)
        loss_kl_zp_ng_values.append(kl_zp_ng_total)
        loss_rec_values.append(rec_loss_total)
        loss_enc_values.append(enc_loss_total)
        loss_dec_values.append(dec_loss_total)

    #save model after training
    model.save_weights(f'./{model_dir}/model_intro_vae_completed_after_{n_epochs}_epochs.h5')

    # Plot Curves
    plot_curves(loss_kl_z_values, loss_kl_zr_values, loss_kl_zp_values, loss_kl_zr_ng_values, loss_kl_zp_ng_values,
                loss_rec_values, loss_enc_values, loss_dec_values, fig_dir)
# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#


if __name__ == '__main__':

    # ========== Global Parameters ==================== #
    NUM_EPOCHS  = 5000
    BATCH_SIZE  = 32
    LATENT_SIZE = 32
    SAVE_PERIOD = 500  # When to save a checkpoint

    intro_vae_frame_prediction_main(batch_size=BATCH_SIZE,
                                    n_epochs=NUM_EPOCHS,
                                    save_per=SAVE_PERIOD,
                                    latent_space_size=LATENT_SIZE)
