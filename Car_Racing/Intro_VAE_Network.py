#!/usr/bin/env python3

"""
Author      : David Valencia
Date        : March / 2021
Description :

"""
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from Load_Batch_Data import load_data_generator

# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#


class Flatten(nn.Module):
    def forward(self, input_v):
        return input_v.view(input_v.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input_v):
        return input_v.view(input_v.size(0), 256, 4, 4)


class Decoder_Net(nn.Module):

    def __init__(self, z_dim=32):
        super(Decoder_Net, self).__init__()

        self.z_dim = z_dim
        self.action_dim = 3

        self.fc4 = nn.Linear(in_features=(self.z_dim + self.action_dim), out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=4096)

        self.decoder_network = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=1, padding=1),
            nn.Sigmoid()
        )

    def decode_img(self, z, a):
        concatenate = torch.cat((z, a), -1)  # add actions to z vector (None, 38)

        h_dense_1 = self.fc4(concatenate)
        h_dense_2 = self.fc5(h_dense_1)
        xr = self.decoder_network(h_dense_2)
        return xr

    def forward(self, z, a):
        x_r = self.decode_img(z, a)
        return x_r


class Encoder_Net(nn.Module):

    def __init__(self, z_dim=32):
        super(Encoder_Net, self).__init__()

        self.z_dim = z_dim

        self.encoder_network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(in_features=4096, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.z_dim)
        self.fc3 = nn.Linear(in_features=256, out_features=self.z_dim)

    def encode_img(self, x):
        h = self.encoder_network(x)
        h_dense = self.fc1(h)
        mu, log_var = self.fc2(h_dense), self.fc3(h_dense)
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.encode_img(x)
        return mu, log_var


class IntroVAE(nn.Module):
    def __init__(self, z_dim):
        super(IntroVAE, self).__init__()

        self.z_dim = z_dim
        self.encoder = Encoder_Net(z_dim)
        self.decoder = Decoder_Net(z_dim)

    def reparametrization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)  # todo careful here maybe I need .to(device) or put in outside in helper
        z = mu + std * eps
        return z

    def forward(self, x, a):
        mu, log_var = self.encoder(x)
        z  = self.reparametrization(mu, log_var)
        xr = self.decoder(z, a)
        return xr, z, mu, log_var


# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
# Helpers Functions

def calc_reconstruction_loss(target, recon_x):
    recon_error = F.mse_loss(recon_x, target, reduction='none')
    recon_error = recon_error.sum((1, 2, 3))
    recon_error = 0.5 * torch.mean(recon_error)
    return recon_error


def calc_kl_loss(logvar, mu):
    kld = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=-1)
    kld = torch.mean(kld)
    return kld


def plot_curves(loss_kl_z_values, loss_kl_zr_values, loss_kl_zp_values,
                loss_rec_values, loss_enc_values, loss_dec_values, fig_dir):

    print("Plotting Curves")

    epoch_plot = range(1, len(loss_kl_z_values) + 1)
    plt.plot(epoch_plot, loss_kl_z_values, label='KLD_Z')
    plt.plot(epoch_plot, loss_kl_zp_values, label='KLD_Zp')
    plt.plot(epoch_plot, loss_kl_zr_values, label='KLD_Zr')

    plt.plot(epoch_plot, loss_rec_values, label='Reconst Loss')
    plt.plot(epoch_plot, loss_enc_values, label='Encoder Loss')
    plt.plot(epoch_plot, loss_dec_values, label='Decoder Loss')

    plt.title('Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{fig_dir}/Training_Curves.png')
    plt.show()


# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
# Train Functions

def intro_vae_frame_prediction(device=torch.device("cuda:0"), batch_size=32,
                               latent_size=32, num_epochs=100, save_period=100):

    #  ------------------- load data ------------------------#
    img_input, img_target, vector_actions = load_data_generator(batch_size=batch_size, mode='train')

    # --------------Hyper and helpers parameters --------------#
    beta = 1.0
    alpha = 0.25
    marginal = 1.90

    loss_kl_z_values  = []
    loss_kl_zr_values = []
    loss_kl_zp_values = []
    loss_rec_values = []
    loss_enc_values = []
    loss_dec_values = []

    fig_dir = './Images_Result'
    os.makedirs(fig_dir, exist_ok=True)  # Create the folder to save images

    model_dir = './Model_Saved'
    os.makedirs(model_dir, exist_ok=True)  # Create the folder to save model weights

    # --------------build  and configure model --------------#
    model = IntroVAE(z_dim=latent_size).to(device)
    optimizer_e = torch.optim.Adam(model.encoder.parameters(), lr=0.00005)
    optimizer_d = torch.optim.Adam(model.decoder.parameters(), lr=0.00005)

    #  ------------------- train_model ----------------------#
    for epoch in range(1, num_epochs + 1):

        loss_kl_z_batch  = []
        loss_kl_zr_batch = []
        loss_kl_zp_batch = []
        loss_rec_batch = []
        loss_enc_batch = []
        loss_dec_batch = []
        start_time = time.time()  # star time for each epoch

        for idx, ((img_in, _), (img_t, _), act) in enumerate(zip(img_input, img_target, vector_actions), 1):

            img_in, img_t, act = img_in.to(device), img_t.to(device), act.to(device)
            zp = torch.randn(size=(img_in.size(0), latent_size)).to(device)  # noise batch
            ap = torch.zeros(size=(img_in.size(0), 3)).to(device)  # action noise #todo carefully here try other option

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False

            mu, log_var = model.encoder(img_in)  # batch of real images
            z = model.reparametrization(mu, log_var)

            xr = model.decoder(z, act)
            xp = model.decoder(zp, ap)

            mu_r_ng, log_var_r_ng = model.encoder(xr.detach())
            mu_p_ng, log_var_p_ng = model.encoder(xp.detach())

            l_ae = calc_reconstruction_loss(img_t, xr)  # reconstruction loss

            l_reg_z = calc_kl_loss(mu, log_var)
            l_reg_zr_ng = calc_kl_loss(mu_r_ng, log_var_r_ng)
            l_reg_zp_ng = calc_kl_loss(mu_p_ng, log_var_p_ng)

            enc_adv_l = torch.clamp(marginal - l_reg_zr_ng, min=0) + torch.clamp(marginal - l_reg_zp_ng, min=0)
            encoder_loss = l_reg_z + alpha * enc_adv_l + beta * l_ae

            optimizer_e.zero_grad()
            encoder_loss.backward()
            optimizer_e.step()

            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            mu, log_var = model.encoder(img_in)
            z  = model.reparametrization(mu, log_var)

            xr = model.decoder(z, act)
            xp = model.decoder(zp, ap)

            mu_r, log_var_r = model.encoder(xr)
            mu_p, log_var_p = model.encoder(xp)

            l_ae = calc_reconstruction_loss(img_t, xr)  # reconstruction loss

            l_reg_zr = calc_kl_loss(mu_r, log_var_r)
            l_reg_zp = calc_kl_loss(mu_p, log_var_p)

            dec_adv_l = l_reg_zr + l_reg_zp
            decoder_loss = alpha * dec_adv_l + beta * l_ae

            optimizer_d.zero_grad()
            decoder_loss.backward()
            optimizer_d.step()

            # ========= show info per bach ==================
            if torch.isnan(encoder_loss) or torch.isnan(decoder_loss):
                raise SystemError("NaN values")

            info  = f" Epoch:[{epoch}/{num_epochs}], Batch:[{idx}/{len(img_input)}],"
            info += f" Enc_Loss:{encoder_loss:4f}, Dec_Loss:{decoder_loss:4f}, Rec_Loss:{l_ae:4f},"
            info += f" KL_z:{l_reg_z:4f}, KL_zr:{l_reg_zr:4f}, KL_zp:{l_reg_zp:4f}"
            print(info)

            # ========= save some images =============
            if epoch % save_period == 0 and idx == 7:
                print("Saving Image Sample")
                save_image(
                    torch.cat([img_t[0:16], xr[0:16], xp[0:16]], dim=0).data.cpu(),
                    f'./{fig_dir}/sample_epoch_{epoch}.png', nrow=16)

            loss_kl_z_batch.append(l_reg_z.data.cpu().item())
            loss_kl_zr_batch.append(l_reg_zr.data.cpu().item())
            loss_kl_zp_batch.append(l_reg_zp.data.cpu().item())
            loss_rec_batch.append(l_ae.data.cpu().item())
            loss_enc_batch.append(encoder_loss.data.cpu().item())
            loss_dec_batch.append(decoder_loss.data.cpu().item())
            #break

        # ========= show info per epoch ==================
        end_time = time.time() - start_time  # time per epoch
        start_time = 0

        kl_z_total = np.mean(loss_kl_z_batch)
        kl_zr_total = np.mean(loss_kl_zr_batch)
        kl_zp_total = np.mean(loss_kl_zp_batch)
        rec_loss_total = np.mean(loss_rec_batch)
        enc_loss_total = np.mean(loss_enc_batch)
        dec_loss_total = np.mean(loss_dec_batch)

        info_epoch = f"\n====> Epoch:[{epoch}/{num_epochs}] - Time/epoch:{end_time:4.4f} -"
        info_epoch += f" Enc Loss:{enc_loss_total:4f} - Dec Loss:{dec_loss_total:4f} - Rec Loss:{rec_loss_total:4f} -"
        info_epoch += f" KL_z:{kl_z_total:4f} - KL_zr:{kl_zr_total:4f} - KL_zp:{kl_zp_total:4f} <====\n"
        print(info_epoch)

        loss_kl_z_values.append(kl_z_total)
        loss_kl_zr_values.append(kl_zr_total)
        loss_kl_zp_values.append(kl_zp_total)
        loss_rec_values.append(rec_loss_total)
        loss_enc_values.append(enc_loss_total)
        loss_dec_values.append(dec_loss_total)

    # Save Model
    torch.save(model.state_dict(), f'./{model_dir}/model_intro_vae_completed_after_{num_epochs}_epochs')

    # Plot Curves
    plot_curves(loss_kl_z_values, loss_kl_zr_values, loss_kl_zp_values,
                loss_rec_values, loss_enc_values, loss_dec_values, fig_dir)

# -------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------#


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # ========== Global Parameters ==================== #
    NUM_EPOCHS  = 5000
    BATCH_SIZE  = 32
    LATENT_SIZE = 32
    SAVE_PERIOD = 100  # When to save a checkpoint

    intro_vae_frame_prediction(device=device,
                               batch_size=BATCH_SIZE,
                               latent_size=LATENT_SIZE,
                               num_epochs=NUM_EPOCHS,
                               save_period=SAVE_PERIOD)
