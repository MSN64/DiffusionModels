import copy
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from my_modules import UNet, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

logging.basicConfig(format = "%(asctime)s - %(levelname)s: %(message)s", level = logging.INFO, datefmt = "%I:%M:%S")

IMAGE_SIZE = 32

class Diffusion:
    def __init__(self, noise_steps = 4000, beta_start = 1e-4/4, beta_end = 0.02/4, s = 0.008, img_size = IMAGE_SIZE, device = "cuda"): #Beta bounds/4 for T=4000
        self.noise_steps = noise_steps
        self.beta_start = beta_start         #needed for linear schedule
        self.beta_end = beta_end             #needed for linear schedule
        self.img_size = img_size
        self.device = device
        #self.s = s                            #offset for cosine schedule

        self.beta = self.prepare_noise_schedule().to(device)                       #beta schedule
        self.alpha = 1. - self.beta                                                #defining alphas
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)


#ALGORITHM 1:

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)    #using linear beta schedule for now
    """

    #cosine schedule: s is an offset of 0.008
    def prepare_noise_schedule(self):
        steps = self.noise_steps + 1
        x = torch.linspace(0, steps, steps) / steps                                           #x = t/T
        alphas_cumprod = torch.cos((x + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2              #f(t), s: offset
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]                                        #alpha_bar_(t) = f(t)/f(0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])                #alpha_bar_(t-1)
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)

        #Add a small value to betas to prevent them from being zero
        #small_value = 1e-4
        #betas = betas.clamp(min=small_value)

        return betas[1:]
        """

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]         #select sqrt of alpha value at timestep t
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)                                               #x-sized tensor filled with random num from a gaussian dist.
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
        # x_{t} = \sqrt{\Bar{\alpha}_{t}}x_{0} + \sqrt{1 - \Bar{\alpha}_{t}}\epsilon


    def sample_timesteps(self, n):                                                 #sample random time-step t from a uniform dist. (tensor of size n (no. of images))
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))        #Returns tensor filled with random int generated uniformly between low (inclusive) and high


    def sample(self, model, n):                                                    #takes in the model used for sampling, n: no. of images we want to sample
        logging.info(f"Sampling {n} new images....")
        model.eval()                                                               #sets the PyTorch model to evaluation mode, disabling operations like dropout. Useful for inference and testing
        with torch.no_grad():                                                      #disables gradient calculation, useful for inference, when you are sure that you will not call Tensor.backward()
            #FOLLOWING ALGORITHM 2:
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)  #sampling n initial images (3,64,64) (3 channels with 64x64) from standard normal dist. (Gaussian noise) x_{T}. Changed 3 to 1

            for i in tqdm(reversed(range(1, self.noise_steps)), position = 0):     #loop going over T timesteps (1000), from highest until 1, tqdm: Instantly make your loops show a smart progress meter
                t = (torch.ones(n) * i).long().to(self.device)                     #creating timesteps by creating tensor of length n (of no. of images), of val 1, then prod. with current timestep
                predicted_noise = model(x, t)                                      #\epsilon_{\theta}: feed this tensor t of length n (np. of images) into model, together with the current images, so each image complemented with corresponding timestep into the model
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]                 #'None': dummy dimensions added in order to have same dimensionality as the image we are doing element-wise multiplication with
                beta = self.beta[t][:, None, None, None]

                if torch.isnan(predicted_noise).any():
                    print(f"NaN in predicted_noise at step {i}")

                if i > 1:                                                          #only need Gaussian noise (z in denoising paper) for the timesteps greater than one in RP
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                #Update x
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise  #removing a bit of noise from sampled images in each iteration as we go in reverse
                # x_{t-1} = \frac{1}{\sqrt{\alpha_{t}}}(x_{t} - \frac{(1-\alpha_{t})}{\sqrt{1-alpha_bar_{t}}}\epsilon_{\theta}) + \sigma_{t}z

                if torch.isnan(x).any():
                    print(f"NaN in x after update at step {i}")
                    print(f"alpha: {alpha}")
                    print(f"alpha_hat: {alpha_hat}")
                    print(f"beta: {beta}")
                    print(f"predicted_noise: {predicted_noise}")
                    print(f"noise: {noise}")
                    break


        model.train()
        #x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        #x = x.clamp(-1, 1)
        #x = (x + 1) / 2  # Normalize to [0, 1]
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)   #load dataset
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size = args.image_size, device = device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))                     #log training stats
    l = len(dataloader)
    print("len(dataloader): ", l)

    #Initialize EMA
    ema = EMA(beta=0.995)  # Adjust decay rate, beta, as needed
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    epoch_loss_values = []

    # *** Added code to record start time ***
    start_time = time.time()

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)  #load dataset

        #Empty list to store loss values for each batch
        batch_loss_values = []

        #ALGORITHM 1:
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            #print("Images shape: ", images.shape)                              #(10, 180, 180) -> (batch_size, img_size, img_size)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)          #sample random timesteps from range [1, 1000] (tensor of size n)
            #print("t shape: ", t.shape)                                        #t -> size n equal to batch_size
            x_t, noise = diffusion.noise_images(images, t)                      #noise the n no. of images up to the specified timestep t, also returns noise epsilon as it's needed for the loss function

            predicted_noise = model(x_t, t)                                     #feed noised images to UNet model, which predicts the noise in the image (epsilon_theta)
            loss = mse(noise, predicted_noise)                                  #mse between actual noise and predicted noise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #UPDATE EMA:
            ema.step_ema(ema_model, model)

            #Append loss for each batch
            batch_loss_values.append(loss.item())

            #logging
            pbar.set_postfix(MSE = loss.item())
            logger.add_scalar("MSE", loss.item(), global_step = epoch * l + i)

        epoch_loss = sum(batch_loss_values) / len(batch_loss_values)
        epoch_loss_values.append(epoch_loss)

        logging.info(f"Mean loss for epoch {epoch}: {epoch_loss}")

        if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            sampled_images = diffusion.sample(model, n = 16)
            ema_sampled_images = diffusion.sample(ema_model, n = 16)
            print("sampled_images type: ", type(sampled_images), ", shape: ", sampled_images.shape)

            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_{IMAGE_SIZE}_ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_{IMAGE_SIZE}_ckpt_ema.pt"))


    # *** record end time and calculate total training time ***
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {int(total_time // 3600)} hour(s), {int((total_time % 3600) // 60)} minute(s), {int(total_time % 60)} second(s)")

    #Plot loss function against number of epochs
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(range(args.epochs), epoch_loss_values, label="MSE Loss", color = 'orange')
    ax.set_xlabel('Epoch', fontsize=22)
    ax.set_ylabel('MSE Loss', fontsize=22)
    ax.set_title('MSE Loss vs. Epochs', fontsize=24)
    ax.legend()

    plt.savefig(os.path.join("results", args.run_name, "loss_plot.png"))
    np.save(os.path.join("results", args.run_name, "epoch_loss_values.npy"), np.array(epoch_loss_values))

    #return epoch_loss_values


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 2500       #500
    args.batch_size = 80    #12
    print("Batch size: ", args.batch_size)
    args.image_size = IMAGE_SIZE
    #args.dataset_path = r"D:\SnapshotQuijote\QuijoteImages\test"  #r"D:\SnapshotQuijote\QuijoteImages"
    args.dataset_path = r"/data/cluster/mnoor/Quijote1Channel32"                    #6000 samples
    #args.dataset_path = r"/data/cluster/mnoor/Quijote1Channel32_8000"
    #args.dataset_path = r"/data/cluster/mnoor/QuijoteImages"

    args.device = "cuda"
    args.lr = 1.5e-4
    train(args)


if __name__ == '__main__':
    launch()