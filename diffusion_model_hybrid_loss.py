import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from my_modules import UNet, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math

#CHECK IF DEVICE IS SET TO CUDA
logging.basicConfig(format = "%(asctime)s - %(levelname)s: %(message)s", level = logging.INFO, datefmt = "%I:%M:%S")

IMAGE_SIZE = 32
LAMBDA_VLB = 0.001                                                                                                      #weighting of the variational lower bound loss in the hybrid loss objective

class Diffusion:   #noise_steps = 4000, beta_start = 1e-4/4, beta_end = 0.02/4
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, s = 0.008, img_size = IMAGE_SIZE, device = "cuda"): #Beta bounds/4 for T=4000
        self.noise_steps = noise_steps
        self.beta_start = beta_start         #needed for linear schedule
        self.beta_end = beta_end             #needed for linear schedule
        self.img_size = img_size
        self.device = device
        #self.s = s                          #offset for cosine schedule

        self.beta = self.prepare_noise_schedule().to(device)                                                            #beta schedule, tensor(1000,)
        self.alpha = 1. - self.beta                                                                                     #defining alphas, tensor(1000,)
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0).to(device)                                                  #alpha bar, tensor(1000,)

        # Define required quantities
        self.sqrt_alpha_t = torch.sqrt(self.alpha)                                                                      #sqrt(alpha_t), tensor(1000,)
        self.alpha_hat_prev = torch.cat([torch.tensor([1.], device=device), self.alpha_hat[:-1]])                  # alpha_hat(t-1), tensor(1000,)
        self.sqrt_alpha_hat_prev = torch.sqrt(self.alpha_hat_prev)                                                      # sqrt(alpha_hat(t-1)), tensor(1000,)
        self.sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat)                                                              # sqrt(alpha_hat(t)), tensor(1000,)
        self.sqrt_one_minus_alpha_hat_t = torch.sqrt(1. - self.alpha_hat)                                               # sqrt(1 - alpha_hat(t)), tensor(1000,)
        self.sqrt_one_over_alpha_hat_t = torch.sqrt(1. / self.alpha_hat)                                                # sqrt(1 / alpha_hat(t)), tensor(1000,)
        self.sqrt_ratio_alpha_hat_t = torch.sqrt((1. - self.alpha_hat) / self.alpha_hat)                                # sqrt((1 - alpha_hat(t)) / alpha_hat(t)), tensor(1000,), not tensor(1000, 1, 1, 1) yet

        self.tilde_beta = self.beta * (1 - self.alpha_hat_prev) / (1 - self.alpha_hat)
        self.log_beta = torch.log(self.beta).to(device)                                                                 #log(\beta_{t}), needed for learnt variance eqn (15)
        self.log_tilde_beta = torch.log(self.tilde_beta)                                                                #tensor(1000,), log(\Tilde{\beta}_{t}),  needed for learnt variance eqn (15)

        #Clip log_tilde_beta to remove variance at t=0, as it will be strange (edogariu) (-inf at t = 0 without clipping)
        log_tilde_beta_clipped = torch.cat([self.log_tilde_beta[1:2], self.log_tilde_beta[1:]])
        self.log_tilde_beta_clipped = log_tilde_beta_clipped.to(device)

        #Additional coefficients for posterior mean calculation
        self.posterior_mean_coef_x0 = (torch.sqrt(self.alpha_hat_prev) * self.beta / (1. - self.alpha_hat))
        self.posterior_mean_coef_xt = (torch.sqrt(self.alpha) * (1 - self.alpha_hat_prev)) / (1. - self.alpha_hat)

#ALGORITHM 1:

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)                                         #tensor(1000,), using linear beta schedule for now


    def noise_images(self, x, t):                                                                                       #t: tensor(no.of images,), x: tensor(no. of images, 1, 32, 32)
        sqrt_alpha_hat = self.sqrt_alpha_hat_t[t][:, None, None, None]                                                  #tensor(no. of images, 1, 1, 1), select sqrt of alpha value at timestep t
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat_t[t][:, None, None, None]                              #tensor(no. of images, 1, 1, 1)
        epsilon = torch.randn_like(x)                                                                                   #tensor(no. of images, 1, 32, 32), x-sized tensor filled with random num from a gaussian dist.
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon                                         #return x_t (noised image), and gaussian noise
        # x_{t} = \sqrt{\Bar{\alpha}_{t}}x_{0} + \sqrt{1 - \Bar{\alpha}_{t}}\epsilon                                    #since x and epsilon are sizes of image, and alphas are (n, 1, 1, 1), each pixel value in x or eps is scaled by alpha type variables

    def sample_timesteps(self, n):                                                                                      #n (int), sample random time-step t from a uniform dist.
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))                                             #tensor(no. of images,) with different t values from range, returns tensor filled with random int time-steps generated uniformly between low (inclusive) and high

    def posterior_mean(self, x_t, x_0, t):                                                                              #eqn (7) in DDPM, t: tensor(no.of images,)
        beta_t = self.beta[t][:, None, None, None]                                                                      #tensor(no. of images, 1, 32, 32)
        sqrt_alpha_hat_prev = self.sqrt_alpha_hat_prev[t][:, None, None, None]                                          #tensor(no. of images, 1, 32, 32)
        sqrt_alpha_t = self.sqrt_alpha_t[t][:, None, None, None]                                                        #...
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat_t[t][:, None, None, None]

        x0_coef = self.posterior_mean_coef_x0[t][:, None, None, None]
        xt_coef = self.posterior_mean_coef_xt[t][:, None, None, None]                                                   #...
        post_mean = (x0_coef * x_0) + (xt_coef * x_t)                                                                   #tensor(no. of images, 1, 32, 32)
        return post_mean


    def sample(self, model, n):                                                                                         #takes in the model used for sampling, n: no. of images we want to sample
        logging.info(f"Sampling {n} new images....")
        model.eval()                                                                                                    #sets the PyTorch model to evaluation mode, disabling operations like dropout. Useful for inference and testing
        with torch.no_grad():                                                                                            #disables gradient calculation, useful for inference, when you are sure that you will not call Tensor.backward()
            #FOLLOWING ALGORITHM 2:
            #changed ..(n, 3, ...) to (n, 1, ...) below.
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)                                       #sampling n initial x_{T} noise images (3 or 1, 64, 64) (3 or 1 channel(s) with 32x32) from standard normal dist. (Gaussian noise) x_{T}. Changed 3 to 1

            for i in tqdm(reversed(range(1, self.noise_steps)), position = 0):                                          #loop going over T timesteps (1000), from highest T until 1, tqdm: Instantly make your loops show a smart progress meter
                t = (torch.ones(n) * i).long().to(self.device)                                                          #creating timesteps by creating tensor of length n (of no. of images), of val 1, then prod. with current timestep. So, n values of t are generated, for feeding it to the model
                predicted_noise, log_variance = model(x, t)                                                             #tensor(no. of images, 1, 32, 32), \epsilon_{\theta}: feed this tensor t of length n (no. of images) into model, together with the current images, so each image complemented with corresponding timestep into the model
                #Don't need to do this I think:
                log_variance = torch.mean(log_variance, dim=[2, 3], keepdim=True)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:                                                                                               #only need Gaussian noise (z in denoising paper) for the timesteps greater than one in RP
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # Update x
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise      #removing a bit of noise from sampled images in each iteration as we go in reverse
                # x_{t-1} = \frac{1}{\sqrt{\alpha_{t}}}(x_{t} - \frac{(1-\alpha_{t})}{\sqrt{1-alpha_bar_{t}}}\epsilon_{\theta}) + \sigma_{t}z

        model.train()
        #x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        #x = x.clamp(-1, 1)
        #x = (x + 1) / 2  # Normalize to [0, 1]
        return x


    #Hybrid loss objective:
    def hybrid_loss(self, model, x_t, t, noise, beta, alpha_hat, alpha, log_beta, log_tilde_beta_clipped, x_0):         #input the UNet model which predicts the mean and log variance of the noise. t: time-step at which noise is added                                                                                       #x_start: original clean image. noise added to get x_t from x_start
        pred_noise, vect_v = model(x_t, t)                                                                              #log_beta and log_tilde_beta: upper and lower bound, respectively. mean_pred is the noise prediction
        vect_v = torch.mean(vect_v, dim=[2, 3], keepdim=True)                                                           #pred_noise size tensor(no. of images, 1, 32, 32), vect_v tensor(n, 1, 1, 1)

        beta = beta[t][:, None, None, None]                                                                             #tensor(no. of images, 1, 1, 1)
        log_beta = log_beta[t][:, None, None, None]                                                                     #tensor(no. of images, 1, 1, 1)
        log_tilde_beta_clipped = log_tilde_beta_clipped[t][:, None, None, None]                                         #tensor(no. of images, 1, 1, 1)
        alpha_hat = alpha_hat[t][:, None, None, None]                                                                   #tensor(no. of images, 1, 1, 1)
        alpha = alpha[t][:, None, None, None]                                                                           #tensor(no. of images, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat_t[t][:, None, None, None]                              #tensor(no. of images, 1, 1, 1)

        #Predicted variance parameterisation:
        v = torch.sigmoid(vect_v)                                                                                       #squashes log_var_pred values into range (0,1) using sigmoid function, output same size: tensor(n, 1, 1, 1)
        pred_log_var = v * log_beta + (1 - v) * log_tilde_beta_clipped                                                  #interpolates between log_beta[t] and log_tilde_beta[t] using value v. tensor(no. of images, 1, 1, 1)
        var = torch.exp(pred_log_var)                                                                                   #eqn (15). Is this needed?, tensor(n, 1, 1, 1)

        #Simple Loss (MSE of noise):
        simple_loss = F.mse_loss(pred_noise, noise)                                                                     #eqn (14), zero-dimensional tensor () -> scalar tensor

        #q_posterior_mean:
        posterior_mean = Diffusion.posterior_mean(self, x_t, x_0, t)                                                    #posterior mean tilde \mu, tensor(n, 1, 32, 32) (should be size of image)
        #true posterior_variance of q:
        posterior_log_var = log_tilde_beta_clipped                                                                      #tensor(no. of images, 1, 1, 1), (not size of image, supposed to act like scaling of image type variables)

        #Stop-gradient for mean prediction in VLB term:                                                                 #stop-gradient: detaches mean_pred from the computational graph, so it's treated as a constant in the following calculations. Done to prevent gradient of mean prediction from affecting VLB loss, ensures that loss_vlb doesn't backdrop through pred_noise, only var
        pred_noise = pred_noise.detach()                                                                                #tensor(no. of images, 1, 32, 32)

        #predicted mean, \mu_{\theta}:
        pred_mean = (x_t - (beta / sqrt_one_minus_alpha_hat) * pred_noise) / torch.sqrt(alpha)                          #tensor(no. of images, 1, 1, 1), like posterior_mean

        #Initialize kl_loss as a scalar tensor
        kl_loss = torch.tensor(0.0, device=self.device)

        for i in range(t.size(0)):
            if t[i] == 0:
                #Calculate negative log-likelihood of start image appearing in predicted distribution
                kl_loss += self.log_likelihood(x_0, pred_mean, pred_log_var)
            else:
                kl_loss += self.kl_div(posterior_mean, posterior_log_var, pred_mean, pred_log_var)

        #Combining the losses:
        loss = simple_loss + LAMBDA_VLB * kl_loss                                                                       #Hybrid loss objective. lambda = 0.001 controls importance of L_{VLB}. Should be zero-dim. tensor
        return loss, simple_loss, kl_loss


    def kl_div(self, mean_1, log_var_1, mean_2, log_var_2):                                                             #KL Divergence between actual and pred. dist. (true_mean, true_log_var, pred_mean, pred_log_var)
        """
        Returns KL-Divergence between two Gaussian's with given parameters.
        Expects tensor inputs and returns tensor output for use in loss functions.
        Output is measured in nats (log base e).
        """
                                                                                                                        #Formula is KL(p,q) = log(sigma2/sigma1) + (sigma1^2+(mu1-mu2)^2)/(2sigma2^2) - 1/2
                                                                                                                        #                   = log(var2/var1)/2 + var1/2var2 + (mu1-mu2)^2/2sigma2^2 - 1/2
        kl = ((log_var_2 - log_var_1) + torch.exp(log_var_1 - log_var_2) +
                ((mean_1 - mean_2) ** 2) * torch.exp(-log_var_2) - 1.0) / 2
        kl_sum = torch.sum(kl, dim=[1, 2, 3])                                                                           #kl of size (n, 1, 32, 32). This return statement sums all the elements along the specified dims.
        return kl_sum

    def log_likelihood(self, target, mean, log_var):
        """
        Returns log-likelihood of Gaussian with given mean and log variance, without discretization.
        Expects tensor inputs and returns tensor output for use in loss functions.
        target should be an image with values in [-1.0, 1.0]
        Output is measured in nats (log base e).
        """
        assert target.shape == mean.shape == log_var.shape

        #Calculate standard deviation from log variance
        std = torch.exp(0.5 * log_var)

        #Calculate squared error term
        squared_error = 0.5 * ((target - mean) / std)**2

        #Calculate log likelihood
        log_likelihood = -torch.log(std) - squared_error - 0.5 * torch.log(torch.tensor(2 * math.pi))

        return log_likelihood

    def approx_cdf(self, x):
        """
        Returns approximate value of cdf(x) for the Gaussian with zero mean and unit variance. x can be a tensor.
        Approximation is from:
        "Page, E. (1977). Approximations to the cumulative normal function and its inverse for use on a pocket calculator."
        """
        y = math.sqrt(2.0 / math.pi) * (x + 0.0444715 * (x ** 3))
        return 0.5 * (1.0 + torch.tanh(y))



#train:
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)                                                     #load dataset
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    #mse = nn.MSELoss()
    diffusion = Diffusion(img_size = args.image_size, device = device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))                     #log training stats
    l = len(dataloader)
    print("len(dataloader): ", l)

    #Initialize EMA
    ema = EMA(beta=0.995)                                                           #Adjust decay rate, beta, as needed. Take beta = 0.999 next. This is the value taken in the IDDPM paper.
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    #Empty list to store loss values for each epoch
    epoch_loss_values = []
    epoch_simple_loss_values = []
    epoch_kl_loss_values = []

    # *** Added code to record start time ***
    start_time = time.time()

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)  #load dataset

        #Empty list to store loss values for each batch
        batch_loss_values = []
        batch_simple_loss_values = []
        batch_kl_loss_values = []

        #ALGORITHM 1:
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)                                           #torch.tensor
            #print("Images shape: ", images.shape)                               #(80, 32, 32) -> (batch_size, img_size, img_size)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)           #sample random timesteps from range [1, 1000], tensor(n,)
            #print("t shape: ", t.shape)                                         #t -> size n equal to batch_size
            x_t, noise = diffusion.noise_images(images, t)                       #noise the n no. of images up to the specified timestep t, also returns noise epsilon as it's needed for the loss function

            loss, simple_loss, kl_loss = diffusion.hybrid_loss(model, x_t, t, noise, diffusion.beta, diffusion.alpha_hat, diffusion.alpha, diffusion.log_beta, diffusion.log_tilde_beta_clipped, images)

            #predicted_noise = model(x_t, t)                                     #feed noised images to UNet model, which predicts the noise in the image (epsilon_theta)
            #loss = mse(noise, predicted_noise)                                  #mse between actual noise and predicted noise

            optimizer.zero_grad()                                                #zero out the gradients for the optimizer. This is necessary because, by default, gradients in PyTorch accumulate
            loss.backward()                                                      #computes the gradients of the loss with respect to the model parameters
            optimizer.step()                                                     #updates the model parameters based on the computed gradients

            #UPDATE EMA:
            ema.step_ema(ema_model, model)

            #Append loss for each batch
            batch_loss_values.append(loss.item())
            batch_simple_loss_values.append(simple_loss.item())
            batch_kl_loss_values.append(kl_loss.item())

            #logging
            pbar.set_postfix(Hybrid_Loss = loss.item())
            logger.add_scalar("Hybrid Loss", loss.item(), global_step = epoch * l + i)
            logger.add_scalar("Simple Loss", simple_loss.item(), global_step = epoch * l + i)
            logger.add_scalar("KL Loss", kl_loss.item(), global_step = epoch * l + i)

        #Compute mean loss for the epoch and append to epoch_loss_values
        epoch_loss = sum(batch_loss_values) / len(batch_loss_values)
        epoch_simple_loss = sum(batch_simple_loss_values) / len(batch_simple_loss_values)
        epoch_kl_loss = sum(batch_kl_loss_values) / len(batch_kl_loss_values)

        epoch_loss_values.append(epoch_loss)
        epoch_simple_loss_values.append(epoch_simple_loss)
        epoch_kl_loss_values.append(epoch_kl_loss)

        #Log mean loss for the epoch
        logging.info(f"Mean loss for epoch {epoch}: {epoch_loss}")
        logging.info(f"Mean simple loss for epoch {epoch}: {epoch_simple_loss}")
        logging.info(f"Mean KL loss for epoch {epoch}: {epoch_kl_loss}")

        if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            sampled_images = diffusion.sample(model, n = 16)                     ##images.shape[0] -> 16 to sample 16 images instead. Sample images after EACH epoch
            ema_sampled_images = diffusion.sample(ema_model, n = 16)
            print("sampled_images type: ", type(sampled_images), ", shape: ", sampled_images.shape)

            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))                         #save
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_{IMAGE_SIZE}_ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_{IMAGE_SIZE}_ckpt_ema.pt"))


    # *** record end time and calculate total training time ***
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {int(total_time // 3600)} hour(s), {int((total_time % 3600) // 60)} minute(s), {int(total_time % 60)} second(s)")

    #Plot loss function against number of epochs
    fig, ax = plt.subplots(figsize = (10, 10))

    ax.plot(range(args.epochs), epoch_loss_values, label = "Hybrid Loss", color = 'orange')
    ax.plot(range(args.epochs), epoch_simple_loss_values, label = "Simple Loss", color = 'blue')
    ax.plot(range(args.epochs), epoch_kl_loss_values, label = "KL Loss", color = 'green')
    ax.set_xlabel('Epoch', fontsize = 22)
    ax.set_ylabel('Loss', fontsize = 22)
    ax.set_title('Loss vs. Epochs', fontsize = 24)
    ax.legend()

    plt.savefig(os.path.join("results", args.run_name, "loss_plot.png"))
    np.save(os.path.join("results", args.run_name, "epoch_loss_values.npy"), np.array(epoch_loss_values))
    np.save(os.path.join("results", args.run_name, "epoch_simple_loss_values.npy"), np.array(epoch_simple_loss_values))
    np.save(os.path.join("results", args.run_name, "epoch_kl_loss_values.npy"), np.array(epoch_kl_loss_values))

    #return epoch_loss_values


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 2500       #500
    args.batch_size = 80     #12
    print("Batch size: ", args.batch_size)
    args.image_size = IMAGE_SIZE
    args.dataset_path = r"D:\SnapshotQuijote\QuijoteImages\QuijoteDatasetOneChannel\Test"
    #args.dataset_path = r"/data/cluster/mnoor/Quijote1Channel32"                    #6000 samples
    #args.dataset_path = r"/data/cluster/mnoor/Quijote1Channel32_8000"
    #args.dataset_path = r"/data/cluster/mnoor/QuijoteImages"

    args.device = "cpu"  #"cuda"
    args.lr = 1.5e-4      #try 1e-4, this is the value take in the IDDPM paper.
    train(args)


if __name__ == '__main__':
    launch()


