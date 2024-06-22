import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

IMAGE_SIZE = 32

def plot_images(images):
    plt.figure(figsize = (40, 40))   #32, 32
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim = -1),
    ], dim = -2).permute(1, 2, 0).cpu())
    plt.show()


""""
#OLD: for jpg images.
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)     
    print("Grid shape: ", grid.shape, ", grid type: ", type(grid))
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()           
    #print("ndarr shape: ", ndarr.shape, ", ndarr type: ", type(ndarr))
    im = Image.fromarray(ndarr)
    im.save(path)
"""

"""
def save_images(images, path):
    num_images = images.size(0)  
    num_rows = 2
    num_cols = num_images // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))  

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j] if num_images > 1 else axs  
            index = i * num_cols + j
            each_image = images[index].squeeze().cpu().numpy()  
            ax.imshow(each_image, origin='lower', cmap='gist_heat')
            ax.axis('off')

    plt.tight_layout()  
    plt.savefig(path)   

    plt.show()
"""

def save_images(images, path):
    num_images = images.size(0)  #Get the number of images
    num_rows = 4
    num_cols = num_images // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))  #Create subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  #Set spacing between rows and columns

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j] if num_images > 1 else axs
            index = i * num_cols + j
            each_image = images[index].squeeze().cpu().numpy()
            ax.imshow(each_image, origin='lower', cmap='gist_heat')
            ax.axis('off')

    plt.savefig(path)

    #plt.show()

def save_image_samples(image, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(image.squeeze().cpu().numpy(), origin='lower', cmap='gist_heat')
    plt.axis('off')
    plt.savefig(path)
    plt.close()



#For noising_test:
def save_images2(images, path, **kwargs):
    num_images = images.size(0)  #Get the number of images. images: ([8, 1, 64, 64])
    fig, axs = plt.subplots(1, num_images, figsize = (8 * num_images, 8))  #Create subplots

    for i in range(num_images):
        ax = axs[i] if num_images > 1 else axs
        each_image = images[i, 0].cpu().numpy()
        ax.imshow(each_image, origin = 'lower', cmap = 'gist_heat')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path)

    plt.show()


def get_data(args):
    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size. Resize image to a square.
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale = (0.8, 1.0)), #performs a random crop of the image with the specified size args.image_size
        #torchvision.transforms.ToTensor(), #also normalizes the pixel values to the range [0.0, 1.0].
        #torchvision.transforms.Normalize((0.5,), (0.5,)) #Since out input has one channel only
    ])

    #Dataset contains .pt files
    dataset = torchvision.datasets.DatasetFolder(args.dataset_path, loader = torch.load, extensions = ('.pt',))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)
    return dataloader

"""
def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size. Resize image to a square.
        torchvision.transforms.RandomResizedCrop(args.image_size, scale = (0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform = transforms)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)
    return dataloader
"""


def setup_logging(run_name):
    os.makedirs("models", exist_ok = True)
    os.makedirs("results", exist_ok = True)
    os.makedirs(os.path.join("models", run_name), exist_ok = True)
    os.makedirs(os.path.join("results", run_name), exist_ok = True)
