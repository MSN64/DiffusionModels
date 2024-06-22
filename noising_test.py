import torch
from torchvision.utils import save_image
from diffusion_model import Diffusion
from utils import get_data
from utils import save_images2
import argparse
from torch import nan
import numpy as np

IMAGE_SIZE = 32

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = IMAGE_SIZE

#args.dataset_path = r"D:\SnapshotQuijote\QuijoteImages\QuijoteDatasetOneChannel\Tensors"
#args.dataset_path = r"D:\SnapshotQuijote\QuijoteNoise" #
args.dataset_path = r"D:\SnapshotQuijote\QuijoteImages\QuijoteDatasetOneChannel\NoiseCheck"


def check_for_nan_inf(tensor, name=""):
    nan_exists = torch.isnan(tensor).any()
    inf_exists = torch.isinf(tensor).any()
    if nan_exists or inf_exists:
        print(f"{name} contains NaN: {nan_exists}, Inf: {inf_exists}")
        return True
    return False



dataloader = get_data(args)
print("Type: ", type(dataloader))

diff = Diffusion(device = "cpu") #"cuda"

image = next(iter(dataloader))[0]
print("Image shape: ", image.shape)
print("Image type: ", type(image))
print("Max: ", torch.max(image))
print("Min: ", torch.min(image))


#t = torch.Tensor([0, 150, 200, 300, 450, 600, 800, 999]).long()   #save noised images of FP at these timesteps
t = torch.Tensor([0, 200, 400, 600, 800, 1000, 1999, 3999]).long()
noised_image, _ = diff.noise_images(image, t)                      #Noised image shape:  torch.Size([8, 1, img_size, img_size]), batch of 8 images.


nan_values = torch.isnan(noised_image).any()
print("Nan in noised_image?", nan_values)

inf_values = torch.isinf(noised_image).any()
print("Inf in noised_image?", inf_values)


print("Noised image type: ", type(noised_image))                  #Noised image type:  <class 'torch.Tensor'>
print("Noised image shape: ", noised_image.shape)
print("Noised image 0 shape: ", noised_image[0].shape)            #Noised image 0 shape:  torch.Size([1, img_size, img_size])
print("Max of noised image 0: ", torch.max(noised_image[0]))
print("Min of noised image 0: ", torch.min(noised_image[0]))

print("Max of noised image 3: ", torch.max(noised_image[3]))
print("Min of noised image 3: ", torch.min(noised_image[3]))

print("Max of noised image 6: ", torch.max(noised_image[6]))
print("Min of noised image 6: ", torch.min(noised_image[6]))

print("Max of noised image 7: ", torch.max(noised_image[7]))
print("Min of noised image 7: ", torch.min(noised_image[7]))

#print("Noised_image values cosine:", noised_image)

save_images2(noised_image.add(1).mul(0.5), r"D:\SnapshotQuijote\QuijoteImages\QuijoteDatasetOneChannel\NoiseResults\timestep_4000_02468102040_betabounds_32.png")