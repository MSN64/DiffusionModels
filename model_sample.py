import os
import torch
from my_modules import UNet
from diffusion_model import Diffusion
from PIL import Image
from matplotlib import pyplot as plt
import utils
import time

IMAGE_SIZE = 32

save_dir = "/data/cluster/mnoor/DiffusionModels/venv/model_32_2500_batch_80_T4000"
img_dir = "/data/cluster/mnoor/DiffusionModels/venv/model_32_2500_batch_80_T4000/generated_images"

device = "cuda"
model = UNet().to(device)
diffusion = Diffusion(img_size=IMAGE_SIZE, device=device)

#List of epochs to load models from
epochs = [2499, 1499, 499]

start_time = time.time()

for epoch in epochs:
    for model_type in ['ema', 'non_ema']:
        model_label = f"epoch_{epoch}_{model_type}"
        print(f"Loading {model_type} model from epoch {epoch}:")
        if model_type == 'ema':
            ckpt_path = f"/data/cluster/mnoor/DiffusionModels/venv/models/DDPM_Unconditional/{epoch}_32_ckpt_ema.pt"
            save_subdir = os.path.join(save_dir, "EMA", f"epoch_{epoch}")
            img_subdir = os.path.join(img_dir, "EMA", f"epoch_{epoch}")
        else:
            ckpt_path = f"/data/cluster/mnoor/DiffusionModels/venv/models/DDPM_Unconditional/{epoch}_32_ckpt.pt"
            save_subdir = os.path.join(save_dir, "Non_EMA", f"epoch_{epoch}")
            img_subdir = os.path.join(img_dir, "Non_EMA", f"epoch_{epoch}")

        os.makedirs(save_subdir, exist_ok=True)
        os.makedirs(img_subdir, exist_ok=True)

        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

        print(f"Generating samples for {model_label}:")
        generated_tensors = []

        for i in range(600):  #Generate 600 samples
            print(f"Sample {i}:")
            x = diffusion.sample(model, n = 1)  #Generate one sample at a time
            generated_tensors.append(x.squeeze(0))
            if (i + 1) % 50 == 0:
                utils.save_image_samples(x.squeeze(0), os.path.join(img_subdir, f"sample_{i}_{model_type}_{epoch}.png"))

            torch.save(x, os.path.join(save_subdir, f"sample_{i}_{model_type}_{epoch}.pt"))

print("Sampling completed for all epochs and model types.")

end_time = time.time()
total_time = end_time - start_time
print(
    f"Total time: {int(total_time // 3600)} hour(s), {int((total_time % 3600) // 60)} minute(s), {int(total_time % 60)} second(s)")