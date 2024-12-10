import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image



# the result dir, we set it to scratch dir, which has more storage space
RESULTS_DIR = "/scratch/kyq5pg/MLIA_final/results/"
GENERATED_IMAGES_PATH = "/scratch/kyq5pg/MLIA_final/samples/"



from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

""" 
    The Unet and GaussianDiffusion and Trainer instances should be the same as the on we used in training script 
"""
# Define the U-Net model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn = False    # we have to set this to False to solve the problem of RuntimeError: No available kernel. Aborting execution.
)
# Define the diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size=128,         # Image size
    timesteps=1000,         # Total diffusion timesteps
    sampling_timesteps=250  # Sampling timesteps (for faster sampling)
)
# Trainer configuration
trainer = Trainer(
    diffusion,
    './processed-datasets/train_data',
    # results_folder = './results',
    results_folder = RESULTS_DIR,
    train_lr=8e-5,
    train_batch_size = 32,
    train_num_steps=700000,         # Total training steps
    save_and_sample_every = 1000,
    # save_and_sample_every = 20,
    gradient_accumulate_every=2,    # Gradient accumulation steps
    ema_decay=0.995,                # Exponential moving average decay
    amp=True,                       # Enable mixed precision
    calculate_fid=False,              # Whether to calculate FID during training
)
    
    
    
"""
    sample and save images.
    images will be saved as being sampled every batch
"""
def sample_images(trainer, checkpoint_fullname, num_samples=1000):
    # generate synthetic images and save during sampling
    print("Generating and saving synthetic images...")
    # num_samples = 1000
    batch_size = trainer.batch_size

    # create subdirectory for generated images based on the checkpoint name
    checkpoint_name = checkpoint_fullname.split('.')[0]  # get "model-21" from "model-21.pt"
    sample_dir_of_checkpoint = os.path.join(GENERATED_IMAGES_PATH, checkpoint_name)
    os.makedirs(sample_dir_of_checkpoint, exist_ok=True)

    # start sampling and saving images batch by batch
    print(f"Sampling and Saving generated images to subdirectory: {sample_dir_of_checkpoint}...")
    counter = 0  # Track the total number of images saved
    for _ in tqdm(range(num_samples // batch_size), desc="Sampling", unit="batch"):
        with torch.no_grad():
            images = trainer.ema.ema_model.sample(batch_size=batch_size)  # Use the EMA model

        # Save each image in the current batch
        for img in images:
            # (img + 1) / 2 de-normalizes the images back to the range [0, 1] for proper visualization and saving.
            save_image((img + 1) / 2, f'{sample_dir_of_checkpoint}/sample_{counter}.png')
            counter += 1

    print(f"Saved {counter} images to {sample_dir_of_checkpoint}.")
    print("Finished!")

    
"""
    Load the model check point then sample images
"""    
# define the list of model checkpoints we need to load to sample for this time
# checkpoint_lst = [
#     "model-1.pt", 
#     "model-3.pt",
#     "model-5.pt",
# ]

# checkpoint_lst = [
#     "model-7.pt",
#     "model-17.pt", 
#     "model-19.pt",
# ]

# checkpoint_lst = [
#     "model-21.pt",
#     "model-23.pt",
#     "model-25.pt", 
# ]

# checkpoint_lst = [
#     "model-27.pt",
#     "model-29.pt",
#     "model-31.pt",
# ]

########################################################################
########################################################################

# checkpoint_lst = [
#     "model-9.pt", 
#     "model-33.pt",
#     "model-35.pt",
#     "model-22.pt",
# ]

# checkpoint_lst = [
#     "model-37.pt",
#     "model-39.pt", 
#     "model-41.pt",
#     "model-24.pt",
# ]

# checkpoint_lst = [
#     "model-2.pt",
#     "model-4.pt",
#     "model-6.pt", 
#     "model-26.pt",
# ]

# checkpoint_lst = [
#     "model-8.pt",
#     "model-18.pt",
#     "model-20.pt",
#     "model-28.pt",
# ]

########################################################################
########################################################################


# checkpoint_lst = [
#     "model-11.pt", 
#     "model-13.pt",
#     "model-15.pt",
#     "model-34.pt",
#     "model-42.pt",
#     "model-50.pt",
# ]

# checkpoint_lst = [
#     "model-43.pt",
#     "model-45.pt", 
#     "model-47.pt",
#     "model-36.pt",
#     "model-44.pt",
#     "model-14.pt",
# ]

# checkpoint_lst = [
#     "model-49.pt",
#     "model-10.pt",
#     "model-12.pt", 
#     "model-38.pt",
#     "model-46.pt",
#     "model-51.pt",
# ]

# checkpoint_lst = [
#     "model-16.pt",
#     "model-30.pt",
#     "model-32.pt",
#     "model-40.pt",
#     "model-48.pt",
# ]


########################################################################
########################################################################
### model-1   to   model-51 finished
########################################################################

# checkpoint_lst = [
#     "model-52.pt", 
#     "model-53.pt",
#     "model-54.pt",
#     "model-55.pt",
#     "model-56.pt",
#     "model-57.pt",
# ]

# checkpoint_lst = [
#     "model-58.pt",
#     "model-59.pt", 
#     "model-60.pt",
#     "model-61.pt",
#     "model-62.pt",
#     "model-63.pt",
# ]

# checkpoint_lst = [
#     "model-64.pt",
#     "model-65.pt",
#     "model-66.pt", 
#     "model-67.pt",
#     "model-68.pt",
#     "model-69.pt",
# ]

checkpoint_lst = [
    "model-70.pt",
    "model-71.pt",
    "model-72.pt",
    "model-73.pt",
    "model-74.pt",
    "model-75.pt",
]



# load and sample for each checkpoint given in the list
for checkpoint in checkpoint_lst:
    milestone = int(checkpoint.split('-')[1].split('.')[0])
    # Load the checkpoint 
    trainer.load(milestone)
    print(f"Check point {checkpoint} is loaded, which was trained with {milestone * 1000} training steps.")
    # start to sample images for this checkpoint
    sample_images(trainer, checkpoint, num_samples=1000)

    
print("All finished!")