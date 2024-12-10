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
    Load the latest model check point for sampling
"""
import os

latest_milestone = None

# Filter and sort checkpoints by the numerical milestone
# checkpoints = [
#     checkpoint for checkpoint in os.listdir(RESULTS_DIR)
#     if checkpoint.startswith('model-') and checkpoint.endswith('.pt')
# ]
# checkpoints = sorted(
#     checkpoints,
#     key=lambda x: int(x.split('-')[1].split('.')[0]),
#     reverse=True
# )

# if not checkpoints:
#     raise FileNotFoundError("No checkpoints found in the results directory!")

# # Get the latest checkpoint
# if checkpoints:
#     latest_checkpoint = checkpoints[0]
#     latest_milestone = int(latest_checkpoint.split('-')[1].split('.')[0])
    
#     # Load the checkpoint 
#     trainer.load(latest_milestone)
    
#     print(f"Check point {latest_checkpoint} is loaded, which was trained with {latest_milestone * 1000} training steps.")
# else:
#     print("No checkpoint found.")
    
    
    

# for test
latest_checkpoint = "model-21.pt"
latest_milestone = int(latest_checkpoint.split('-')[1].split('.')[0])
# Load the checkpoint 
trainer.load(latest_milestone)
print(f"Check point {latest_checkpoint} is loaded, which was trained with {latest_milestone * 1000} training steps.")    
    
 

# Use Accelerator for multi-GPU
from accelerate import Accelerator
# Initialize the accelerator
accelerator = Accelerator()

from ema_pytorch import EMA
# Check if Trainer's EMA exists, and create it if necessary
if not hasattr(trainer, 'ema') or not trainer.ema:
    # initialize it manually using the same logic as in the Trainer class source code!
    print("Initializing EMA model dynamically...")
    trainer.ema = EMA(
        diffusion,      # Pass the model from Trainer
        beta=0.995,     # Match the EMA decay value used during training
        update_every=10 # Match the update frequency
    )
    trainer.ema.to(accelerator.device)  # Move EMA to the appropriate device

# Prepare EMA model for multi-GPU
trainer.ema.ema_model = accelerator.prepare(trainer.ema.ema_model)

    
    
"""
    sample and save images.
    images will be saved as being sampled every batch
"""
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os

# generate synthetic images and save during sampling
print("Generating and saving synthetic images...")
num_samples = 5000
# batch_size = trainer.batch_size
scaled_batch_size = trainer.batch_size * accelerator.num_processes  # Scale batch size for multiple GPUs


# create subdirectory for generated images based on the checkpoint name
checkpoint_name = latest_checkpoint.split('.')[0]  # get "model-21" from "model-21.pt"
sample_dir_of_checkpoint = os.path.join(GENERATED_IMAGES_PATH, checkpoint_name)
os.makedirs(sample_dir_of_checkpoint, exist_ok=True)


# Only show progress bar on the main process
if accelerator.is_main_process:
    progress_bar = tqdm(range(num_samples // scaled_batch_size), desc="Sampling", unit="batch")
else:
    progress_bar = range(num_samples // scaled_batch_size)
    
    

# start sampling and saving images batch by batch
print(f"Sampling and Saving generated images to subdirectory: {sample_dir_of_checkpoint}...")
counter = 0  # Track the total number of images saved
for _ in progress_bar:
    with torch.no_grad():
        images = trainer.ema.ema_model.sample(batch_size=scaled_batch_size)  # Use the EMA model
    
    # Save each image in the current batch
    for img in images:
        # Save Images on the Main Process Only
        if accelerator.is_main_process:
            # (img + 1) / 2 de-normalizes the images back to the range [0, 1] for proper visualization and saving.
            save_image((img + 1) / 2, f'{sample_dir_of_checkpoint}/sample_{counter}.png')
        counter += 1

print(f"Saved {counter} images to {sample_dir_of_checkpoint}.")
print("Finished!")
