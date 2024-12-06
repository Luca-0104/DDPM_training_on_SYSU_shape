import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


# Define the U-Net model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn = True
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
    results_folder = './results',
    train_lr=8e-5,
    train_batch_size = 32,
    train_num_steps=700000,         # Total training steps
    save_and_sample_every = 1000,
    gradient_accumulate_every=2,    # Gradient accumulation steps
    ema_decay=0.995,                # Exponential moving average decay
    amp=True,                       # Enable mixed precision
    calculate_fid=False,              # Whether to calculate FID during training
)


# Check for the latest checkpoint
latest_milestone = None
for checkpoint in sorted(os.listdir('./results'), reverse=True):
    if checkpoint.startswith('model-') and checkpoint.endswith('.pt'):
        latest_milestone = int(checkpoint.split('-')[1].split('.')[0])
        break

if latest_milestone is not None:
    print(f"Resuming from checkpoint at step {latest_milestone * 1000}")
    trainer.load(latest_milestone)  # Load the latest checkpoint
    

# Start training
trainer.train()