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
    results_folder = './results-10000steps',
    train_lr=8e-5,
    train_batch_size = 32,
    train_num_steps=10000,         # Total training steps
    save_and_sample_every = 500,
    gradient_accumulate_every=1,    # Gradient accumulation steps
    ema_decay=0.995,                # Exponential moving average decay
    amp=True,                       # Enable mixed precision
    calculate_fid=True,              # Whether to calculate FID during training
)

# Start training
trainer.train()