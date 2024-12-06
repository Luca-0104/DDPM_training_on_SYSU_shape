from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Define the U-Net model
# model = Unet(
#     dim=64,
#     dim_mults=(1, 2, 4, 8),
#     flash_attn = True
# )
model = Unet(
    dim=32,
    dim_mults=(1, 2),
    flash_attn = True
)

# Define the diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size=128,         # Image size
    # timesteps=1000,         # Total diffusion timesteps
    timesteps=100,         # Total diffusion timesteps
    # sampling_timesteps=250  # Sampling timesteps (for faster sampling)
    sampling_timesteps=50  # Sampling timesteps (for faster sampling)
)

# Trainer configuration
trainer = Trainer(
    diffusion,
    './processed-datasets/train_data',
    results_folder = './quicktrain_results',
    # train_lr=8e-5,
    train_lr=1e-5,
    train_batch_size = 16,
    # train_num_steps=700000,         # Total training steps
    train_num_steps=1000,         # Total training steps
    gradient_accumulate_every=2,    # Gradient accumulation steps
    # gradient_accumulate_every=1,    # Gradient accumulation steps
    ema_decay=0.995,                # Exponential moving average decay
    amp=True,                       # Enable mixed precision
    calculate_fid=False,              # Whether to calculate FID during training
    # save_and_sample_every = 1000,
    save_and_sample_every = 100,
    
)

# Start training
trainer.train()