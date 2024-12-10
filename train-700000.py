import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


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

RESULTS_DIR = "/scratch/kyq5pg/MLIA_final/results/"

# Trainer configuration
trainer = Trainer(
    diffusion,
    './processed-datasets/train_data',
    # results_folder = './results',
    results_folder = RESULTS_DIR,
    train_lr=1e-5,    # decreased from 8e-5
    train_batch_size = 32,
    train_num_steps=700000,         # Total training steps
    save_and_sample_every = 1000,
    # save_and_sample_every = 20,
    gradient_accumulate_every=4,    # Gradient accumulation steps (increased from 2, to reduce training instability)
    ema_decay=0.990,                # Exponential moving average decay  (decreased from 0.995 for faster ema updates)
    amp=True,                       # Enable mixed precision
    calculate_fid=False,              # Whether to calculate FID during training
)


# for test
# print("========================================================================")
# print("========================================================================")
# print("========================================================================")
# print(sorted(os.listdir(RESULTS_DIR), reverse=True))

""" for loading the latest model check point to continue training """
# Check for the latest checkpoint
latest_milestone = None

# Filter and sort checkpoints by the numerical milestone
checkpoints = [
    checkpoint for checkpoint in os.listdir(RESULTS_DIR)
    if checkpoint.startswith('model-') and checkpoint.endswith('.pt')
]
checkpoints = sorted(
    checkpoints,
    key=lambda x: int(x.split('-')[1].split('.')[0]),
    reverse=True
)

# Get the latest checkpoint
if checkpoints:
    latest_checkpoint = checkpoints[0]
    latest_milestone = int(latest_checkpoint.split('-')[1].split('.')[0])
    print(f"Resuming from checkpoint at step {latest_milestone * 1000}")
    trainer.load(latest_milestone)  # Load the latest checkpoint
else:
    print("No checkpoint found.")
    

# bugs with following code for loarding the latest checkpoint
# # Check for the latest checkpoint
# latest_milestone = None
# for checkpoint in sorted(os.listdir(RESULTS_DIR), reverse=True):
#     if checkpoint.startswith('model-') and checkpoint.endswith('.pt'):
#         latest_milestone = int(checkpoint.split('-')[1].split('.')[0])
#         break

# if latest_milestone is not None:
#     print(f"Resuming from checkpoint at step {latest_milestone * 1000}")
#     trainer.load(latest_milestone)  # Load the latest checkpoint
    

# Start training
trainer.train()