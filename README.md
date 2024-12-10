# Denoising Diffusion Probabilistic Models (DDPM) Training on SYSU-Shape Dataset

The experiment presents our exploration of Denoising Diffusion Probabilistic Models (DDPM) on the SYSU-Shape dataset. Leveraging an existing DDPM implementation, we designed a comprehensive pipeline for dataset preprocessing, model training, image sampling, and evaluation. The training was conducted on UVA Rivanna's high-performance computing infrastructure, utilizing multi-GPU support via Hugging Face's accelerate library. To overcome time constraints, we implemented checkpoint-based training resumption and systematically adjusted hyperparameters to address challenges such as overfitting and training instability on the smaller dataset. The model’s performance was evaluated using Frechet Inception Distance (FID) and Inception Score (IS) across 75,000 training steps. While our DDPM model achieved an IS of 5.6 and an FID of 175.2, it displayed reasonable diversity and quality in generated images, highlighting its capacity to adapt to the unique characteristics of the SYSU-Shape dataset. These results underline opportunities for improvement in hyperparameter optimization and training strategies to further enhance generative performance.

# Usage Instructions

Here we provide a comprehensive instruction on how to run our experiments on the UVA Rivanna server.

## Step 1
You should clone the SYSU-Shape dataset from its GitHub repo first, using the following command at the root dir of the current project:
```bash
git clone https://github.com/bearpaw/sysu-shape-dataset.git
```
After that, you should see a dir called `sysu-shape-dataset` at your root dir, which is the cloned dataset

## Step 2
Now, you should open the Jupiter notebook file `ddpm_SYSU.ipynb` and run all the cells, which are for data processing. After that, you should see a new folder called `processed-dataset` in your project root, which contains 3 subfolders, called `eval_data`, `train_data` and `normalized_eval_data`.

## Step 3
Now, you need to prepare the existing DDPM implementation. As they support to be installed using `pip`, you can run the following code to get their source code:
```bash
pip install denoising_diffusion_pytorch
```

After that, before starting training using their model implementation, you have to modify a bit of their model to make sure there will not be an `Error: Cannot allocate memory` error on the Rivanna server. You may want to follow the instructions below:
```bash
pip show denoising_diffusion_pytorch
```
This command will show you where your denoising_diffusion_pytorch source code is installed. It has the output similar to the following:
```
bash-4.4$pip show denoising-diffusion-pytorch
Name: denoising-diffusion-pytorch
Version: 2.1.1
Summary: Denoising Diffusion Probabilistic Models - Pytorch
Home-page: https://github.com/lucidrains/denoising-diffusion-pytorch
Author: Phil Wang
Author-email: lucidrains@gmail.com
License: MIT
Location: /home/kyq5pg/.local/lib/python3.11/site-packages
Requires: accelerate, einops, ema-pytorch, numpy, pillow, pytorch-fid, scipy, torch, torchvision, tqdm
```
In my example above, the location of it is
```
/home/kyq5pg/.local/lib/python3.11/site-packages
```
Therefore, you can find the source code here:
```
bash-4.4$ls /home/kyq5pg/.local/lib/python3.11/site-packages/denoising_diffusion_pytorch
attend.py                              guided_diffusion.py            repaint.py
classifier_free_guidance.py            __init__.py                    simple_diffusion.py
continuous_time_gaussian_diffusion.py  karras_unet_1d.py              version.py
denoising_diffusion_pytorch_1d.py      karras_unet_3d.py              v_param_continuous_time_gaussian_diffusion.py
denoising_diffusion_pytorch.py         karras_unet.py                 weighted_objective_gaussian_diffusion.py
elucidated_diffusion.py                learned_gaussian_diffusion.py
fid_evaluation.py                      __pycache__
```
You have to open the source code file called `denoising_diffusion_pytorch.py` and locate the line 947, where they originally have the following code:
```
dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
```
You should substitute this line of code with the following if you are going to use 4 GPU to train:
```
dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
```


## Step 4
Before running our training script `train-700000.py` on GPU, you should open this file to modify the code in line 22, which is the directory for storing the model checkpoints and quick samples for each checkpoint. It looks like the following:
```python
RESULTS_DIR = "/scratch/kyq5pg/MLIA_final/results/"
```
This is currently, my current scratch dir on UVA Rivanna. You should change it to your desired dir.

After that, please go to your terminal to install and configure the Hugging Face `accelerate` with the following commands, which is for supporting the multi-GPU training.
```bash
pip install accelerate
```
```bash
accelerate config
```
Then you can run our training script on multiple GPU with the following comsmand:
```bash
accelerate launch train-700000.py
```

## Step 5
After the training process, or after getting some training checkpoints even if not finished all training steps, you can start to run our sample scripts to sample images. We have two sample scripts, one is called `sample_latest_checkpoint.py`, and the other is called `sample.py`.

The `sample_latest_checkpoint.py` script will automatically find the



