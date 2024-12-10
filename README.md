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
For the process of `accelerate config`, we recommend you follow the following steps if you have never used it on the UVA Rivanna. If you are familiar with it, just ignore this tutorial.
```bash
bash-4.4$accelerate config
----------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
```

```bash
Which type of machine are you using?                                                                                              
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training                                                                                                       
    multi-CPU                                                                                                                     
    multi-XPU                                                                                                                     
 ➔  multi-GPU
    multi-NPU
    multi-MLU
    multi-MUSA
    TPU
```


```bash
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                          
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:    
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                   
Do you want to use DeepSpeed? [yes/NO]:                                                                                           
Do you want to use FullyShardedDataParallel? [yes/NO]:                                                                            
Do you want to use Megatron-LM ? [yes/NO]:                                                                                        
How many GPU(s) should be used for distributed training? [1]:4 
```

For the last line above, you should set it to the number of GPUs you want to use. Here I used 4.


```bash
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:                                 
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: yes
----------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
Please select a choice using the arrow or number keys, and selecting with enter
    no                                                                                                                            
 ➔  fp16
    bf16
    fp8
```


Then you can run our training script on multiple GPUs with the following command:
```bash
accelerate launch train-700000.py
```

## Step 5
After the training process, or after getting some training checkpoints even if not finished all training steps, you can start to run our sample scripts to sample images. We have two sample scripts, one is called `sample_latest_checkpoint.py`, and the other is called `sample.py`.

The `sample_latest_checkpoint.py` script will automatically find the latest trained model checkpoint and load it then sample 5,000 images from pure noise using this model checkpoint.

The `sample.py` script can sample 1,000 images for each of the model checkpoints in a list of defined model checkpoints.

If you want to run `sample_latest_checkpoint.py`, please first open it and modify the code in line 2 and 3 like the following:
```python
RESULTS_DIR = "/scratch/kyq5pg/MLIA_final/results/"
GENERATED_IMAGES_PATH = "/scratch/kyq5pg/MLIA_final/samples/"
```
You should change them to your own dirs. `RESULTS_DIR` should be the dir storing your model checkpoints, which you have configured in `train-700000.py` file. The `GENERATED_IMAGES_PATH` should be the dir you want it to store the sampled images.
Then you can use the following command to run it:
```
python sample_latest_checkpoint.py
```

If you want to run `sample.py`, similarly, you should open it and find the code in line 9 and 10 to modify the `RESULTS_DIR` and `GENERATED_IMAGES_PATH` to your own dirs.

Additionally, for `sample.py`, you have to find the code in line 214 like the following:
```python
checkpoint_lst = [
    "model-70.pt",
    "model-71.pt",
    "model-72.pt",
    "model-73.pt",
    "model-74.pt",
    "model-75.pt",
]
```
You should change it to a list of model checkpoints you want to sample images for.

After that, you can run `sample.py`, which supports using HuggingFace `accelerate` to assign the task to a specific GPU. You can use the following command:
```bash
accelerate config
```
This time, you should follow the following instruction to finish the config. 
```
Which type of machine are you using?                                                                                              
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  No distributed training
    multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    multi-MLU
    multi-MUSA
    TPU
```
```bash
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO 
```
```bash
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:0
```
This is for selecting the index of your GPU to be assigned with this task.

After that, you can run the following command to start the sampling task:
```bash
accelerate launch sample.py
```

After that, if you want to run another sampling task on another GPU in parallel, you can just run the command `accelerate config` again to assign it to another GPU, and then modify the `checkpoint_lst` in `sample.py` to another list of checkpoints, then use the command `accelerate launch sample.py` to start the second task.


## Step 6
After sampling images for several model checkpoints, now you can start to evaluate them using our `ddpm_evaluate.ipynb` notebook. You need to open the file to do some customizations. First, you should locate the 3rd cell, and modify the `GENERATED_IMAGES_ROOT` to your dir where is storing the sampled images, which should be the one you have configured in the `sample_latest_checkpoint.py` and `sample.py` files. You do NOT need to modify any other dirs in this notebook.

Then you may need to change the number of model checkpoints you want to evaluate, because, by default, we hard-coded it to 75 in our experiments. If you do not have sampled images for the first 75 checkpoints, please modify the following line in the 7th cell to yours:
```python
for checkpoint in checkpoints[:75]:
```
You may have the concern that in Python, if a list has a length smaller than 75, the code `checkpoints[:75]` will be okay, and just read all of the items in the list. So why need to change this? But, in this loop, we require the checkpoint being evaluated must have sample images in the sample process, otherwise, it will be meaningless. But, sometimes, you may have a model checkpoint, but never sampled images for it, so it will cause some problems. Therefore, we suggest you change this value in the code to yours.

After that, you can run all the cells in the `ddpm_evaluate.ipynb` notebook for evaluation. After all the cells finished, it will produce the following images and csv files in your project root:
```
FID_vs_TrainingSteps.png
IS_vs_TrainingSteps.png
FID_vs_TrainingSteps_First25.png
IS_vs_TrainingSteps_First25.png
evaluation_results.csv                  # storing the FID, IS Mean, IS Std for each model checkpoint evaluated
best_scores.csv                         # storing the model checkpoint with the best FID, and the model checkpoint with the best IS Mean
```

