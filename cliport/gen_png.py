"""在cliport/cliport文件夹中运行"""

import os
import cv2
import imageio
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from cliport.utils import utils
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

n_demos = 100
mode = "test"
task = 'put-block-in-bowl-seen-colors'

root_dir = '/home/a/acw799/cliport'
data_dir = '/home/a/acw799'
assets_root = os.path.join(root_dir, 'cliport/environments/assets/')
config_file = 'eval.yaml' 
vcfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
vcfg['data_dir'] = os.path.join(data_dir, 'data')
vcfg['mode'] = mode
vcfg['task'] = task
vcfg['train_config'] = "cfg/inference.yaml"
tcfg = utils.load_hydra_config(vcfg['train_config'])

# Load dataset
ds = RavensDataset(os.path.join(vcfg['data_dir'], f'{vcfg["task"]}-{vcfg["mode"]}'), 
                   tcfg, 
                   n_demos=n_demos,
                   augment=False)

# Initialize environment and task.
env = Environment(
    assets_root,
    disp=False,
    shared_memory=False,
    hz=480,
    record_cfg=False
)

# create file path
color_dir = "~/concept-fusion/data/bowl/color"
depth_dir = "~/concept-fusion/data/bowl/depth"

os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

for i in tqdm(range(n_demos), desc="saving color and depth:", total=n_demos):
    episode, seed = ds.load(i)

     # Set task
    task_name = vcfg['task']
    task = tasks.names[task_name]()
    task.mode = mode

    # Set environment
    env.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = env.info
    reward = 0
         
    # Get batch
    batch = ds.process_goal((obs, None, reward, info), perturb_params=None)
    # break
    
    # Get color and depth inputs
    img = batch['img']
    img = torch.from_numpy(img)
    color = np.uint8(img.detach().cpu().numpy())[:,:,:3]
    color = color.transpose(1,0,2)

    depth = np.array(img.detach().cpu().numpy())[:,:,3]
    depth = depth.transpose(1,0)

    # save .png files
    color_filename = os.path.join(color_dir, f"{i:05}.png")
    depth_filename = os.path.join(depth_dir, f"{i:05}.png")
    imageio.imwrite(color_filename, color)  # color(H, W, 3)
    
    normalized_depth = (255 * (depth / depth.max())).astype(np.uint8)  
    imageio.imwrite(depth_filename, normalized_depth)