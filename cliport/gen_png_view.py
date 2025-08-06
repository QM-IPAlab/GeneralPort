import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from natsort import natsorted
import torch
import os
import imageio

with open("~/data/put-block-in-bowl-seen-colors-test/color/000000-10001.pkl", "rb") as f:
    data = pkl.load(f)

print(type(data))
print(data.shape)

color_dir = "~/concept-fusion/data/bowl/view/color"

for sample_idx, sample in enumerate(data):
    print(f"Sample {sample_idx}:")
    sample_dir = f"{sample_idx:05}"
    os.makedirs(color_dir, exist_ok=True)
    
    for frame_idx, frame in enumerate(sample):
        print(f" Saving Frame {frame_idx}:")
        # save .png files
        color_filename = os.path.join(color_dir, f"{frame_idx:05}.png")
        imageio.imwrite(color_filename, frame)  # color(H, W, 3)

        # # Matplotlib 
        # plt.imshow(frame.astype('uint8'))  
        # plt.title(f"Sample {sample_idx}, Frame {frame_idx}")
        # plt.axis('off') 
        # plt.show()
    break