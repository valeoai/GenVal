import os
import glob
import torch
import random
import json
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline


# Refine model from HF
def get_refine_model(device=0):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None)
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(device)
    return pipeline


if __name__ == "__main__":
    # Parser for args
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="On which GPU to run on")
    args = parser.parse_args()
    device = args.device

    # Get refine model
    refine_model = get_refine_model(device)

    # Get crop positions paths
    root_path = "../images/ood_objects/with_positions/"
    positions_paths = glob.glob(f"{root_path}**/*.json", recursive=True)
    random.shuffle(positions_paths)

    save_path = "../images/ood_objects/refined/"

    for position_path in tqdm(positions_paths):
        obj = position_path.split("/")[-2]
        positions = json.load(open(position_path, 'r'))
        file = next(iter(positions))
        positions = positions[file]

        if not os.path.exists(os.path.join(save_path, obj)):
            os.makedirs(os.path.join(save_path, obj))

        if os.path.exists(os.path.join(save_path, obj, file)):
            print(f"{os.path.join(save_path, obj, file)} already exists, skipping...")
            continue

        else:
            # Get image to refine
            image_to_refine = os.path.join(root_path, obj, file)
            image_to_refine = Image.open(image_to_refine).convert('RGB')
            
            # Get mask
            mask = np.zeros(shape=image_to_refine.size[::-1])
            size_crop = positions["size_mask"]
            x, y = positions["pos_mask"]
            mask[x: x+size_crop, y: y+size_crop] = 1

            # Refine
            refined_image = refine_model(prompt=obj, image=image_to_refine, mask_image=mask, height=1024, width=2048, strength=0.65).images[0]

            # Save refined image
            refined_image = refined_image.save(os.path.join(save_path, obj, file))
