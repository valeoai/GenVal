import glob
import cv2
import numpy as np
import random
import torch
import json
import os
import argparse
import shutil

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline


# Utils functions
def read_img(im_path):
    return cv2.imread(im_path)[..., ::-1]  # Return RGB


def crop_resize(img, size_crop=384, size_mask=256):
    # Get position of the crop
    x = random.randint(img.shape[0]//4, img.shape[0] - size_crop)
    y = random.randint(0, img.shape[1] - size_crop)
    pos_mask_original = (x + (size_crop - size_mask)//2, y + (size_crop - size_mask)//2)
    pos_mask_inpaint = ((size_crop - size_mask)//2, (size_crop - size_mask)//2)

    # Get cropped img and mask
    crop_img = img[x: x+size_crop, y: y+size_crop]
    mask = np.zeros(shape=(size_crop, size_crop), dtype=np.uint8)
    mask[(size_crop - size_mask)//2: (size_crop + size_mask)//2, (size_crop - size_mask)//2: (size_crop + size_mask)//2] = 255

    # Then downsample to target resolution
    resized_img = cv2.resize(crop_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(mask, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)  # To keep either black or white values especially on the borders of the mask

    # Blur the mask so that the generation is smoother
    resized_mask = cv2.blur(resized_mask, (128, 128))

    return resized_img, resized_mask, pos_mask_original, pos_mask_inpaint


def get_pipe(silent=True, device=0):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=silent)
    return pipe


def inpaint(img, mask, prompt, pipe, num_images_per_prompt, guidance_scale):
    pil_img = Image.fromarray(img).convert('RGB')
    pil_mask = Image.fromarray(mask).convert('RGB')
    inpainted_imgs = pipe(prompt=prompt, 
                          image=pil_img, 
                          mask_image=pil_mask, 
                          num_images_per_prompt=num_images_per_prompt,
                          strength=1,  # Ensure full noisy image in mask region
                          guidance_scale=guidance_scale).images
    return [np.array(inpainted_img) for inpainted_img in inpainted_imgs]


def resize_to_original(imgs, size_crop=384):
    return [cv2.resize(img, dsize=(size_crop, size_crop), interpolation=cv2.INTER_CUBIC) for img in imgs]


def paste(original_img, resized_inpainted_imgs, pos_mask_original, pos_mask_inpaint, size_mask=256):
    # We want to paste the inpainted img on the original position where we cropped the image
    final_imgs = []
    final_img = original_img.copy()

    x_final_image = pos_mask_original[0]
    y_final_image = pos_mask_original[1]
    x_gen_image = pos_mask_inpaint[0]
    y_gen_image = pos_mask_inpaint[1]

    for resized_inpainted_img in resized_inpainted_imgs:
        final_img[x_final_image: x_final_image+size_mask, y_final_image: y_final_image+size_mask] = resized_inpainted_img[x_gen_image: x_gen_image+size_mask, y_gen_image: y_gen_image+size_mask]
        final_imgs.append(final_img)

    return final_imgs


def get_full_inpainted_img(pipe, original_img, prompt, num_images_per_prompt, guidance_scale):
    # Get random sized masks
    size_mask = random.randint(256, 512)
    size_crop = int(1.5*size_mask)

    # Crop and get size and position of the crop and the mask
    resized_img, resized_inpaint_mask, pos_mask_original, pos_mask_inpaint = crop_resize(original_img, size_crop, size_mask)

    # Inpaint and resize to original
    inpainted_imgs = inpaint(resized_img, resized_inpaint_mask, prompt, pipe, num_images_per_prompt, guidance_scale)
    resized_inpainted_imgs = resize_to_original(inpainted_imgs, size_crop)

    # Paste to get final image
    final_imgs = paste(original_img, resized_inpainted_imgs, pos_mask_original, pos_mask_inpaint, size_mask)

    return final_imgs, pos_mask_original, size_mask


def main():
    # Parser args
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="On which GPU to run on")
    args = parser.parse_args()
    device = args.device

    # Get input paths
    in_paths = glob.glob("../datasets/cityscapes/leftImg8bit/val/**/*.png", recursive=True)
    num_images_per_prompt = 1
    guidance_scale = 15

    with open('objects.txt', 'r') as f:
        objects = f.readlines()

    objects = [el.split("\n")[0] for el in objects]

    # Get good number of objects to generate for each GPU when running parallel jobs
    num_devices = torch.cuda.device_count()
    nb_obj_per_device = len(objects) / num_devices
    objects = objects[int(device/num_devices*len(objects)): int(device/num_devices*len(objects) + nb_obj_per_device)]

    print(f"Will generate examples for {len(objects)} different objects on device {device}: {objects}")

    pipe = get_pipe(silent=True, device=device)

    for obj in tqdm(objects):
        prompt = f"A photo of an {obj}"

        # Handling path
        path = f"../images/ood_objects/with_positions/{obj}"
        if not os.path.exists(path):
            os.makedirs(path)

        # To generate on multiple GPUs on different machines
        list_indexes = list(range(512))
        random.shuffle(list_indexes)

        for i in tqdm(list_indexes):
            if os.path.exists(f"{path}/{i}.png"):
                print(f"File {i}.png already exists, skipping...")
                continue

            else:
                crop_positions = {}
                in_path = random.choice(in_paths)
                original_img = read_img(in_path)
                final_imgs, pos_mask_original, size_mask = get_full_inpainted_img(pipe, original_img, prompt, num_images_per_prompt, guidance_scale)

                # Saving image
                concat_final_imgs = np.concatenate(final_imgs, 1)
                cv2.imwrite(f"{path}/{i}.png", concat_final_imgs[..., ::-1])
                crop_positions[f"{i}.png"] = {"pos_mask": pos_mask_original,
                                              "size_mask": size_mask}

                with open(os.path.join(path, f"{i}_pos.json"), "w") as f:
                    json.dump(crop_positions, f, indent=4)

                # Copying gt
                gt_path = in_path.replace("leftImg8bit", "gtFine").replace("gtFine.png", "gtFine_labelIds.png")
                shutil.copy(gt_path, os.path.join(path, f"{i}_gt.png"))

if __name__ == "__main__":
    main()