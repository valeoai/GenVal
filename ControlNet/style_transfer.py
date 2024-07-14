import cv2
import torch
import einops
import numpy as np
import random
import config
import argparse
import os
import os.path as osp

from share import *
from annotator.util import resize_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from dataset import Cityscapes


# Utils
def get_model(path_ckpt):
    # Loading model
    path_model = 'models/cldm_v15_small.yaml'
    model = create_model(path_model).cpu()
    model.load_state_dict(load_state_dict(path_ckpt, location='cuda'))
    model = model.cuda()
    model.eval()
    model.sd_locked = True
    model.only_mid_control = False
    return model


def get_sampler(model):
    return DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler):
    with torch.no_grad():
        img = resize_image(input_image, image_resolution)
        H, W = img.shape[:2]
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        control = torch.from_numpy(img.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results[0]


if __name__ == "__main__":
    # Parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default="", help='Domain to which we want to generate')
    parser.add_argument('--seed', type=int, default=-1, help="Seed to use when generating")
    parser.add_argument('--num_samples', type=int, default=512, help="Number of samples to generate")
    parser.add_argument('--ddim_steps', type=int, default=25, help="Number of DDIM steps")
    args = parser.parse_args()

    # Symlink Cityscapes dataset
    if not osp.exists('training/target'):
        os.symlink(osp.realpath('../datasets/cityscapes/leftImg8bit'), 'training/target')

    if not osp.exists('training/seg'):
        os.symlink(osp.realpath('../datasets/cityscapes/gtFine'), 'training/seg')

    # Get dataset and model
    dataset = Cityscapes(split="val")
    cities = os.listdir(osp.join("training/seg", dataset.split))
    path_ckpt = '../checkpoints/controlnet_cs.ckpt'

    model = get_model(path_ckpt)
    ddim_sampler = get_sampler(model)
    
    # Hparams
    a_prompt = ""
    n_prompt = ""
    num_samples = args.num_samples
    image_resolution = 512
    ddim_steps = args.ddim_steps
    guess_mode = False
    strength = 1.0
    scale = 8.0
    seed = args.seed
    eta = 0.0
    domain = args.domain

    # Save paths
    path_gen = f"../images/style_transfer/num_samples_{num_samples}/{seed}/{domain}/leftImg8bit/{dataset.split}"
    path_gt = f"../images/style_transfer/num_samples_{num_samples}/{seed}/{domain}/gtFine/{dataset.split}"
    path_original = f"../images/style_transfer/num_samples_{num_samples}/{seed}/{domain}/original/{dataset.split}"

    if not osp.exists(path_gen):
        os.makedirs(path_gen)
        for city in cities:
            os.makedirs(osp.join(path_gen, city))

    if not osp.exists(path_gt):
        os.makedirs(path_gt)
        for city in cities:
            os.makedirs(osp.join(path_gt, city))

    if not osp.exists(path_original):
        os.makedirs(path_original)
        for city in cities:
            os.makedirs(osp.join(path_original, city))

    # Main loop
    for num in range(num_samples):
        i = random.randint(0, len(dataset)-1)
        data = dataset[i]

        name_file = f"{domain}_{num}_" + dataset.data[i]["target"].split("/")[-1]
        city = dataset.data[i]["target"].split("/")[-2]

        original_image = ((data["jpg"]+1)*127.5).astype(int)
        input_image = data["hint"]
        prompt = data["txt"] + f", in {domain}"

        gen_image = process(input_image, prompt, a_prompt, n_prompt, 1, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler)
        gt = (input_image * 34 - 1).astype(int)

        cv2.imwrite(osp.join(path_gen, city, name_file), gen_image[..., ::-1])
        cv2.imwrite(osp.join(path_gt, city, name_file.split('_leftImg8bit.png')[0] + "_labelIds.png"), gt)
        cv2.imwrite(osp.join(path_original, city, name_file), original_image[..., ::-1])
        