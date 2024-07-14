import json
import cv2
import numpy as np
import random

from torch.utils.data import Dataset


class Cityscapes(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.data = []

        with open(f'./training/{self.split}.jsonl', 'r') as f:
            self.prompts = [json.loads(line)for line in f]
        
        for el in self.prompts:
            self.data.append({"source": el["conditioning_image"].replace("leftImg8bit", "gtFine_labelIds"), "target": el["image"], "prompt": el["text"]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(f'{source_filename}', cv2.IMREAD_UNCHANGED).astype(np.float32)
        target = cv2.imread(f'{target_filename}').astype(np.float32)

        # Same random cropping to 512x512 for source and target images
        source, target = self.crop(source, target)

        source = (source + 1) / 34  # CityScapes ids go theoretically from -1 to 33 (no -1 but here in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
        source = source[..., np.newaxis]

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

    def crop(self, source, target):
        # Get min dimension between height and width to get square
        h, w = source.shape[:2]
        min_dim = min(h, w)

        # Crop image on min dimension
        if h == min_dim:
            y = random.randint(0, w-min_dim)  # Get random width index on which to start crop
            source = source[:, y: y+min_dim]
            target = target[:, y: y+min_dim]
        else:
            x = random.randint(0, h-min_dim)  # Get random height index on which to start crop
            source = source[x: x+min_dim, :]
            target = target[x: x+min_dim, :]

        # Then downsample to target resolution, keeping nearest to get integers for discrete segmentation
        source = cv2.resize(source, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        target = cv2.resize(target, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

        return source, target
