import glob
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class GeneratedOODDataset(Dataset):
    def __init__(self, root_path):
        """We create all the paths for the edited images and the corresponding masks"""
        # First get all images in the dataset
        imgs_paths = glob.glob("../images/ood_objects/refined/**/*.png")

        # Then get corresponding masks and positions
        masks_root_path = os.path.join(root_path, "masks")

        imgs_metadata = {
            img_path: os.path.join(masks_root_path, "/".join(img_path.split("/")[-2:]).replace(".png", "_mask.png")) 
            for img_path in imgs_paths
        }

        self.imgs_metadata = {}

        # For uncurated dataset, masks not found have an empty .txt file associated.
        # We filter the metadata by checking if the .png mask file exists and only keep images that have corresponding masks
        for img_path in imgs_metadata:
            if os.path.exists(imgs_metadata[img_path]):
                self.imgs_metadata[img_path] = imgs_metadata[img_path]


    def __len__(self):
        return len(self.imgs_metadata)


    def __getitem__(self, idx):
        """
        Returns:
            - the image at index idx 
            - the corresponding binary mask extracted by Grounded-SAM
        """
        # Extracting paths
        img_path = list(self.imgs_metadata)[idx]
        mask_path = self.imgs_metadata[img_path]

        # Read data
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Adapting mask to only get OOD object
        mask[mask == 255] = 1

        return img_path, {"img": img, "mask": mask}
