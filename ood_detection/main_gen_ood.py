import argparse
import json
import os
from tqdm import tqdm

from dataset import GeneratedOODDataset
from model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ANN-R50", help="Which model to evaluate")
    parser.add_argument("--device", type=int, default=0, help="On which GPU to run on")
    args = parser.parse_args()

    # Get args
    model_name = args.model_name
    device = args.device

    # Instantiate model and dataset
    root_path = "../images/ood_objects"
    model = Model(model_name=model_name, device=device)
    gen_ood_dataset = GeneratedOODDataset(root_path)

    # Going through all examples in the dataset
    save_dict = {}
    save_path = os.path.join(f"gen/{model_name}_metrics.json")

    for img_path, sample in tqdm(gen_ood_dataset):
        img, mask = sample["img"], sample["mask"]

        save_dict[img_path] = model.get_all_metrics(img, mask)

        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=4)
