import glob
import json
import argparse
import os

from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm


def main():
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cityscapes', type=str, help='dataset to process in cityscapes, idd, acdc_fog, acdc_night, acdc_rain, acdc_snow')
    parser.add_argument('--device', default=0, type=int, help='device to run on')
    args = parser.parse_args()

    # Get arguments
    dataset = args.dataset
    device = int(args.device)

    # Get files to read from
    config = json.load(open('config.json'))
    dpath = config[dataset]
    files = {dataset: glob.glob(dpath, recursive=True)}

    # Initializing captions
    captions = {}
    os.makedirs('captions', exist_ok=True)

    # Initializing interrogator
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", cache_path='cache', device=device, quiet=True))

    # Looping over datasets and files to get both captions and embeddings
    for dset in files:
        captions[dset] = {}

        for file in tqdm(files[dset]):
            # Read image
            image = Image.open(file).convert('RGB')

            # Get captions and embeddings
            caption = ci.interrogate(image)
            captions[dset][file] = caption

    # Save files
    json.dump(captions, open(f'captions/captions_{dset}.json', 'w'), indent=4)


if __name__ == '__main__':
    main()