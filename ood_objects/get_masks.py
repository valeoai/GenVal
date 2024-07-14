import os
import cv2
import random
import torch
import argparse
import json
import torchvision
import numpy as np

from PIL import Image
from tqdm import tqdm
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


# Params
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../checkpoints/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "../checkpoints/sam_vit_h_4b8939.pth"

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Utils for mask detection
def get_grounding_dino():
    return Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


def get_sam(device):
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    return SamPredictor(sam)


def detect_object(image, grounding_dino_model, obj):
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=[obj],
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    return detections


def nms_post_process(detections):
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    return detections


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Segment with SAM from boxes extracted from GroundingDINO"""
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def convert_detection_to_mask(detections, sam_predictor, image):
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    return detections[0]


def get_mask_from_image(image, obj, grounding_dino, sam_predictor):
    gr_detections = detect_object(image, grounding_dino, obj)
    pp_detections = nms_post_process(gr_detections)
    final_detections = convert_detection_to_mask(pp_detections, sam_predictor, image)
    masks = final_detections.mask.astype(float).transpose(1, 2, 0)
    return masks


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="On which GPU to run on")
    args = parser.parse_args()
    device = args.device

    # Get models for GroundingSAM
    grounding_dino = get_grounding_dino()
    sam_model = get_sam(device=device)

    # List objects to get masks from
    root_path = "../images/ood_objects/refined/"
    objects = os.listdir(root_path)
    random.shuffle(objects)
    
    # For each object, we get all images and extract the mask from it
    for obj in tqdm(objects):
        images_paths = os.listdir(os.path.join(root_path, obj))
        save_path_object = os.path.join(root_path.replace("refined", "masks"), obj)
        random.shuffle(images_paths)  # For other jobs running in parallel

        if not os.path.exists(save_path_object):
            os.makedirs(save_path_object)

        for image_path in tqdm(images_paths):
            # Get save path
            save_path = os.path.join(save_path_object, f"{image_path.split('.png')[0]}_mask.png")

            if os.path.exists(save_path) or os.path.exists(save_path.replace("png", "txt")):
                print(f"{image_path} has already been processed, skipping...")
                continue
            
            # Get image
            image = np.array(Image.open(os.path.join(root_path, obj, image_path)).convert('RGB'))

            # Get mask position and size
            pos_path = os.path.join(root_path.replace("refined", "with_positions"), obj, image_path.split(".png")[0] + "_pos.json")
            pos = json.load(open(pos_path, 'r'))[image_path]
            x, y = pos["pos_mask"]
            size_mask = pos["size_mask"]

            # Get crop
            cropped_image = image[x: x+size_mask, y: y+size_mask]

            # Inference
            try:
                cropped_mask = get_mask_from_image(cropped_image, obj, grounding_dino, sam_model)
                final_mask = np.zeros_like(image)[..., 0][..., None]
                final_mask[x: x+size_mask, y: y+size_mask] = cropped_mask
                cv2.imwrite(save_path, 255*final_mask)
            except Exception as e:
                print(e)
                print(f"No mask found for {image_path} for {obj}...")

                # Create empty txt file for other jobs running in parallel, and to know for which image no mask was found
                f = open(save_path.replace("png", "txt"), "w")
                