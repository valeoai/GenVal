import torch
import sys
import os
sys.path.append("../relis")

from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from path_dicts import mmseg_models_configs, mmseg_models_checkpoints
from metrics import compute_all_metrics


class Model():
    def __init__(self, model_name, device=0):
        # Instantiate model's config
        config = os.path.join("../relis", mmseg_models_configs[model_name])
        checkpoint = mmseg_models_checkpoints[model_name]

        model = init_segmentor(config, checkpoint, device=f'cuda:{device}')

        # Set model decoder to provide features
        model.decode_head.provide_features = True

        # Set up config of the model to process the dataset
        model.cfg.test_pipeline = [
                                    {'type': 'LoadImageFromFile'},
                                    {'type': 'MultiScaleFlipAug',
                                        'img_scale': (2048, 1024),
                                        'flip': False,
                                        'transforms': [
                                            {'type': 'Resize', 'keep_ratio': True},
                                            {'type': 'RandomFlip'},
                                            {'type': 'Normalize',
                                                'mean': [123.675, 116.28, 103.53], # TODO: Should we adapt it to target dsets?
                                                'std': [58.395, 57.12, 57.375],
                                                'to_rgb': True},
                                            {'type': 'ImageToTensor', 'keys': ['img']},
                                            {'type': 'Collect', 'keys': ['img']}
                                        ]
                                    }
                                    ]
        self.model = model

    
    def get_entropy(self, img):
        """Return entropy for an input image based on the model's output confidences"""
        confidences = inference_segmentor(self.model, img, output_logits=True).squeeze()  # get output confidences
        entropy = -torch.sum(confidences * torch.log(confidences + 1e-15), axis=0)  # Compute entropy

        return entropy
    
    def raw_entropy(self, img, mask):
        """Given an image and a mask, returns the mean entropy in the masked zone"""
        entropy = self.get_entropy(img)
        masked_mean_entropy = entropy[mask.nonzero()].mean().item()  # Compute mean entropy on masked zone

        return masked_mean_entropy

    
    def rel_entropy(self, img, mask):
        """Given an image and a mask, returns the relative mean entropy in the masked zone with respect to the background"""
        entropy = self.get_entropy(img)
        
        masked_mean_entropy = entropy[mask.nonzero()].mean().item()  # Compute mean entropy on masked zone
        background_mean_entropy = entropy[mask == 0].mean().item()  # Compute mean entropy on background

        return masked_mean_entropy / background_mean_entropy

    
    def get_entropy_metrics(self, img, mask):
        """Get metrics using entropy from softmax output"""
        entropy = self.get_entropy(img).cpu().numpy()
        metrics = compute_all_metrics(-entropy.ravel(), mask.ravel())  # Using negative entropy here

        return metrics


    def get_confidence_metrics(self, img, mask):
        """Get metrics using max confidence from softmax output"""
        confidences = inference_segmentor(self.model, img, output_logits=True).squeeze()
        max_confidences = torch.max(confidences, axis=0)[0].cpu().numpy()

        metrics = compute_all_metrics(max_confidences.ravel(), mask.ravel())

        return metrics


    def get_logits_metrics(self, img, mask):
        """Get metrics using max logit from pre-softmax output"""
        logits = inference_segmentor(self.model, img, output_logits=True, pre_softmax=True).squeeze()
        max_logits = torch.max(logits, axis=0)[0].cpu().numpy()

        metrics = compute_all_metrics(max_logits.ravel(), mask.ravel())

        return metrics

    
    def get_all_metrics(self, img, mask):
        raw_entropy = self.raw_entropy(img, mask)
        rel_entropy = self.rel_entropy(img, mask)

        entropy_metrics = self.get_entropy_metrics(img, mask)
        confidence_metrics = self.get_confidence_metrics(img, mask)
        logits_metrics = self.get_logits_metrics(img, mask)

        return {
            "raw_entropy": raw_entropy,
            "rel_entropy": rel_entropy,
            "entropy_metrics": entropy_metrics,
            "confidence_metrics": confidence_metrics,
            "logits_metrics": logits_metrics
        }
