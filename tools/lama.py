import os

from lama.saicinpainting.evaluation.utils import move_to_device
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from lama.saicinpainting.training.trainers import load_checkpoint

from lama.saicinpainting.training.data.masks import MixedMaskGenerator
from lama.saicinpainting.evaluation.data import pad_img_to_modulo

class LamaProcessor:
    def __init__(self, cfg):
        self.device = torch.device('cpu')
        model_path = cfg['LAMA_MODEL_PATH']
        train_config_path = os.path.join(model_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)
        args = {}
        self.mask_generator = MixedMaskGenerator(**args)
        self.pad_out_to_modulo = 8
    
    def process_image(self, image, to_gray = False):
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        out_img = image.astype('float32') / 255
        return out_img

    
    def process(self, original_image, mask):
        print('Processing LAMA...')
        original_image = self.process_image(original_image)
        mask = self.process_image(mask, to_gray=True)
        result = dict(image=original_image, mask=mask[None, ...])
        result['unpad_to_size'] = result['image'].shape[1:]
        result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
        result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        batch = default_collate([result])
        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)                    
            result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                result = result[:orig_height, :orig_width]

        result = np.clip(result * 255, 0, 255).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print('LAMA processing done')
        return result