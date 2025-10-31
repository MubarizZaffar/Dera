"""
Dera is a VPR method [1] trained on both Google Street View (accurate GPS) and an internal geometrically verified Mapillary dataset (highy diverse and GPS-verified) 

[1] https://github.com/amaralibey/Bag-of-Queries

This python file is written for compatibility with hloc, Lamar, CrocoDL, etc. Feel free to use under CC BY license.

Written in October 2025.
"""

import torch
import torchvision.transforms as T

from ..utils.base_model import BaseModel


class Dera(BaseModel):
    required_inputs = ["image"]

    def _init(self, conf):
        self.vpr_model = torch.hub.load("MubarizZaffar/Dera", "get_trained_dera").eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        im_size = (322, 322) # to be used with DinoV2 backbone

        self.input_transform = T.Compose([
        T.Resize(im_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    ])
    
    def _forward(self, data):
        image = self.input_transform(data["image"])
        desc, _ = self.vpr_model(image)
        # desc = desc.squeeze()
        return {
            "global_descriptor": desc,
        }
        
