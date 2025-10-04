dependencies = ['torch', 'torchvision']

import sys
import os

# Add BoQ's src directory directly to path
boq_root = os.path.dirname(__file__)  # Root of the cloned repo
sys.path.append(os.path.join(boq_root, "src"))  

import torch
from backbones import DinoV2
from boq import BoQ

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns

def get_trained_dera(backbone_name="dinov2", output_dim=12288):
    output_dim = 12288
    
    # load the backbone
    backbone = DinoV2()
    # load the aggregator
    aggregator = BoQ(
        in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
        proj_channels=512,
        num_queries=64,
        num_layers=2,
        row_dim=output_dim//512, # 32 for dinov2
    )
        
    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://drive.google.com/file/d/1AKVBzd7gEHvKGhV4IO0IlPrjl6D7dgcN/view?usp=sharing",
            map_location=torch.device('cpu')
        )
    )
    return vpr_model
