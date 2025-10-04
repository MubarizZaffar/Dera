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
        num_queries=128,
        num_layers=2,
        row_dim=output_dim//512, # 32 for dinov2
    )
        
    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    
    checkpoint = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/MubarizZaffar/Dera/resolve/main/boq_dinov2l14_ocd.ckpt",
    map_location=torch.device('cpu')
)

    # Handle PyTorch Lightning checkpoint structure
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Optionally remove 'model.' prefix if needed
    # state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    vpr_model.load_state_dict(state_dict, strict=False)

    return vpr_model
