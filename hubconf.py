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
    
    
    checkpoint = torch.hub.load_state_dict_from_url(
            "https://release-assets.githubusercontent.com/github-production-release-asset/1069619774/bfbd03f2-10c9-4429-aab3-f80143dc2177?sp=r&sv=2018-11-09&sr=b&spr=https&se=2025-10-04T18%3A11%3A02Z&rscd=attachment%3B+filename%3Dboq_dinov2l14_ocd.ckpt&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2025-10-04T17%3A10%3A42Z&ske=2025-10-04T18%3A11%3A02Z&sks=b&skv=2018-11-09&sig=DGO7sO1AXTktpSnWgBjQ8VDgakVmBJBKydUR6Dzu%2B40%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc1OTYwMjk0OCwibmJmIjoxNzU5NTk5MzQ4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.1TcIsL4TH3QINUqpN-bHyx6UgzzarjKnjZmN2jcKJ00&response-content-disposition=attachment%3B%20filename%3Dboq_dinov2l14_ocd.ckpt&response-content-type=application%2Foctet-stream",
    map_location=torch.device('cpu')
)

    # Handle PyTorch Lightning checkpoint structure
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Optionally remove 'model.' prefix if needed
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    vpr_model.load_state_dict(state_dict, strict=False)

    return vpr_model
