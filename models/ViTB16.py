import torch
import torch.nn as nn
from torchvision import models

class ViTB16(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.2):
        super(ViTB16, self).__init__()
        # Load pre-trained ViT-B/16
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # ViT has a 'heads' attribute for classification
        in_features = self.backbone.heads.head.in_features
        
        # Replace the head with an MLP specifically for tea disease patterns
        self.backbone.heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(), 
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)