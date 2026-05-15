import torch
import torch.nn as nn
from torchvision import models

class TeaLeafHybridModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super(TeaLeafHybridModel, self).__init__()
        
        # 1. CNN Branch (EfficientNet) for local texture (spots/rust)
        # EfficientNet is efficient and captures fine-grained patterns
        self.cnn_branch = models.efficientnet_v2_s(weights='DEFAULT').features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. Transformer Branch (MobileNetV3 as a fast feature proxy)
        # We use MobileNetV3's deep features to represent global leaf structure
        self.mobile_branch = models.mobilenet_v3_small(weights='DEFAULT').features
        self.mobile_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Combined features: EfficientNetV2-S (1280) + MobileNetV3-S (576)
        total_features = 1280 + 576

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Local features
        cnn_feat = self.cnn_branch(x)
        cnn_feat = self.cnn_pool(cnn_feat).flatten(1)
        
        # Global/Structural features
        mob_feat = self.mobile_branch(x)
        mob_feat = self.mobile_pool(mob_feat).flatten(1)
        
        # Fusion
        combined = torch.cat((cnn_feat, mob_feat), dim=1)
        return self.classifier(combined)