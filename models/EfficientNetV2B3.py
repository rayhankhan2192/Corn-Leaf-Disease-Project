import torch.nn as nn
from torchvision import models

class EfficientNetV2B3(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3):
        super(EfficientNetV2B3, self).__init__()
        try:
            self.backbone = models.efficientnet_v2_b3(weights=models.EfficientNet_V2_B3_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        
        if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
            in_features = self.backbone.classifier[1].in_features
        else:
            in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)