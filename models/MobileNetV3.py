import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        self.dropout_rate = dropout_rate
        super(MobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained).features
        num_features = 960 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.Hardswish(inplace=True), 
            nn.BatchNorm1d(512),      
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)  
        x = self.classifier(x)
        return x