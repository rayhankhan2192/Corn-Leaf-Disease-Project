import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super(ResNet50Model, self).__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        
        # MLP head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),          # Stabilizes deep feature transition
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),          # Prevents gradient collapse
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)