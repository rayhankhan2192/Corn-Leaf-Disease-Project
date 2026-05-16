import torch.nn as nn
from torchvision import models

class VGG19Model(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(VGG19Model, self).__init__()
        self.backbone = models.vgg19(weights='DEFAULT')
        
        in_features = self.backbone.classifier[0].in_features
        
        # FIX: Assign to .classifier, not .fc
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