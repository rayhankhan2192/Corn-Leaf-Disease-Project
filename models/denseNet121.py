import torch.nn as nn
from torchvision import models

class DenseNet121Model(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(DenseNet121Model, self).__init__()
        self.backbone = models.densenet121(weights='DEFAULT')
        
        in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)