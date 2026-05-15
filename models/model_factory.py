import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json
import os
import logging

from models.cnn_model import CustomCNN
from models.ViTB16 import ViTB16
from models.EfficientNetV2B3 import EfficientNetV2B3
from models.MobileNetV3 import MobileNetV3
from models.denseNet121 import DenseNet121Model
from models.VGG19 import VGG19Model
from models.ResNet50 import ResNet50Model
from models.Hybrid_CNN_Transformer import TeaLeafHybridModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
    if model_name == 'customcnn':
        return CustomCNN(num_classes=num_classes)
    elif model_name == 'vitb16':
        return ViTB16(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'efficientnetb3':
        return EfficientNetV2B3(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'mobilenetv3':
        return MobileNetV3(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == 'densenet121':
        return DenseNet121Model(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'vgg19':
        return VGG19Model(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'resnet50':
        return ResNet50Model(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'hybridmodel':
        return TeaLeafHybridModel(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for image classification
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss
