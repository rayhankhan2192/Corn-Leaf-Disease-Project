import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


def create_data_loaders(
    data_dir: str, 
    batch_size: int = 32, 
    aug_type: str = 'standard', 
    class_names: Optional[List[str]] = None, 
    image_size: Tuple[int, int] = (224, 224),
    remove_bg: bool = True 
):
    full_ds = CornLeafDataset(
        data_dir, subset='full', class_names=class_names, 
        image_size=image_size, remove_bg=remove_bg
    )
    
    train_idx, temp_idx = train_test_split(
        range(len(full_ds)), train_size=0.8, stratify=full_ds.targets, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, stratify=[full_ds.targets[i] for i in temp_idx], random_state=42)

    def build_subset(indices, subset_name):
        mode = aug_type if subset_name == 'train' else 'none'
        ds = CornLeafDataset(
            data_dir, transform=get_leaf_transforms(image_size, mode), 
            subset=subset_name, class_names=full_ds.class_names, remove_bg=remove_bg
        )
        ds.samples = [full_ds.samples[i] for i in indices]
        ds.targets = [s[1] for s in ds.samples]
        ds.log_summary()
        return DataLoader(ds, batch_size=batch_size, shuffle=(subset_name == 'train'))

    train_loader = build_subset(train_idx, 'train')
    val_loader = build_subset(val_idx, 'val')
    test_loader = build_subset(test_idx, 'test')

    return train_loader, val_loader, test_loader, full_ds.class_weights