import argparse
import torch
import torch.nn as nn
import logging
import os
import numpy as np

# Important: Imports matching the exact folder structure
from utils.dataloader import create_data_loaders, save_augmentation_samples
from utils.train_evaluation import Trainer
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Augmentation Strategy: {args.aug_type.upper()}")
    logger.info(f"Background Removal: {args.remove_bg}")

    # 1. Load Data
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_names=None, 
        image_size=(224, 224),
        aug_type=args.aug_type,
        #remove_bg=args.remove_bg  # Uncomment when you are ready to use rembg again
    )

    # 2. Dynamic Class Loading & Weighting
    CLASSES = train_loader.dataset.class_names
    num_classes = len(CLASSES)
    logger.info(f"Dynamically loaded {num_classes} classes: {CLASSES}")

    train_labels = train_loader.dataset.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights: {class_weights.numpy()}")

    # 3. Initialize Model & Loss
    model = get_model(model_name=args.model_name, num_classes=num_classes)
    model = model.to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.loss == "labelsmoothing":
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 4. Setup Trainer
    trainer = Trainer(
        model=model, 
        device=device, 
        class_names=CLASSES, 
        model_name=args.model_name,
        class_weights=class_weights,
        aug_type=args.aug_type,
        patience=args.patience,
        save_dir=args.save_dir
    )

    # 5. Save Preprocessing Samples
    if args.save_samples > 0:
        save_augmentation_samples(
            train_loader=train_loader,
            save_dir=trainer.base_dir,
            num_samples=args.save_samples
        )

    # 6. Execute Training
    logger.info(f"Starting training: {args.model_name} | Aug: {args.aug_type} | Loss: {args.loss}...")
    trainer.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs,
        criterion=criterion,
        lr=args.lr
    )

    # 7. Final Evaluation
    test_metrics, test_loss = trainer.evaluate(test_loader) 
    logger.info(f"Final Results - Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Final Results - Loss: {test_loss:.4f}")
    
    # 8. Generate Explainable AI (XAI) Visualizations
    trainer.generate_gradcam(test_loader, num_images=20)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Corn Leaf Disease Standalone Training")
    
    # Data & Architecture
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model-name", type=str, default="customcnn", choices=["customcnn", "vitb16", "efficientnetb3", "mobilenetv3", "densenet121", "vgg19", "resnet50", "hybridmodel"], help="Model from factory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--aug-type", type=str, default="standard", choices=["none", "standard", "enhanced"])
    
    # Preprocessing
    parser.add_argument("--save-samples", type=int, default=20, help="Number of images to save for preprocessing verification")
    parser.add_argument("--remove-bg", action='store_true', help="Use rembg to isolate leaf")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy", "focal", "labelsmoothing"])
    
    # Early Stopping & Checkpointing
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()