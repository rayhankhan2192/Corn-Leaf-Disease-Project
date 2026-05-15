import os
import cv2
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import Grad-CAM libraries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, device, class_names, model_name, class_weights, aug_type, patience=10, save_dir="checkpoints"):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model_name = model_name.lower()
        self.aug_type = aug_type
        self.patience = patience
        
        # Checkpoint directory
        self.base_dir = os.path.join(save_dir, f"{model_name}_{aug_type}")
        os.makedirs(self.base_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.base_dir, "best_model.pth")
        
        # Results directory (for Plots & Grad-CAM)
        self.result_dir = os.path.join("Result", f"{self.model_name}_{self.aug_type}")
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # Dictionary to track metrics for plotting
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self, train_loader, val_loader, epochs, criterion, lr):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        for epoch in range(1, epochs + 1):
            logger.info(f"--- Epoch {epoch}/{epochs} ---")
            
            self.model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            train_bar = tqdm(train_loader, desc="Training")
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(train_targets, train_preds)

            val_metrics, val_loss = self.evaluate(val_loader, prefix="Validation")
            
            logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

            # Save metrics to history for plotting
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])

            scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                logger.info(f"Validation loss improved. Saving checkpoint to {self.checkpoint_path}")
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.early_stop_counter += 1
                logger.info(f"No improvement. Early stopping counter: {self.early_stop_counter}/{self.patience}")
                
            if self.early_stop_counter >= self.patience:
                logger.info("Early stopping triggered!")
                break
                
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            
        # Plot training curves at the end of training
        self.plot_metrics()

    def evaluate(self, dataloader, prefix="Test"):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        eval_criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = eval_criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        acc = accuracy_score(all_targets, all_preds)
        
        if prefix == "Test":
            logger.info(f"\n--- Final {prefix} Classification Report ---")
            report = classification_report(all_targets, all_preds, target_names=self.class_names, zero_division=0)
            print(report)
            # Generate Confusion Matrix only for the final Test set
            self.plot_confusion_matrix(all_targets, all_preds)

        metrics = {'accuracy': acc}
        return metrics, avg_loss

    def plot_metrics(self):
        """Plots and saves the training/validation loss and accuracy curves."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', marker='o')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(epochs, self.history['val_acc'], label='Val Accuracy', marker='o')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, "training_metrics_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Training metrics curve saved to {save_path}")

    def plot_confusion_matrix(self, targets, preds):
        """Plots and saves a visually appealing confusion matrix."""
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix ({self.model_name.upper()})')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def generate_gradcam(self, test_loader, num_images=20):
        """Generates and saves Grad-CAM heatmaps for a subset of the test data."""
        logger.info("Generating Grad-CAM Visualizations...")
        
        target_layers = []
        if self.model_name == "customcnn":
            target_layers = [self.model.features[-1]]
        elif self.model_name == "resnet50":
            target_layers = [self.model.layer4[-1]]
        elif self.model_name == "efficientnetb3":
            target_layers = [self.model.features[-1]]
        elif self.model_name == "mobilenetv3":
            target_layers = [self.model.features[-1]]
        elif self.model_name == "vgg19":
            target_layers = [self.model.features[-1]]
        elif self.model_name == "densenet121":
            target_layers = [self.model.features.norm5]
        elif self.model_name == "vitb16":
            logger.warning("Standard Grad-CAM doesn't work well with ViT out-of-the-box. Skipping Grad-CAM.")
            return
        else:
            logger.warning(f"Target layer mapping not defined for {self.model_name}. Skipping Grad-CAM.")
            return

        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        self.model.eval()
        count = 0
        
        for images, labels in test_loader:
            images = images.to(self.device)
            grayscale_cams = cam(input_tensor=images, targets=None)

            for i in range(images.size(0)):
                if count >= num_images:
                    logger.info(f"Grad-CAM generation complete. Images saved to {self.result_dir}/")
                    return

                img_tensor = images[i].cpu().permute(1, 2, 0).numpy()
                img_unnorm = std * img_tensor + mean
                img_unnorm = np.clip(img_unnorm, 0, 1)

                grayscale_cam = grayscale_cams[i, :]
                visualization = show_cam_on_image(img_unnorm, grayscale_cam, use_rgb=True)

                true_label = self.class_names[labels[i].item()]
                with torch.no_grad():
                    pred = self.model(images[i].unsqueeze(0))
                    pred_idx = torch.argmax(pred).item()
                    pred_label = self.class_names[pred_idx]

                status = "PASS" if true_label == pred_label else "FAIL"
                filename = f"gradcam_{count}_{status}_True-{true_label}_Pred-{pred_label}.jpg"
                filepath = os.path.join(self.result_dir, filename)

                visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, visualization)
                
                count += 1