#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from PIL import Image
import re
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import time
from datetime import datetime
from google.colab import drive
import shutil
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

# Mount Google Drive
drive.mount('/content/drive')

# Set the path to your repository in Google Drive
REPO_PATH = '/content/drive/MyDrive/aiproject_cursor'  # Change this to your repository path
os.chdir(REPO_PATH)  # Change working directory to repository

# Add repository to Python path
sys.path.append(REPO_PATH)

class AlexNetPlus(nn.Module):
    def __init__(self, pretrained=True, freeze_features=True):
        super(AlexNetPlus, self).__init__()
        # Load pretrained AlexNet
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features  # Original feature extractor
        
        # Optionally freeze early layers
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            for param in self.features.parameters():
                param.requires_grad = True
        
        # Add extra conv layer for more complexity
        self.extra_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classifier: extend AlexNet's classifier with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.extra_conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def check_gpu():
    """Check GPU availability and print device information"""
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        return torch.device("cuda")
    else:
        print("GPU is not available. Using CPU instead.")
        return torch.device("cpu")

def verify_data_structure(data_dir):
    """Verify the data directory structure and count images"""
    single_root_dir = os.path.join(data_dir, "1_root_images")
    double_root_dir = os.path.join(data_dir, "2_root_images")
    
    # Check if directories exist
    if not os.path.exists(single_root_dir):
        raise FileNotFoundError(f"Directory not found: {single_root_dir}")
    if not os.path.exists(double_root_dir):
        raise FileNotFoundError(f"Directory not found: {double_root_dir}")
    
    # Count images in each directory
    single_root_images = [f for f in os.listdir(single_root_dir) if f.endswith('.jpeg')]
    double_root_images = [f for f in os.listdir(double_root_dir) if f.endswith('.jpeg')]
    
    print(f"\nData Statistics:")
    print(f"Single root images: {len(single_root_images)}")
    print(f"Double root images: {len(double_root_images)}")
    print(f"Total images: {len(single_root_images) + len(double_root_images)}")
    
    # Verify image format and patient IDs
    print("\nVerifying image formats and patient IDs...")
    patient_ids = set()
    tooth_numbers = set()
    
    for img_list in [single_root_images, double_root_images]:
        for img in img_list:
            match = re.match(r'patient(\d+)_(\d+)', img)
            if not match:
                print(f"Warning: Invalid filename format: {img}")
                continue
            patient_id, tooth_num = match.groups()
            patient_ids.add(patient_id)
            tooth_numbers.add(tooth_num)
    
    print(f"Unique patient IDs: {len(patient_ids)}")
    print(f"Unique tooth numbers: {tooth_numbers}")
    
    return single_root_dir, double_root_dir

class RootImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_transforms():
    """Get data augmentation transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # AlexNet expects 256x256 input
        transforms.CenterCrop(224),     # Then center crop to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def visualize_augmentations(image_path, transform, n_samples=5):
    """Visualize augmented images"""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(15, 3))
    
    # Original image
    plt.subplot(1, n_samples + 1, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Augmented images
    for i in range(n_samples):
        plt.subplot(1, n_samples + 1, i + 2)
        aug_image = transform(image)
        plt.imshow(aug_image.permute(1, 2, 0))
        plt.title(f'Aug {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_folds(image_paths, labels, n_folds=5):
    """Create true 5-fold cross-validation ensuring no patient overlap between train and test"""
    # Group images by patient ID
    patient_groups = defaultdict(list)
    for path, label in zip(image_paths, labels):
        patient_id, _ = extract_patient_info(path)
        if patient_id is not None:
            patient_groups[patient_id].append((path, label))
    
    # Create folds
    patient_ids = list(patient_groups.keys())
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(patient_ids)
    
    # Calculate patients per fold
    patients_per_fold = len(patient_ids) // n_folds
    extra_patients = len(patient_ids) % n_folds
    
    folds = []
    start_idx = 0
    
    # Create each fold
    for fold in range(n_folds):
        # Calculate number of patients for this fold
        fold_size = patients_per_fold + (1 if fold < extra_patients else 0)
        
        # Get test patients for this fold
        test_patients = patient_ids[start_idx:start_idx + fold_size]
        # Get train patients (all patients not in test)
        train_patients = patient_ids[:start_idx] + patient_ids[start_idx + fold_size:]
        
        # Get corresponding images and labels
        train_data = []
        test_data = []
        
        # Add data to appropriate sets based on patient ID
        for patient_id, patient_data in patient_groups.items():
            if patient_id in train_patients:
                train_data.extend(patient_data)
            elif patient_id in test_patients:
                test_data.extend(patient_data)
        
        folds.append((train_data, test_data))
        start_idx += fold_size
    
    # Verify fold creation
    print("\nFold Statistics:")
    total_patients = len(patient_ids)
    print(f"Total unique patients: {total_patients}")
    print(f"Expected patients per fold: {patients_per_fold} (with {extra_patients} folds having +1)")
    
    for i, (train_data, test_data) in enumerate(folds):
        train_patients = set(extract_patient_info(path)[0] for path, _ in train_data)
        test_patients = set(extract_patient_info(path)[0] for path, _ in test_data)
        
        # Count samples per class in each fold
        train_class_0 = len([x for x in train_data if x[1] == 0])
        train_class_1 = len([x for x in train_data if x[1] == 1])
        test_class_0 = len([x for x in test_data if x[1] == 0])
        test_class_1 = len([x for x in test_data if x[1] == 1])
        
        print(f"\nFold {i+1}:")
        print(f"  Training patients: {len(train_patients)} ({len(train_patients)/total_patients*100:.1f}% of total)")
        print(f"  Testing patients: {len(test_patients)} ({len(test_patients)/total_patients*100:.1f}% of total)")
        print(f"  Training samples: {len(train_data)} (Class 0: {train_class_0}, Class 1: {train_class_1})")
        print(f"  Testing samples: {len(test_data)} (Class 0: {test_class_0}, Class 1: {test_class_1})")
        print(f"  Patient overlap: {len(train_patients.intersection(test_patients))}")
        
        # Verify fold sizes
        assert len(train_patients) + len(test_patients) == total_patients, \
            f"Fold {i+1}: Total patients don't match! Train: {len(train_patients)}, Test: {len(test_patients)}, Total: {total_patients}"
    
    return folds

def extract_patient_info(filename):
    """Extract patient ID and tooth number from filename"""
    match = re.match(r'patient(\d+)_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def augment_dataset(dataset, target_size=10000, save_dir='augmented_images', overwrite=False):
    from torchvision.transforms.functional import to_tensor
    def is_too_dark(image, threshold=0.35):
        return to_tensor(image).mean() < threshold
    # Separate data by class
    class_0_data = [(path, label) for path, label in dataset if label == 0]
    class_1_data = [(path, label) for path, label in dataset if label == 1]
    target_per_class = target_size // 2
    expected_total = target_per_class * 2

    if os.path.exists(save_dir) and not overwrite:
        files = [f for f in os.listdir(save_dir) if f.endswith('.jpeg')]
        class_0_files = [f for f in files if f.startswith('aug_0_')]
        class_1_files = [f for f in files if f.startswith('aug_1_')]
        if len(class_0_files) == target_per_class and len(class_1_files) == target_per_class:
            print(f"\nUsing cached augmentations in {save_dir} (class 0: {len(class_0_files)}, class 1: {len(class_1_files)})")
            augmented_data = []
            for f in class_0_files:
                augmented_data.append((os.path.join(save_dir, f), 0))
            for f in class_1_files:
                augmented_data.append((os.path.join(save_dir, f), 1))
            return augmented_data
        else:
            print(f"\nCached augmentations in {save_dir} are incomplete or mismatched. Regenerating...")
            overwrite = True

    if overwrite and os.path.exists(save_dir):
        print(f"\nOverwriting: Deleting {save_dir} and regenerating augmentations...")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Soft augmentation transform for class balancing
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10)
    ])

    augmented_data = []
    for class_data, class_label in [(class_0_data, 0), (class_1_data, 1)]:
        current_size = len(class_data)
        if current_size >= target_per_class:
            for i, (path, label) in enumerate(class_data[:target_per_class]):
                dest_path = os.path.join(save_dir, f'aug_{class_label}_orig_{i}_{os.path.basename(path)}')
                if not os.path.exists(dest_path):
                    shutil.copy2(path, dest_path)
                augmented_data.append((dest_path, class_label))
            continue
        n_augmentations = target_per_class - current_size
        for i, (path, label) in enumerate(class_data):
            dest_path = os.path.join(save_dir, f'aug_{class_label}_orig_{i}_{os.path.basename(path)}')
            if not os.path.exists(dest_path):
                shutil.copy2(path, dest_path)
            augmented_data.append((dest_path, class_label))
        aug_count = 0
        attempts = 0
        while aug_count < n_augmentations and attempts < n_augmentations * 10:
            idx = np.random.randint(0, current_size)
            image_path, _ = class_data[idx]
            image = Image.open(image_path).convert('RGB')
            augmented_image = augment_transform(image)
            if is_too_dark(augmented_image):
                attempts += 1
                continue  # skip too dark images
            aug_path = os.path.join(save_dir, f'aug_{class_label}_{aug_count}_{os.path.basename(image_path)}')
            augmented_image.save(aug_path)
            augmented_data.append((aug_path, class_label))
            aug_count += 1
            attempts += 1

    final_class_0 = len([x for x in augmented_data if x[1] == 0])
    final_class_1 = len([x for x in augmented_data if x[1] == 1])
    print(f"\nFinal dataset class distribution in {save_dir}:")
    print(f"Class 0 (Single root): {final_class_0}")
    print(f"Class 1 (Double root): {final_class_1}")
    print(f"Total samples: {len(augmented_data)}")
    return augmented_data

# Use the same deterministic transform for all splits
shared_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def plot_training_history(train_losses, val_losses, train_accs, val_accs, fold):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Training and Validation Loss (Fold {fold+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'Training and Validation Accuracy (Fold {fold+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, fold):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Fold {fold+1})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def perform_cross_validation(model, folds, device, batch_size=32, num_epochs=200, early_stopping_patience=20, aug_overwrite=False):
    """Perform k-fold cross validation with early stopping."""
    train_transform, val_transform = get_transforms()
    fold_results = []
    
    for fold, (train_data, test_data) in enumerate(folds):
        print(f"\nFold {fold + 1}/{len(folds)}")
        
        # Augment training data (use caching/overwrite logic)
        print("Augmenting training data...")
        train_data = augment_dataset(train_data, target_size=10000, 
                                   save_dir=f'augmented_images_fold_{fold+1}',
                                   overwrite=aug_overwrite)
        
        # Create datasets
        train_dataset = RootImageDataset(
            [x[0] for x in train_data],
            [x[1] for x in train_data],
            transform=train_transform
        )
        test_dataset = RootImageDataset(
            [x[0] for x in test_data],
            [x[1] for x in test_data],
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
        
        # Print dataset statistics
        test_labels = [x[1] for x in test_data]
        class_counts = np.bincount(test_labels)
        print(f"\nTest set class distribution: {class_counts}")
        print(f"Training set size: {len(train_data)} (balanced through augmentation)")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()  # Standard cross entropy without weights
        optimizer = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': 0.00001},  # Lower learning rate for features
            {'params': model.classifier.parameters(), 'lr': 0.0001}  # Higher learning rate for classifier
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Initialize lists for tracking metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Training loop
        best_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            class_correct = [0, 0]
            class_total = [0, 0]
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                # GoogLeNet: use only main output
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Track per-class accuracy
                for i in range(2):
                    mask = labels == i
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            val_class_correct = [0, 0]
            val_class_total = [0, 0]
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # GoogLeNet: use only main output
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Track per-class accuracy
                    for i in range(2):
                        mask = labels == i
                        val_class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                        val_class_total[i] += mask.sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(test_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print statistics
            print(f'\nEpoch {epoch+1}:')
            print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')
            print('\nPer-class Training Accuracy:')
            for i in range(2):
                if class_total[i] > 0:
                    print(f'Class {i}: {100. * class_correct[i] / class_total[i]:.2f}%')
            print('\nPer-class Validation Accuracy:')
            for i in range(2):
                if val_class_total[i] > 0:
                    print(f'Class {i}: {100. * val_class_correct[i] / val_class_total[i]:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                }, f'models/checkpoints/best_model_fold_{fold+1}.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
        
        # Plot training history
        plot_training_history(train_losses, val_losses, train_accs, val_accs, fold)
        
        # Plot confusion matrix for best model
        plot_confusion_matrix(all_labels, all_preds, fold)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        
        fold_results.append(best_acc)
        print(f"Fold {fold+1} Best Accuracy: {best_acc:.2f}%")
    
    return fold_results

def main():
    # Start timing
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs(os.path.join(REPO_PATH, 'models/checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(REPO_PATH, 'augmented_images'), exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Check GPU
    device = check_gpu()

    # Verify data structure
    data_dir = os.path.join(REPO_PATH, "data")
    single_root_dir, double_root_dir = verify_data_structure(data_dir)

    # Get image paths and labels
    single_root_images = [os.path.join(single_root_dir, f) for f in os.listdir(single_root_dir) 
                         if f.endswith('.jpeg')]
    double_root_images = [os.path.join(double_root_dir, f) for f in os.listdir(double_root_dir) 
                         if f.endswith('.jpeg')]

    image_paths = single_root_images + double_root_images
    labels = [0] * len(single_root_images) + [1] * len(double_root_images)

    # Visualize augmentations
    print("\nVisualizing augmentations...")
    # train_transform, val_transform = get_transforms()
    visualize_augmentations(image_paths[0], shared_transform)

    # --- 5-fold cross-validation (DISABLED for now) ---
    # folds = create_folds(image_paths, labels)
    # model_names = ['alexnet', 'alexnetplus', 'googlenet', 'resnet18', 'efficientnet_b0']
    # compare_models(model_names, folds, device, batch_size=32, num_epochs=200, early_stopping_patience=20, aug_overwrite=False)

    # --- 60/20/20 split: train/val/test ---
    trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        trainval_paths, trainval_labels, test_size=0.25, stratify=trainval_labels, random_state=42)
    # Now: train=60%, val=20%, test=20%
    print(f"Train size: {len(train_paths)}, Val size: {len(val_paths)}, Test size: {len(test_paths)}")

    # Augment training data (use caching/overwrite logic)
    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))
    test_data = list(zip(test_paths, test_labels))
    train_data = augment_dataset(train_data, target_size=10000, save_dir='augmented_images_train', overwrite=True)

    # Use the same deterministic transform for all splits
    # (train, val, test)
    train_dataset = RootImageDataset([x[0] for x in train_data], [x[1] for x in train_data], transform=shared_transform)
    val_dataset = RootImageDataset([x[0] for x in val_data], [x[1] for x in val_data], transform=shared_transform)
    test_dataset = RootImageDataset([x[0] for x in test_data], [x[1] for x in test_data], transform=shared_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # --- Model comparison on single split ---
    model_names = ['alexnet', 'alexnetplus', 'googlenet', 'resnet18', 'efficientnet_b0']
    results = []
    for model_name in model_names:
        print(f"\n=== Training {model_name} on single split ===")
        model = get_model(model_name, pretrained=True, freeze_features=True).to(device)
        criterion = nn.CrossEntropyLoss()
        # Robust optimizer setup
        if hasattr(model, 'features') and hasattr(model, 'classifier') and \
           hasattr(model.features, 'parameters') and hasattr(model.classifier, 'parameters'):
            optimizer = torch.optim.Adam([
                {'params': model.features.parameters(), 'lr': 0.00001},
                {'params': model.classifier.parameters(), 'lr': 0.0001}
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_acc = 0
        patience_counter = 0
        early_stopping_patience = 20
        for epoch in range(1):
            # Training
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            class_correct = [0, 0]
            class_total = [0, 0]
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/1"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                # GoogLeNet: use only main output
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                # Per-class accuracy (train)
                for i in range(2):
                    mask = labels == i
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            y_true, y_pred = [], []
            val_class_correct = [0, 0]
            val_class_total = [0, 0]
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # GoogLeNet: use only main output
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    # Per-class accuracy (val)
                    for i in range(2):
                        mask = labels == i
                        val_class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                        val_class_total[i] += mask.sum().item()
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            scheduler.step(val_acc)
            # Print statistics
            print(f'\nEpoch {epoch+1}:')
            print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')
            print('\nPer-class Training Accuracy:')
            for i in range(2):
                if class_total[i] > 0:
                    print(f'Class {i}: {100. * class_correct[i] / class_total[i]:.2f}%')
            print('\nPer-class Validation Accuracy:')
            for i in range(2):
                if val_class_total[i] > 0:
                    print(f'Class {i}: {100. * val_class_correct[i] / val_class_total[i]:.2f}%')

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'models/checkpoints/best_{model_name}.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs for {model_name}")
                break
        # Plot training history
        plot_training_history(train_losses, val_losses, train_accs, val_accs, 0)
        # Evaluate best model on val set
        model.load_state_dict(torch.load(f'models/checkpoints/best_{model_name}.pth', map_location=device))
        model.eval()
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # GoogLeNet: use only main output
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        val_metrics['model'] = model_name
        val_metrics['split'] = 'val'
        # Evaluate best model on test set
        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # GoogLeNet: use only main output
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())
        test_metrics = compute_metrics(y_true_test, y_pred_test)
        test_metrics['model'] = model_name
        test_metrics['split'] = 'test'
        results.append(val_metrics)
        results.append(test_metrics)
        # Save confusion matrices
        import matplotlib
        import matplotlib.pyplot
        import seaborn
        for split, y_true, y_pred in [('val', y_true_val, y_pred_val), ('test', y_true_test, y_pred_test)]:
            cm = confusion_matrix(y_true, y_pred)
            matplotlib.pyplot.figure(figsize=(6,5))
            seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            matplotlib.pyplot.title(f'Confusion Matrix ({model_name} {split})')
            matplotlib.pyplot.ylabel('True Label')
            matplotlib.pyplot.xlabel('Predicted Label')
            matplotlib.pyplot.savefig(f'results/plots/{model_name}_confmat_{split}.png')
            matplotlib.pyplot.close()
        # PCA analysis on test set
        pca_analysis(model, test_loader, device, model_name)
        # Show PCA plot inline
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(f'results/plots/{model_name}_pca_test.png')
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'PCA of Test Features ({model_name})')
        plt.show()
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results/model_comparison_single_split.csv', index=False)
    print("\nAll model results saved to results/model_comparison_single_split.csv and plots in results/plots/")
    # Print total training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time/3600:.2f} hours")

# --- Model Wrappers ---
def get_model(model_name, pretrained=True, freeze_features=False):
    if model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        # Modify classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
        # Optionally freeze features
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
        return model
    elif model_name == 'alexnetplus':
        return AlexNetPlus(pretrained=pretrained, freeze_features=freeze_features)
    elif model_name == 'googlenet':
        # aux_logits must be True if using pretrained weights
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

# --- Metrics Helper ---
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    # Per-class
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision_score(y_true, y_pred, pos_label=1)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, pos_label=1)
    # For class 0
    sensitivity_0 = recall_score(y_true, y_pred, pos_label=0)
    specificity_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv_0 = precision_score(y_true, y_pred, pos_label=0)
    npv_0 = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return {
        'accuracy': accuracy,
        'f1_1': f1,
        'f1_0': f1_0,
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'ppv_1': ppv,
        'npv_1': npv,
        'specificity_1': specificity,
        'sensitivity_1': sensitivity,
        'ppv_0': ppv_0,
        'npv_0': npv_0,
        'specificity_0': specificity_0,
        'sensitivity_0': sensitivity_0
    }

def extract_features(model, dataloader, device, model_name):
    model.eval()
    features = []
    labels = []
    # Hook for penultimate layer
    feat_out = []
    def hook_fn(module, input, output):
        feat_out.append(output.detach().cpu())
    # Register hook depending on model type
    handle = None
    if model_name in ['alexnet', 'alexnetplus']:
        # Last FC before output
        handle = model.classifier[-2].register_forward_hook(hook_fn)
    elif model_name == 'resnet18':
        handle = model.avgpool.register_forward_hook(lambda m, i, o: feat_out.append(torch.flatten(o, 1).detach().cpu()))
    elif model_name == 'efficientnet_b0':
        handle = model.classifier[-2].register_forward_hook(hook_fn)
    elif model_name == 'googlenet':
        handle = model.avgpool.register_forward_hook(lambda m, i, o: feat_out.append(torch.flatten(o, 1).detach().cpu()))
    else:
        raise ValueError(f'Unknown model for feature extraction: {model_name}')
    # Forward pass
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            _ = model(images)
            labels.extend(lbls.cpu().numpy())
    # Concatenate features
    features = torch.cat(feat_out, dim=0).numpy()
    if handle is not None:
        handle.remove()
    return features, labels

def pca_analysis(model, dataloader, device, model_name):
    import os
    os.makedirs('results/embeddings', exist_ok=True)
    features, labels = extract_features(model, dataloader, device, model_name)
    # Save as .npz
    import numpy as np
    np.savez(f'results/embeddings/{model_name}_test_embeddings.npz', features=features, labels=labels)
    # Save as .csv
    import pandas as pd
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(f'results/embeddings/{model_name}_test_embeddings.csv', index=False)
    print(f'Embeddings saved: results/embeddings/{model_name}_test_embeddings.npz and .csv')
    # PCA and plot
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    import matplotlib.pyplot as plt
    for label in np.unique(labels):
        idx = np.array(labels) == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Class {label}', alpha=0.6)
    plt.title(f'PCA of Test Features ({model_name})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/{model_name}_pca_test.png')
    plt.close()
    print(f'PCA plot saved: results/plots/{model_name}_pca_test.png')

# --- Main Model Comparison Loop ---
def compare_models(model_names, folds, device, batch_size=32, num_epochs=200, early_stopping_patience=20, aug_overwrite=False):
    os.makedirs('results/plots', exist_ok=True)
    all_results = []
    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        model = get_model(model_name, pretrained=True, freeze_features=False).to(device)
        fold_metrics = []
        results = perform_cross_validation(model, folds, device, batch_size, num_epochs, early_stopping_patience, aug_overwrite)
        # For each fold, after training, load best model and compute detailed metrics
        for fold, (train_data, test_data) in enumerate(folds):
            # Load best model
            checkpoint_path = f'models/checkpoints/best_model_fold_{fold+1}.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            # Evaluate on test set
            _, val_transform = get_transforms()
            test_dataset = RootImageDataset([x[0] for x in test_data], [x[1] for x in test_data], transform=val_transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            metrics = compute_metrics(y_true, y_pred)
            metrics['fold'] = fold+1
            metrics['model'] = model_name
            fold_metrics.append(metrics)
            # Save confusion matrix plot
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix ({model_name} Fold {fold+1})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'results/plots/{model_name}_confmat_fold{fold+1}.png')
            plt.close()
        # Save per-model results to CSV
        df = pd.DataFrame(fold_metrics)
        df.to_csv(f'results/{model_name}_fold_metrics.csv', index=False)
        all_results.append(df)
        # Print summary
        print(f"\n{model_name} summary:")
        print(df.describe())
    # Combine all results
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv('results/model_comparison.csv', index=False)
    # Plot bar chart for accuracy and f1_macro
    plt.figure(figsize=(10,6))
    sns.barplot(data=all_df, x='model', y='accuracy', ci='sd')
    plt.title('Model Comparison: Accuracy')
    plt.savefig('results/plots/model_accuracy_comparison.png')
    plt.close()
    plt.figure(figsize=(10,6))
    sns.barplot(data=all_df, x='model', y='f1_macro', ci='sd')
    plt.title('Model Comparison: F1 Macro')
    plt.savefig('results/plots/model_f1_comparison.png')
    plt.close()
    print("\nAll model results saved to results/model_comparison.csv and plots in results/plots/")
    # Generate PDF summary
    generate_summary_pdf()

def generate_summary_pdf():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import glob
    # Read results
    df = pd.read_csv('results/model_comparison.csv')
    pdf_path = 'results/model_comparison_summary.pdf'
    with PdfPages(pdf_path) as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.8, 'Model Comparison Summary', fontsize=24, ha='center')
        plt.text(0.5, 0.7, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=14, ha='center')
        pdf.savefig(); plt.close()
        # Table of metrics (mean/std per model)
        summary = df.groupby('model').agg(['mean', 'std'])
        plt.figure(figsize=(12, 4))
        plt.axis('off')
        table = plt.table(cellText=summary.round(3).values,
                          colLabels=summary.columns,
                          rowLabels=summary.index,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.title('Metrics Summary (mean ± std)')
        pdf.savefig(); plt.close()
        # Add bar charts
        for metric in ['accuracy', 'f1_macro']:
            img_path = f'results/plots/model_{metric}_comparison.png'
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                plt.figure(figsize=(10,6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Model Comparison: {metric.capitalize()}')
                pdf.savefig(); plt.close()
        # Add confusion matrices for each model/fold
        confmat_imgs = sorted(glob.glob('results/plots/*confmat_fold*.png'))
        for img_path in confmat_imgs:
            img = mpimg.imread(img_path)
            plt.figure(figsize=(8,6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path).replace('_', ' '))
            pdf.savefig(); plt.close()
        # (Optional) Add more plots if available
    print(f"Summary PDF saved to {pdf_path}")

if __name__ == "__main__":
    main() 