#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image
import re
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        transforms.Resize((224, 224)),
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def extract_patient_info(filename):
    """Extract patient ID and tooth number from filename"""
    match = re.match(r'patient(\d+)_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def create_folds(image_paths, labels, n_folds=5):
    """Create folds ensuring no patient overlap between train and test"""
    # Group images by patient ID
    patient_groups = defaultdict(list)
    for path, label in zip(image_paths, labels):
        patient_id, _ = extract_patient_info(path)
        if patient_id is not None:
            patient_groups[patient_id].append((path, label))
    
    # Create folds
    patient_ids = list(patient_groups.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    folds = []
    for train_idx, test_idx in kf.split(patient_ids):
        # Get train and test patient IDs
        train_patients = [patient_ids[i] for i in train_idx]
        test_patients = [patient_ids[i] for i in test_idx]
        
        # Get corresponding images and labels
        train_data = [(path, label) for pid in train_patients 
                     for path, label in patient_groups[pid]]
        test_data = [(path, label) for pid in test_patients 
                    for path, label in patient_groups[pid]]
        
        folds.append((train_data, test_data))
    
    return folds

def augment_dataset(dataset, target_size=10000):
    """Augment dataset to reach target size"""
    current_size = len(dataset)
    if current_size >= target_size:
        return dataset
    
    # Calculate number of augmentations needed
    n_augmentations = target_size - current_size
    augmented_data = []
    
    # Create augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])
    
    # Perform augmentations
    for _ in tqdm(range(n_augmentations), desc="Augmenting dataset"):
        # Randomly select an image
        idx = np.random.randint(0, current_size)
        image_path, label = dataset[idx]
        
        # Load and augment image
        image = Image.open(image_path).convert('RGB')
        augmented_image = augment_transform(image)
        
        # Save augmented image
        aug_path = image_path.replace('.jpeg', f'_aug_{_}.jpeg')
        augmented_image.save(aug_path)
        augmented_data.append((aug_path, label))
    
    return dataset + augmented_data

def perform_cross_validation(model, folds, device, batch_size=32, num_epochs=50):
    """Perform k-fold cross validation"""
    train_transform, val_transform = get_transforms()
    fold_results = []
    
    for fold, (train_data, test_data) in enumerate(folds):
        print(f"\nFold {fold + 1}/{len(folds)}")
        
        # Augment training data
        print("Augmenting training data...")
        train_data = augment_dataset(train_data, target_size=10000)
        
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_acc = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Print statistics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.3f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(test_loader):.3f}, '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'models/checkpoints/best_model_fold_{fold+1}.pth')
        
        fold_results.append(best_acc)
        print(f"Fold {fold+1} Best Accuracy: {best_acc:.2f}%")
    
    return fold_results

def main():
    # Set up paths
    data_dir = "data"
    single_root_dir = os.path.join(data_dir, "1_root_images")
    double_root_dir = os.path.join(data_dir, "2_root_images")
    
    # Get image paths and labels
    single_root_images = [os.path.join(single_root_dir, f) for f in os.listdir(single_root_dir) 
                         if f.endswith('.jpeg')]
    double_root_images = [os.path.join(double_root_dir, f) for f in os.listdir(double_root_dir) 
                         if f.endswith('.jpeg')]
    
    image_paths = single_root_images + double_root_images
    labels = [0] * len(single_root_images) + [1] * len(double_root_images)
    
    # Create folds
    folds = create_folds(image_paths, labels)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RootMorphologyModel().to(device)
    
    # Perform cross-validation
    results = perform_cross_validation(model, folds, device)
    
    # Print final results
    print("\nCross-validation Results:")
    for i, acc in enumerate(results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Mean Accuracy: {np.mean(results):.2f}% ± {np.std(results):.2f}%")

if __name__ == "__main__":
    main() 