import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.root_morphology_model import RootMorphologyModel
from data.preprocess import create_data_loaders

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        tuple: (predictions, true_labels, accuracy)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return np.array(all_predictions), np.array(all_labels), accuracy

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = RootMorphologyModel(num_classes=10).to(device)
    checkpoint = torch.load('models/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    data_dir = 'data'
    _, test_loader = create_data_loaders(data_dir)
    
    # Evaluate model
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Generate classification report
    class_names = [f'Class {i}' for i in range(10)]  # Replace with actual class names
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)
    print('\nConfusion matrix has been saved as confusion_matrix.png')

if __name__ == '__main__':
    main() 