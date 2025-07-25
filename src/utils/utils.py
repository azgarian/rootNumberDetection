import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def create_directory(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, path):
    """
    Save image to file.
    
    Args:
        image (numpy.ndarray): Image to save
        path (str): Path to save the image
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:  # If image is in CxHxW format
        image = np.transpose(image, (1, 2, 0))
    
    # Denormalize if image is normalized
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_predictions(model, data_loader, device, num_samples=5, save_dir=None):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: The model to use for predictions
        data_loader: DataLoader containing images
        device: Device to run inference on
        num_samples (int): Number of samples to visualize
        save_dir (str, optional): Directory to save visualizations
    """
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        
        plt.imshow(img)
        plt.title(f'True: {labels[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'predictions.png'))
        plt.close()
    else:
        plt.show()

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary containing metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 