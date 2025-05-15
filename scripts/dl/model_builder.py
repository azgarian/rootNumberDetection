"""
Contains PyTorch model code to instantiate a TinyVGG model and
functions for training and testing a PyTorch model.
"""
from typing import Dict, List, Tuple
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scripts.dl import helper_functions as hf


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture for binary classification."""
    def __init__(self, input_shape: int, hidden_units: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2809*hidden_units, hidden_units),  # Adjust the input size to 28090
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
)

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device).float().view(-1, 1)  # Ensure y is float and reshaped for BCEWithLogitsLoss

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.round(torch.sigmoid(y_pred))  # Binary classification
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

from typing import Tuple, List

from typing import Tuple, List
import torch

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, List[int], List[int], List[float]]:
    model.eval()
    test_loss, test_acc = 0, 0
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.inference_mode():
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device).float().view(-1, 1)  # Ensure y is float and reshaped for BCEWithLogitsLoss

            test_pred_logits = model(x)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Get predicted probabilities and labels
            test_pred_probs = torch.sigmoid(test_pred_logits)  # Keep as tensor
            test_pred_labels = torch.round(test_pred_probs)  # Round tensor for predicted labels

            # Accumulate true labels and predictions
            y_true.extend(y.cpu().numpy())  # Convert true labels to numpy
            y_pred.extend(test_pred_labels.cpu().numpy())  # Convert predicted labels to numpy
            y_scores.extend(test_pred_probs.cpu().numpy())  # Convert probabilities to numpy

            # Calculate accuracy
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc, y_true, y_pred, y_scores

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          log_dir: str,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model with TensorBoard logging.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        log_dir: Directory for TensorBoard logs.
        device: A target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        A dictionary of training and testing metrics across epochs.
    """
    writer = SummaryWriter(log_dir=log_dir)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc, y_true, y_pred, y_scores = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Calculate precision, recall, F1 score, and ROC-AUC
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, y_scores)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Precision/test", precision, epoch)
        writer.add_scalar("Recall/test", recall, epoch)
        writer.add_scalar("F1 Score/test", f1, epoch)
        writer.add_scalar("ROC AUC/test", roc_auc, epoch)

        # Plot ROC curve and confusion matrix
        hf.plot_roc_curve(y_true, y_scores, writer, epoch)
        hf.plot_confusion_matrix(y_true, y_pred, writer, epoch)

        # Sample predictions visualization
        #inputs, labels = next(iter(test_dataloader))
        #predictions = model(inputs.to(device)).argmax(dim=1)
        #hf.plot_sample_predictions(inputs, labels, predictions.cpu(), writer, epoch)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"precision: {precision:.4f} | "
            f"recall: {recall:.4f} | "
            f"f1_score: {f1:.4f} | "
            f"roc_auc: {roc_auc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        results["roc_auc"].append(roc_auc)

        if scheduler:
            scheduler.step()

    writer.close()  # Close writer at the end of training

    # Return the filled results at the end of the epochs
    return results
