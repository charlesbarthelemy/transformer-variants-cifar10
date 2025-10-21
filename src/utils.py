import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def compute_predictions_and_labels(model, dataloader, device):
    """
    Compute model predictions, true labels, and probabilities.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_class_accuracy(model, dataloader, device, classes):
    """
    Plot accuracy per class.
    """
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))

    for p, t in zip(preds, labels):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    acc = class_correct / class_total * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=acc)
    plt.title("Accuracy per Class")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()


def show_classification_report(model, dataloader, device, classes):
    """
    Print classification report.
    """
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    report = classification_report(labels, preds, target_names=classes, digits=2)
    print("Classification Report:\n", report)


def plot_prediction_distribution(model, dataloader, device, bins=20):
    """
    Plot confidence distribution for correct vs incorrect predictions.
    """
    preds, labels, probs = compute_predictions_and_labels(model, dataloader, device)
    correct_mask = preds == labels
    predicted_probs = np.max(probs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(predicted_probs[correct_mask], bins=bins, alpha=0.7, label="Correct")
    plt.hist(predicted_probs[~correct_mask], bins=bins, alpha=0.7, color="red", label="Incorrect")
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()


def plot_confusion_matrix(model, dataloader, device, classes):
    """
    Plot confusion matrix.
    """
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


def plot_losses(train_losses, val_losses, num_epochs):
    """
    Plot training and validation loss curves.
    """
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training", linewidth=2)
    plt.plot(epochs, val_losses, "--", label="Validation", linewidth=2)
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()


def plot_accuracies(train_accuracies, val_accuracies, num_epochs):
    """
    Plot training and validation accuracy curves.
    """
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training", linewidth=2)
    plt.plot(epochs, val_accuracies, "--", label="Validation", linewidth=2)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()