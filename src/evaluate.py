import time
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# -----------------------------
# 1. Quantitative evaluation
# -----------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    start_time = time.time()
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    inference_time = time.time() - start_time
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%, Inference Time: {inference_time:.2f} s")
    return accuracy, inference_time


def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss


# -----------------------------
# 2. Analytical / visualization tools
# -----------------------------
def compute_predictions_and_labels(model, dataloader, device):
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


def plot_confusion_matrix(model, dataloader, device, classes):
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()


def show_classification_report(model, dataloader, device, classes):
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    report = classification_report(labels, preds, target_names=classes, digits=2)
    print("Classification Report:\n", report)


def plot_class_accuracy(model, dataloader, device, classes):
    preds, labels, _ = compute_predictions_and_labels(model, dataloader, device)
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))
    for p, t in zip(preds, labels):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1
    class_accuracy = class_correct / class_total * 100.0
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=class_accuracy)
    plt.title("Accuracy per Class")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()