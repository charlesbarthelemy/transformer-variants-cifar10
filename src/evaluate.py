import torch
import time
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
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
    print(f"Accuracy: {accuracy:.2f}%, Inference Time: {inference_time:.2f} seconds")
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