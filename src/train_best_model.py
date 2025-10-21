import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import Performer
from src.evaluate import evaluate_model, validate_model
from src.utils import plot_losses, plot_accuracies, plot_confusion_matrix, plot_class_accuracy, show_classification_report, plot_prediction_distribution


def train_best_model(patience=10, num_epochs=100):

        # --- Load best hyperparameters ---
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    best_params = config["best_params"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr, batch_size, dim, n_heads = best_params['learning_rate'], best_params['batch_size'], best_params['dim'], best_params['n_heads']
    depth, dropout, nb_features = best_params['depth'], best_params['dropout'], best_params['nb_features']
    optimizer_type = best_params['optimizer']
    kernel = best_params.get("kernel", "relu")

    # Load data
    trainset, testset, _, _ = get_dataloaders()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = Performer(num_classes=10, dim=dim, n_heads=n_heads, depth=depth, dropout=dropout, nb_features=nb_features, kernel=kernel)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Compute training accuracy at the end of epoch
        train_acc, _ = evaluate_model(model, trainloader, device)
        train_accuracies.append(train_acc)

        # Validation
        val_loss = validate_model(model, testloader, criterion, device)
        val_losses.append(val_loss)
        val_acc, _ = evaluate_model(model, testloader, device)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping activated!")
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Plot training curves
    plot_losses(train_losses, val_losses, len(train_losses))
    plot_accuracies(train_accuracies, val_accuracies, len(val_accuracies))

    # Plot confusion matrix
    plot_confusion_matrix(model, testloader, device, testset.classes)

    # Plot class accuracy
    plot_class_accuracy(model, testloader, device, testset.classes)

    # Show classification report
    show_classification_report(model, testloader, device, testset.classes)

    # Plot distribution of predictions (confidence)
    plot_prediction_distribution(model, testloader, device, testset.classes)

    return model