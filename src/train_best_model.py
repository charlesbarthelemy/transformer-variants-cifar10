import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Performer
from src.dataset import get_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model

# --- Load best hyperparameters ---
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)
best_params = config["best_params"]

# --- Assign parameters ---
lr = best_params["learning_rate"]
batch_size = best_params["batch_size"]
dim = best_params["dim"]
n_heads = best_params["n_heads"]
nb_features = best_params["nb_features"]
depth = best_params["depth"]
dropout = best_params["dropout"]
optimizer_type = best_params["optimizer"]

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data ---
trainset, testset = get_dataloaders(batch_size)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# --- Model ---
model = Performer(
    dim=dim,
    n_heads=n_heads,
    depth=depth,
    dropout=dropout,
    nb_features=nb_features,
    num_classes=10
).to(device)

criterion = nn.CrossEntropyLoss()

# --- Optimizer & Scheduler ---
if optimizer_type == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# --- Train & Evaluate ---
print("Starting final training...")
train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs=20)

print("\nEvaluating final model on test set...")
accuracy, inference_time = evaluate_model(model, testloader, device)
print(f"Final Test Accuracy: {accuracy:.2f}% | Inference Time: {inference_time:.2f}s")

# --- Save model ---
torch.save(model.state_dict(), "models/checkpoints/best_performer_model.pth")
print("Model saved to models/checkpoints/best_performer_model.pth")