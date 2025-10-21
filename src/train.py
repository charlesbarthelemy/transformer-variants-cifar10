import time
import torch
from tqdm import tqdm

def train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs):
    """
    Train the Performer model on CIFAR-10.

    Args:
        model (torch.nn.Module): Performer model instance.
        trainloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): LR scheduler.
        device (torch.device): 'cuda', 'mps', or 'cpu'.
        num_epochs (int): Number of training epochs.

    Returns:
        float: Total training time in seconds.
    """
    model.to(device)
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    return total_time



if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.model import Performer
    import torch.nn as nn
    import torch.optim as optim

    # Config
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trainset, _ = get_dataloaders(batch_size=128)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    model = Performer(dim=512, n_heads=8, depth=4, dropout=0.1, kernel="relu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs=2)