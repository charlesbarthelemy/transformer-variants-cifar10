import time
from tqdm import tqdm

def train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs):
    model.to(device)
    start_time = time.time()  # Start timing

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")

    training_time = time.time() - start_time  # End timing
    print(f"Training Time: {training_time:.2f} seconds")
    return training_time