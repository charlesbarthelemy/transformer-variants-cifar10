import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import csv
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.model import Performer
from src.train import train_model
from src.evaluate import evaluate_model

def hyperparameter_tuning(trainset, parameters_to_test, num_epochs, k_folds=5, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract hyperparameter grids
    learning_rates = parameters_to_test['learning_rates']
    batch_sizes = parameters_to_test['batch_sizes']
    dims = parameters_to_test['dims']
    n_heads_list = parameters_to_test['n_heads_list']
    nb_features_list = parameters_to_test['nb_features_list']
    dropout_list = parameters_to_test['dropout']
    depths = parameters_to_test['depths']
    optimizer_funcs = parameters_to_test['optimizers']

    best_accuracy = 0.0
    best_params = {}

    # Compute total configurations for progress tracking
    total_rounds = len(learning_rates) * len(batch_sizes) * len(dims) * len(n_heads_list) * \
                   len(nb_features_list) * len(depths) * len(optimizer_funcs) * len(dropout_list)
    current_round = 0

    # Prepare the CSV file for logging
    csv_file = 'tuning_results.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as csvfile:
        fieldnames = ['Round', 'Learning Rate', 'Batch Size', 'Dim', 'Dropout', 'Num Heads',
                      'Num Features', 'Depth', 'Optimizer', 'Num Epochs', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only once
        if not file_exists:
            writer.writeheader()

        # Iterate over the hyperparameter combinations
        for lr, batch_size, dim, n_heads, nb_features, depth, dropout, optimizer_func in itertools.product(
                learning_rates, batch_sizes, dims, n_heads_list, nb_features_list, depths, dropout_list, optimizer_funcs):

            current_round += 1

            # Skip invalid configurations
            if dim % n_heads != 0:
                print(f"Skipping configuration: dim={dim}, n_heads={n_heads} (not divisible)")
                continue

            print(f"\nTraining Configuration [{current_round}/{total_rounds}]")
            print(f"Learning Rate: {lr}, Batch Size: {batch_size}, Dim: {dim}, Heads: {n_heads}, "
                  f"Features: {nb_features}, Depth: {depth}, Dropout: {dropout}, Optimizer: {optimizer_func}")

            # Initialize variable to accumulate accuracy over folds
            total_accuracy = 0.0

            # Initialize StratifiedKFold
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

            # Since trainset.targets is a list, we need to convert it to a numpy array
            targets = np.array(trainset.targets)

            for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(targets)), targets)):
                print(f"Fold {fold+1}/{k_folds}")

                # Create subsets
                train_subset = torch.utils.data.Subset(trainset, train_indices)
                val_subset = torch.utils.data.Subset(trainset, val_indices)

                # Data loaders
                trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
                valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

                # Initialize the model with current hyperparameters
                model = Performer(num_classes=10, dim=dim, n_heads=n_heads, depth=depth, dropout=dropout, nb_features=nb_features)
                model.to(device)

                # Set the optimizer and scheduler based on the selected function
                criterion = nn.CrossEntropyLoss()
                if optimizer_func == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
                elif optimizer_func == 'SGD':
                    lr_sgd = lr * 10  # Adjusting learning rate for SGD
                    optimizer = optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

                # Train the model
                try:
                    train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs)
                except RuntimeError as e:
                    print(f"RuntimeError during training: {e}")
                    continue

                # Validate the model
                try:
                    accuracy, _ = evaluate_model(model, valloader, device)
                except RuntimeError as e:
                    print(f"RuntimeError during evaluation: {e}")
                    continue

                print(f"Fold {fold+1}, Accuracy: {accuracy:.2f}%")
                total_accuracy += accuracy

            # Average accuracy over folds
            average_accuracy = total_accuracy / k_folds
            print(f"Average Accuracy over {k_folds} folds: {average_accuracy:.2f}%")

            # Log the results to the CSV file
            writer.writerow({
                'Round': current_round,
                'Learning Rate': lr,
                'Batch Size': batch_size,
                'Dim': dim,
                'Num Heads': n_heads,
                'Num Features': nb_features,
                'Depth': depth,
                'Dropout': dropout,
                'Optimizer': optimizer_func,
                'Num Epochs': num_epochs,
                'Accuracy': average_accuracy
            })

            # Update the best parameters if accuracy improves
            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_params = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'dim': dim,
                    'dropout': dropout,
                    'n_heads': n_heads,
                    'nb_features': nb_features,
                    'depth': depth,
                    'optimizer': optimizer_func
                }

    print(f"\nBest Average Accuracy: {best_accuracy:.2f}% with Parameters: {best_params}")
    return best_params