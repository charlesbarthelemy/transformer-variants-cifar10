import os
import csv
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from src.model import Performer
from src.train import train_model
from src.evaluate import evaluate_model


def hyperparameter_tuning(parameters_to_test, trainset, num_epochs, k_folds=5, seed=42):
    """
    Perform grid search cross-validation over multiple hyperparameter combinations.

    Args:
        parameters_to_test (dict): Dict of hyperparameter lists to test.
        trainset (Dataset): Full training dataset.
        num_epochs (int): Number of training epochs.
        k_folds (int): Number of folds for cross-validation.
        seed (int): Random seed.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Extract search grids
    learning_rates = parameters_to_test['learning_rates']
    batch_sizes = parameters_to_test['batch_sizes']
    dims = parameters_to_test['dims']
    n_heads_list = parameters_to_test['n_heads_list']
    nb_features_list = parameters_to_test['nb_features_list']
    dropout_list = parameters_to_test['dropout']
    depths = parameters_to_test['depths']
    optimizer_funcs = parameters_to_test['optimizers']

    best_accuracy, best_params = 0.0, {}
    total_rounds = len(learning_rates) * len(batch_sizes) * len(dims) * len(n_heads_list) * \
                   len(nb_features_list) * len(depths) * len(optimizer_funcs) * len(dropout_list)

    csv_file = 'outputs/tuning_results.csv'
    os.makedirs('outputs', exist_ok=True)
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as csvfile:
        fieldnames = ['Round', 'Learning Rate', 'Batch Size', 'Dim', 'Dropout', 'Num Heads',
                      'Num Features', 'Depth', 'Optimizer', 'Num Epochs', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        current_round = 0
        targets = np.array(trainset.targets)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for lr, batch_size, dim, n_heads, nb_features, depth, dropout, optimizer_func in itertools.product(
                learning_rates, batch_sizes, dims, n_heads_list, nb_features_list, depths, dropout_list, optimizer_funcs):

            current_round += 1
            if dim % n_heads != 0:
                print(f"Skipping invalid config: dim={dim}, n_heads={n_heads}")
                continue

            print(f"\n[{current_round}/{total_rounds}] lr={lr}, bs={batch_size}, dim={dim}, heads={n_heads}, "
                  f"features={nb_features}, depth={depth}, dropout={dropout}, opt={optimizer_func}")

            total_accuracy = 0.0
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
                print(f"Fold {fold+1}/{k_folds}")

                train_subset = torch.utils.data.Subset(trainset, train_idx)
                val_subset = torch.utils.data.Subset(trainset, val_idx)

                trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

                model = Performer(num_classes=10, dim=dim, n_heads=n_heads, depth=depth, dropout=dropout, nb_features=nb_features).to(device)
                criterion = nn.CrossEntropyLoss()

                if optimizer_func == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

                train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs)
                accuracy, _ = evaluate_model(model, valloader, device)
                total_accuracy += accuracy

            avg_acc = total_accuracy / k_folds
            print(f"Average accuracy: {avg_acc:.2f}%")

            writer.writerow({
                'Round': current_round, 'Learning Rate': lr, 'Batch Size': batch_size,
                'Dim': dim, 'Num Heads': n_heads, 'Num Features': nb_features,
                'Depth': depth, 'Dropout': dropout, 'Optimizer': optimizer_func,
                'Num Epochs': num_epochs, 'Accuracy': avg_acc
            })

            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_params = {
                    'learning_rate': lr, 'batch_size': batch_size, 'dim': dim,
                    'dropout': dropout, 'n_heads': n_heads, 'nb_features': nb_features,
                    'depth': depth, 'optimizer': optimizer_func
                }

    print(f"\nBest Accuracy: {best_accuracy:.2f}% | Best Params: {best_params}")
    return best_params