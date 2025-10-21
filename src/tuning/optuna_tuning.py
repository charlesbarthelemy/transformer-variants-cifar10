import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from src.model import Performer
from src.train import train_model
from src.evaluate import evaluate_model


def objective(trial, trainset, num_epochs=10, k_folds=5, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Search space
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dim = trial.suggest_categorical('dim', [128, 256, 512])
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    nb_features = trial.suggest_categorical('nb_features', [64, 128, 256])
    depth = trial.suggest_int('depth', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    if dim % n_heads != 0:
        raise optuna.TrialPruned()

    targets = np.array(trainset.targets)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    total_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train_subset = torch.utils.data.Subset(trainset, train_idx)
        val_subset = torch.utils.data.Subset(trainset, val_idx)

        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = Performer(num_classes=10, dim=dim, n_heads=n_heads, depth=depth, dropout=dropout, nb_features=nb_features).to(device)
        criterion = nn.CrossEntropyLoss()

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs)
        acc, _ = evaluate_model(model, valloader, device)
        total_accuracy += acc

    return total_accuracy / k_folds  # Optuna will maximize this


def optimize_hyperparameters(trainset, n_trials=30):
    study = optuna.create_study(direction='maximize', study_name='PerformerOptuna')
    study.optimize(lambda trial: objective(trial, trainset), n_trials=n_trials)

    print("\nBest Accuracy: {:.2f}%".format(study.best_value))
    print("Best Parameters:", study.best_params)
    return study.best_params