import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from sklearn.model_selection import StratifiedKFold
from src.model import Performer
from src.train import train_model
from src.evaluate import evaluate_model
from src.dataset import get_dataloaders


def objective(trial, trainset, num_epochs=7, k_folds=5, seed=42):
    """
    Objective function for Optuna hyperparameter tuning with k-fold CV.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # Define search space
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dim = trial.suggest_categorical('dim', [128, 256, 512])
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    nb_features = trial.suggest_categorical('nb_features', [64, 128, 256])
    depth = trial.suggest_int('depth', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    kernel = trial.suggest_categorical("kernel", ["relu", "elu", "gelu", "exp"])

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

        model = Performer(
            num_classes=10,
            dim=dim,
            n_heads=n_heads,
            depth=depth,
            dropout=dropout,
            nb_features=nb_features,
            kernel=kernel
        ).to(device)

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

    return total_accuracy / k_folds


def optimize_hyperparameters(n_trials=30):
    """
    Run Optuna optimization, save best hyperparameters (YAML) and all trials (CSV).
    """

    trainset, _, _, _= get_dataloaders()

    study = optuna.create_study(direction='maximize', study_name='PerformerOptuna')
    study.optimize(lambda trial: objective(trial, trainset), n_trials=n_trials)

    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"\nBest Accuracy: {best_accuracy:.2f}%")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Save best parameters to YAML
    config = {"best_params": best_params, "best_accuracy": best_accuracy}
    os.makedirs("src", exist_ok=True)
    yaml_path = os.path.join("src", "config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nBest parameters saved to: {yaml_path}")

    # Save all trials to CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "optuna_trials.csv")
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)

    print(f"All Optuna trials saved to: {csv_path}")

    return best_params

  
if __name__ == "__main__":
    optimize_hyperparameters(n_trials=50)