# Performer-based Image Classification (CIFAR-10)

This project explores **efficient Transformer architectures** for image classification, with a focus on the **Performer** model (FAVOR+ linear attention).  
The goal was to study how kernel choices, attention mechanisms, and hyperparameter tuning affect performance on **CIFAR-10**.

---

## What’s Inside
- A clean **PyTorch implementation** of Performer with multiple kernel options:
  - `relu`, `elu`, `gelu`, and `exp` feature maps
- Full training pipeline with:
  - Stratified **k-fold cross-validation**
  - **Optuna** hyperparameter tuning
  - Early stopping & learning-rate scheduling
- A “best model” training script that:
  - Loads the optimal hyperparameters from `config.yaml`
  - Trains and evaluates the final model
  - Produces useful performance visualizations (class accuracy, confusion matrix, confidence distribution)

---

## Results (Best Configuration)
- **Accuracy:** ~86% on CIFAR-10 test set  
- Strong & stable performance across classes  
- Learned representations remain robust without full quadratic attention

---

## Repository Structure
.
├── notebooks/              # Exploration and experiments
├── src/
│   ├── model.py            # Performer model variants
│   ├── dataset.py          # Dataloaders + transforms
│   ├── train.py            # Train loop
│   ├── evaluate.py         # Metrics & evaluation
│   ├── tuning/             # Optuna + Grid Search
│   └── config.yaml         # Best hyperparameters (generated)
├── Report.pdf              # Short project write-up
└── README.md

---

## Quick Start
```bash
pip install -r requirements.txt
python src/tuning/optuna_tuning.py     # optional: find best hyperparameters
python src/train_best_model.py         # train + evaluate final model
