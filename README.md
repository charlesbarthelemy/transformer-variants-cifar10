# Efficient Transformers for Image Classification on CIFAR-10

This repository contains a comprehensive study comparing Vision Transformer (ViT) with efficient Performer variants for image classification on the CIFAR-10 dataset. This research was conducted as part of Columbia University's **Data Mining 1** course (Fall 2024) under the supervision of **Dr. Krzysztof Choromanski**, the lead author of the seminal paper "Rethinking Attention with Performers."

## Team Members
- **Charles Barthelemy** - Theory & Architecture, Performer-ReLU, Performer-EXP, Performer-fθ
- **Daniel Clepper** - Vision Transformer implementation and optimization
- **Eileen Feng** - Performer-EXP development and analysis
- **Mingyuan Li** - Vision Transformer baseline and evaluation
- **Ananya Rana** - Performer variants, alternative activations (ELU/GELU), Performer-fθ

The project explores the fundamental trade-offs between computational efficiency and classification accuracy in transformer architectures, building directly on Dr. Choromanski's pioneering work on linear attention mechanisms.

## Project Overview

Traditional Vision Transformers achieve excellent accuracy but suffer from quadratic complexity O(n²) in their attention mechanism, making them computationally expensive for longer sequences or resource-constrained environments. This project investigates **Performer architectures** that use kernel-based approximations to achieve linear O(n) complexity while maintaining competitive accuracy.

### Key Research Questions
- How do different kernel approximations affect the speed-accuracy trade-off?
- Can we design learnable kernels that outperform fixed kernel variants?
- What are the practical implications for deploying efficient transformers?

## Architecture Overview

### Vision Transformer (ViT)
- Standard softmax attention mechanism
- Patches input images into sequences
- Full quadratic attention complexity
- Baseline for comparison

### Performer Variants
All Performer models use the **FAVOR+ mechanism** to approximate attention with linear complexity:

- **Performer-ReLU**: Uses ReLU-based kernel `φ(x) = ReLU(Wx + b)`
- **Performer-EXP**: Uses exponential kernel with stability clamping
- **Performer-GELU/ELU**: Alternative activation functions
- **Performer-fθ**: Novel learnable kernel approach

## Results Summary

| Model | Accuracy (%) | Training Time (s/epoch) | Inference Time (s) |
|-------|-------------|------------------------|-------------------|
| ViT | 84.79% | 56s | 5.0s |
| Performer-ReLU | **87.39%** | 31s | 3.0s |
| Performer-GELU | **88.64%** | 31.5s | 3.05s |
| Performer-EXP | 86.17% | 34s | 3.02s |
| Performer-ELU | 77.48% | **13.11s** | **1.49s** |
| Performer-fθ | 85.08% | 29.85s | 2.95s |

### Key Findings
- **Performer-GELU achieved the highest accuracy (88.64%)** while maintaining similar computational efficiency to other Performer variants
- **Performer-ReLU provided the best balance** with 87.39% accuracy and robust training stability
- **All Performer variants significantly outperformed ViT** in training and inference speed
- **Linear complexity attention** enables practical deployment in resource-constrained environments

## Technical Highlights

### FAVOR+ Mechanism
The core innovation uses random feature maps to approximate softmax attention:
```
Attention(Q,K,V) ≈ φ(Q)(φ(K)ᵀV)
```
where `φ(·)` is a kernel-specific feature map.

### Stability Improvements
- Gradient clipping and mixed precision training
- Numerical stability through clamping and epsilon terms
- Advanced learning rate scheduling strategies

### Comprehensive Hyperparameter Optimization
- Grid search and Optuna optimization
- Cross-validation with stratified k-fold
- Extensive ablation studies on activation functions

## Repository Structure

```
transformer-variants-cifar10/
├── notebooks/
│   ├── ViT_Implementation.ipynb
│   ├── Performer_ReLU.ipynb
│   ├── Performer_EXP.ipynb
│   ├── Performer_GELU_ELU.ipynb
│   └── Performer_ftheta.ipynb
├── src/
│   ├── models/
│   ├── utils/
│   └── training/
├── results/
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── performance_metrics/
├── Report.pdf
├── requirements.txt
└── README.md
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transformer-variants-cifar10.git
   cd transformer-variants-cifar10
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run experiments:**
   ```bash
   # Train ViT baseline
   python src/train_vit.py
   
   # Train Performer variants
   python src/train_performer.py --variant relu
   python src/train_performer.py --variant exp
   ```

4. **Explore notebooks:**
   - Open Jupyter notebooks in `notebooks/` for detailed analysis
   - Each notebook contains complete model implementation and evaluation

## Experimental Setup

- **Dataset**: CIFAR-10 (60,000 32x32 color images, 10 classes)
- **Hardware**: NVIDIA A100 GPU (40GB memory)
- **Evaluation**: 80/20 train/validation split with 5-fold cross-validation
- **Metrics**: Classification accuracy, training time, inference time

### Data Augmentation
- Random horizontal flipping (50% probability)
- Random cropping with padding (32x32, padding=4)
- Channel-wise normalization

## Impact and Applications

This research demonstrates that efficient attention mechanisms can:
- **Reduce computational costs by 40-60%** while maintaining or improving accuracy
- **Enable deployment** on resource-constrained devices
- **Scale to longer sequences** without quadratic memory growth
- **Provide practical alternatives** to standard Vision Transformers

## Research Contributions

1. **Comprehensive benchmark** of efficient transformer variants
2. **Novel learnable kernel approach** (Performer-fθ)
3. **Detailed analysis** of speed-accuracy trade-offs
4. **Practical insights** for real-world deployment

## Future Directions

- Extension to larger datasets (ImageNet, CIFAR-100)
- Integration with other efficiency techniques (pruning, quantization)
- Advanced learnable kernel architectures
- Multi-modal applications

## References

- Choromanski et al. "Rethinking Attention with Performers" (2021)
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)

##� Acknowledgments

This project was conducted as part of **Columbia University's Data Mining course (Fall 2024)** under the supervision of **Dr. Krzysztof Choromanski**. Dr. Choromanski, as the lead author of the foundational paper "Rethinking Attention with Performers," provided invaluable guidance throughout this research. His pioneering work on FAVOR+ and linear attention mechanisms served as the theoretical foundation for our experimental investigations.

We extend our gratitude to Dr. Choromanski for his mentorship and for creating the original Performer architecture that made this comparative study possible. His insights into efficient attention mechanisms directly influenced our approach to implementing and evaluating these transformer variants.
