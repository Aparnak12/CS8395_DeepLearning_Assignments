# Homework 2 - CS 8395: Special Topics in Deep Learning

This assignment explores theoretical and practical aspects of deep learning. The third question focuses on implementing and evaluating deep architectures for signal classification under translation transformations.

## Q3: Deep Learning Blueprint (`DL_HW2_Q3.ipynb`)

This notebook classifies 1D signals that differ by the number of Gaussian "bumps" and tests how model architecture and data augmentation impact performance.

**Dataset**: `hw2_p3.pkl` (contains 25 samples per class for training, plus test set)

### Q3.1: MLP Classifier (Interpolation Regime)

- Implements a fully connected neural network (MLP) with sufficient parameters to achieve 100% training accuracy.
- Uses Adam optimizer.
- Logs training and test accuracy per epoch.
- Plots:
  - Train vs test accuracy.
- Reports total number of parameters and comments on generalization performance.

### Q3.2: MLP with Data Augmentation

- Introduces random **circular shift** as data augmentation during training using `torch.roll`.
- Trains the same MLP with augmented data.
- Plots:
  - Train vs test accuracy for augmented regime.
- Discusses the effect of this augmentation on model invariance and generalization.

### Q3.3: Convolutional Network with Translation Invariance

- Builds a **1D CNN** composed of:
  - A `Conv1d` layer with circular padding for translation **equivariance**
  - A non-linearity (e.g., ReLU)
  - A **global average pooling** layer for translation **invariance**
  - A final `Linear` layer for classification
- Trains the network with Adam.
- Logs and plots train and test accuracy.
- Reports parameter count and generalization performance.

---

> All coding is done in PyTorch. Plots include appropriate labels and legends.
