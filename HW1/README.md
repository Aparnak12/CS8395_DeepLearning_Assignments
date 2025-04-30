# Homework 1 - CS 8395: Special Topics in Deep Learning

This assignment explores key deep learning concepts through implementation and analysis, including gradient descent behavior, MLP training dynamics, and the bias-variance trade-off under overparameterization.

---

## Q1: Gradient Descent and Convergence Rate (`DL_HW1_Q1.ipynb`)

Numerical validation of gradient descent convergence on a nonlinear regression task using Gaussian basis functions.

- **Q1.1**: Derives and implements gradient descent update rule.
- **Q1.2**: Computes theoretical upper bound for learning rate \( \epsilon \).
- **Q1.3**: Calculates Hessian eigenvalues to determine a numerical bound on \( \epsilon \).
- **Q1.4**: Runs gradient descent with \( \epsilon = 1/\lambda_{\max} \), logs:
  - Loss vs iteration plot.
  - Log of norm-squared gradient vs theoretical upper bound plot.
- **Q1.5**: Plots regression curve fit against data.

**Input File**: `hw1_p1.npy`

---

## Q2: Multi-Layer Perceptrons (MLP) (`DL_HW1_Q2 (1).ipynb`)

Examines how MLP size and initialization affect the solution to a classification task.

- **Q2.1**: Designs minimal architecture to solve a toy dataset. Includes weights, biases, and activations.
- **Q2.2**: Implements and trains the MLP across 10 random initializations. Plots accuracy histogram and analyzes failures.
- **Q2.3**: Repeats training with a larger hidden layer (10–100 units), compares histograms and results.

**Input File**: `hw1_p2.npy`

---

## Q3: Bias, Variance, and Overparameterization (`DL_HW1_Q3.ipynb`)

Analyzes the impact of model complexity and regularization on bias and variance using multiple synthetic datasets.

- **Q3.1**: Fits models with \( M = 7 \) for 10 datasets. Plots:
  - Each fit \( f_i \) vs ground truth \( f^* \)
  - Mean prediction \( \bar{f} \) vs \( f^* \)
  - Estimated bias and variance.
- **Q3.2**: Repeats above steps with \( M = 21 \); compares results with \( M = 7 \).
- **Q3.3**: Implements ridge regression with \( \lambda = 0.1 \) for \( M = 21 \); compares bias-variance behavior.

**Input Files**: `hw1_p3.npy`, `hw1_p3_gt.npy`

---

> All results are obtained using Python (NumPy, PyTorch/MLP as applicable), with visualizations done via matplotlib.
