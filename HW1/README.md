# Homework 1 - CS 8395: Special Topics in Deep Learning

This assignment explores key deep learning concepts through implementation and analysis, including gradient descent behavior, MLP training dynamics, and the bias-variance trade-off under overparameterization.

---

## Q1: Gradient Descent and Convergence Rate (`DL_HW1_Q1.ipynb`)

Numerical validation of gradient descent convergence on a nonlinear regression task using Gaussian basis functions.

- **Q1.1**: Derives and implements gradient descent update rule.
- **Q1.2**: Computes theoretical upper bound for the learning rate (ε).
- **Q1.3**: Calculates eigenvalues of the Hessian matrix to determine a numerical bound on ε.
- **Q1.4**: Runs gradient descent with ε = 1 / λ_max, and logs:
  - Loss vs. iteration plot
  - Log of norm-squared gradient vs. theoretical upper bound plot
- **Q1.5**: Plots the final regression curve fit against the dataset.

**Input File**: `hw1_p1.npy`

---

## Q2: Multi-Layer Perceptrons (MLP) (`DL_HW1_Q2.ipynb`)

Examines how MLP size and initialization affect the solution to a toy classification task.

- **Q2.1**: Designs the smallest possible MLP to solve the problem. Includes weights, biases, and activations.
- **Q2.2**: Trains the MLP over 10 random initializations and plots a histogram of accuracies. Analyzes any training failures.
- **Q2.3**: Repeats with a larger hidden layer (10–100 units), compares accuracy distributions and training behavior.

**Input File**: `hw1_p2.npy`

---

## Q3: Bias, Variance, and Overparameterization (`DL_HW1_Q3.ipynb`)

Analyzes the impact of model complexity and regularization on bias and variance using multiple synthetic datasets.

- **Q3.1**: Trains models with M = 7 on 10 datasets. Plots:
  - Each individual fit (fᵢ) vs. ground truth (f*)
  - Mean prediction (f̄) vs. f*
  - Estimates of bias and variance
- **Q3.2**: Repeats with M = 21 and compares results to M = 7.
- **Q3.3**: Uses ridge regression with λ = 0.1 for M = 21. Compares updated bias and variance behavior.

**Input Files**: `hw1_p3.npy`, `hw1_p3_gt.npy`

---

> All results are implemented in Python using NumPy and PyTorch, with plots generated using Matplotlib.
