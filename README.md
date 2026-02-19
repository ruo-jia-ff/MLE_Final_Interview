# CNN-Based Euclidean Distance Model (PyTorch)

## Overview

This project implements a **PyTorch Convolutional Neural Network (CNN)** that computes the **Euclidean Distance** between two vectors.

For two vectors:

- \( U = (u_1, \dots, u_n) \)  
- \( V = (v_1, \dots, v_n) \)

The Euclidean distance is defined as:

\[
\text{Euclidean\_Distance}(U, V)
=
\sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}
\]

The CNN architecture structurally reproduces this exact mathematical computation using convolution, elementwise squaring, summation, and a square root operation.

---

## Mathematical Implementation

The model computes the Euclidean distance in four stages:

1. **Subtraction**  
   A convolution layer with kernel size `(2,1)` operates across the stacked vector pair to compute:
   \[
   (u_i - v_i)
   \]

2. **Squaring**  
   A custom activation function squares each element:
   \[
   (u_i - v_i)^2
   \]

3. **Summation**  
   A fully connected layer (with weights fixed to 1 and no bias) sums all squared values:
   \[
   \sum_{i=1}^{n} (u_i - v_i)^2
   \]

4. **Square Root**  
   The final output applies:
   \[
   \sqrt{\cdot}
   \]

This exactly matches the Euclidean distance formula.

---
