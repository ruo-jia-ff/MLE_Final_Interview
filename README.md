# CNN-Based Euclidean Distance Model (PyTorch)

## Overview

This project implements a Convolutional Neural Network (CNN) in PyTorch that computes the Euclidean distance between two vectors.

For two vectors:

U = (u1, u2, ..., un)  
V = (v1, v2, ..., vn)

The Euclidean distance between U and V is defined as:

Euclidean_Distance(U, V) = sqrt( sum from i=1 to n of (ui - vi)^2 )

The network architecture is designed to structurally reproduce this exact mathematical computation.

---

## Mathematical Interpretation

The model performs the Euclidean distance calculation in four conceptual steps:

1. Element-wise subtraction  
   Compute the difference between corresponding elements of U and V:
   (ui - vi)

2. Squaring  
   Square each difference:
   (ui - vi)^2

3. Summation  
   Sum all squared values:
   sum from i=1 to n of (ui - vi)^2

4. Square root  
   Take the square root of the total sum:
   sqrt( sum from i=1 to n of (ui - vi)^2 )

The final output of the network matches the Euclidean distance definition exactly.

---

## Input Structure

The model expects each sample to contain a pair of vectors (U, V) of equal length n.

Each input represents:
- One vector U
- One vector V
- Both stacked together as a single sample

The batch dimension allows multiple vector pairs to be processed simultaneously.

---

## Training Objective

The network is trained using supervised learning.

For each pair (U, V):
- The model produces a predicted distance.
- The true Euclidean distance is computed analytically.
- The loss measures the difference between predicted and true distance.

The objective is to minimize the error between the network output and the true Euclidean distance.

---

## Educational Purpose

Although Euclidean distance can be computed directly using a formula, this project demonstrates:

- How neural networks can represent deterministic mathematical functions.
- How convolutional structures can encode structured arithmetic operations.
- The relationship between neural network layers and classical mathematical operators.

This project is intended for educational and analytical exploration of how CNNs can model exact mathematical transformations.
