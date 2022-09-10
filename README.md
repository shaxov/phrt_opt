# Optimization algorithms for solving phase retrieval problem

The repository contains a set of phase retrieval algorithms that are implemented in Python.

## 1. Problem formulation

Let $`A\in\mathbb{C}^{m\times n}`$ be a transmission matrix and $`b\in\mathbb{R}^{m}_+`$ be a vector of the square root of intensity measurements, $`m > n`$. Then the problem writes as

```math
\text{Find} \; x\in\mathbb{C}^n \; \text{such that} \; |Ax| = b.
```

Typically, the number of measurements $`m`$ must be such that $`m \geq 4n`$ to find $`x`$. The transmission matrix $`A`$ must be a random matrix where real and imaginary parts of $`a_{ij}`$ must have a normal distribution.
