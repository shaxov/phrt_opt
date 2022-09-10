# Optimization algorithms for solving phase retrieval problem

The repository contains a set of phase retrieval algorithms that are implemented in Python. To install the library run in terminal
```
pip install git+https://gitlab.xlim.fr/shpakovych/phrt-opt
```

## Problem formulation

Let $`A\in\mathbb{C}^{m\times n}`$ be a transmission matrix and $`b\in\mathbb{R}^{m}_+`$ be a vector of the square root of intensity measurements, $`m > n`$. Then the problem writes as

```math
\text{Find} \; x\in\mathbb{C}^n \; \text{such that} \; |Ax| = b.
```

Typically, the number of measurements $`m`$ must be such that $`m \geq 4n`$ to find $`x`$. The transmission matrix $`A`$ must be a random matrix where real and imaginary parts of $`a_{ij}`$ must have a normal distribution.

To generate a phase retrieval problem the following Python code can be used.

```python
import numpy as np

n = 8
m = 8 * n

# Transmission matrix
tm = np.random.randn(m, n) + 1j * np.random.randn(m, n)

# Solution vector
x = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)

# Vector of intensity measurements
b = np.abs(tm @ x)
```

## Implemented algorithms

The library contains four phase retrieval algorithms: gradient descent, Gauss-Newton, alterating projections, and ADMM. Each of these algorithms solve the original problem using its different equivalent reformulations.

### Gradient descent

The equivalent reformulation of the original problem writes

```math
\min_{x\in\mathbb{C}^n} f(x):=\frac{1}{2}\||Ax|^2 - b^2\|^2.
```

The algorithm is a simple gradient descent $`x^{(k+1)} = x^{(k)} - \alpha^{(k)} \nabla f(x^{(k)})`$, where

```math
\nabla f(x) = \frac{1}{m}A^*\big[(|Ax|^2-b^2)\odot Ax \big],
```

is caclucated by means of Wirtinger calculus and $`\odot`$ operator means component by component multiplication of two vectors. 

To use the gradient descent method the following Python code can be used.

```python
import phrt_opt

x_hat = phrt_opt.methods.gradient_descent(tm, b)
```
#### Line-search

The step length $`\alpha^{(k)}`$ can be calculated using:
* backtracking line-search (Armijo)
```python
x_hat = phrt_opt.methods.gradient_descent(
    tm, b,
    linesearch=phrt_opt.linesearch.Backtracking(),
)
 ```
* line-search, which is based on a secant equation (Barzilai and Borwein).
```python
x_hat = phrt_opt.methods.gradient_descent(
    tm, b,
    linesearch=phrt_opt.linesearch.Secant(),
)
 ```

## Gauss-Newton

The equivalent reformulation of the original problem writes

```math
\min_{x\in\mathbb{C}^n} f(x):=\frac{1}{2m}\||Ax|^2 - b^2\|^2.
```
Following the general scheme of Gauss-Newton method, we denote a residual function $`r:\mathbb{C}^n\rightarrow\mathbb{R}^m_+`$ as

```math
r(x) = |Ax|^2 - b^2,
```
and its jacobian matrix in terms of Wirtinger calculus writes as

```math
\nabla r(x) = 
\begin{pmatrix}
    A\odot \bar{A}\bar{x} & \bar{A}\odot Ax
\end{pmatrix}
\subset \mathbb{C}^{m\times 2n},
```

Then, the descent direction $`p\in\mathbb{C}^{2n}`$ is a solution of the system

```math
\nabla r(x)^* \nabla r(x) p = - \nabla r(x)^* r(x).
```

To use the gradient descent method the following Python code can be used.

```python
import phrt_opt

x_hat = phrt_opt.methods.gauss_newton(tm, b)
```

### Symmetric system solver

There are two implemented methods that can be used for solving the Gauss-Netwon system at each iteration:
* Cholesky solver
```python
x_hat = phrt_opt.methods.gauss_newton(
    tm, b,
    quadprog=phrt_opt.quadprog.Cholesky(),
)
```
* Conjugate gradient descent solver
```python
x_hat = phrt_opt.methods.gauss_newton(
    tm, b,
    quadprog=phrt_opt.quadprog.ConjugateGradient(),
)
```
