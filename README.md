# Phase retrieval optimization algorithms

The repository contains a set of phase retrieval algorithms developed in the [thesis [Chapter 4]](https://www.theses.fr/2022LIMO0120) and implemented in Python. 

## Installation
To install the library, execute
```
pip install git+https://gitlab.com/shaxov/phrt_opt
```

## Problem formulation

Let $`A\in\mathbb{C}^{m\times n}`$ be a transmission matrix and $`b\in\mathbb{R}^{m}_+`$ be a vector of the square root of the intensity measurements, $`m > n`$. Then the problem is written as

```math
\text{Find} \; x\in\mathbb{C}^n \; \text{such that} \; |Ax| = b.
```

Typically, the number of measurements $`m`$ must be such that $`m \geq 4n`$ to find $`x`$. The transmission matrix $`A`$ must be a random matrix where real and imaginary parts of $`a_{ij}`$ must have a normal distribution.

<details>
<summary>To generate a phase retrieval problem the following Python code can be used.</summary>

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

</details>

## Implemented algorithms

The library contains four phase retrieval algorithms: gradient descent, Gauss-Newton, alterating projections, and ADMM. Each of these algorithms solve the original problem using its different equivalent reformulations.

<details>
<summary><h3>Gradient descent</h3></summary>

[Section 4.2](https://www.theses.fr/2022LIMO0120)

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
</details>

<details>
<summary><h3>Gauss-Newton</h3></summary>

[Section 4.3](https://www.theses.fr/2022LIMO0120)

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

Then $`x^{(k+1)} = x^{(k)} + \alpha^{(k)} p^{(k)}_{1:n}`$, where subscript $`_{1:n}`$ means that we take the first $`n`$ elements of vector $`p`$. The step length $`\alpha^{(k)}`$ can be computed in the same way as for the gradient descent method. 

To use the Gauss-Newton method the following Python code can be used.

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
</details>

<details>
<summary><h3>Alternating projections</h3></summary>

[Section 4.5](https://www.theses.fr/2022LIMO0120)

The equivalent reformulation of the original problem writes

```math
\min_{(x,y)\in\mathbb{C}^n\times\mathbb{C}^m} \frac{1}{2} \|Ax-y\|^2 \;\; \text{such that} \;\; |y| = b.
```

The algorithm contains two consecutive updates:
```math
\begin{align*}
y^{(k+1)} &= b\odot\exp\big( i\arg(Ax^{(k)}) \big),\\
x^{(k+1)} &= A^\dag y^{(k+1)},
\end{align*}
```
where $`A^\dag`$ is a Moore-Penrose inverse.

To use the Gauss-Newton method the following Python code can be used.

```python
import phrt_opt

x_hat = phrt_opt.methods.alternating_projections(tm, b)
```
</details>

<details>
<summary><h3>ADMM</h3></summary>

[Section 4.6](https://www.theses.fr/2022LIMO0120)

The equivalent reformulation of the original problem writes

```math
\min_{(y,z,\xi)\in\mathbb{C}^m\times\mathbb{C}^m\times\mathbb{C}^m} \frac{1}{2} \|\xi\|^2 \;\;
\text{such that} \;\; y - z = \xi, \; y\in \operatorname{range}(A), \; z\in\mathcal{M}_b,
```

where then $`x = A^\dag y`$,  $`\mathcal{M}_b = \{ z\in\mathbb{C}^m:|z|=b \}`$.

The algorithm contains three consecutive updates:

```math
\begin{align*}
z^{(k+1)} &= b\odot\exp\big( i\arg(y^{(k)} + (1 - \rho^{(k)})) \big),\\
y^{(k+1)} &= AA^\dag z^{(k+1)},\\
\lambda^{(k+1)} &= \frac{1}{1 + \rho^{(k+1)}}\big( \lambda^{(k)} + y^{(k+1) - z^{(k+1)} \big)
\end{align*}
```

where $`A^\dag`$ is a Moore-Penrose inverse and variable $`\xi`$ was eliminated and parameter $`\rho^{(k)}`$ is updated by one of the following strategies: `constant`, `linear`, `exponential`, and `auto`. The default strategy is set to `auto` as the best one.

```python
import phrt_opt

x_hat = phrt_opt.methods.admm(tm, b)
```

#### Strategies for parameter $`\rho`$

A parameter $`\rho^{(k)}`$ can be updated by one of the following strategies:
<details>
<summary>Constant</summary>

```python
x_hat = phrt_opt.methods.admm(
    tm, b,
    strategy=phrt_opt.strategies.constant(.5),
)
```

</details>

<details>
<summary>Linear</summary>

```python
x_hat = phrt_opt.methods.admm(
    tm, b,
    strategy=phrt_opt.strategies.linear(),
)
```

</details>


<details>
<summary>Exponential</summary>

```python
x_hat = phrt_opt.methods.admm(
    tm, b,
    strategy=phrt_opt.strategies.exponential(),
)
```

</details>

<details>
<summary>Auto</summary>

```python
x_hat = phrt_opt.methods.admm(
    tm, b,
    strategy=phrt_opt.strategies.auto(),
)
```

</details>

</details>



