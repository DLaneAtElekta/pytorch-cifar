# Gradient Operations in tensorBASIC

This document describes the gradient/autograd operations available in tensorBASIC, powered by PyTorch's automatic differentiation.

## Overview

tensorBASIC now supports automatic differentiation (autograd) for training machine learning models. These operations are based on PyTorch's autograd system and allow you to:

1. Create trainable parameters
2. Perform forward pass computations
3. Compute gradients automatically with backpropagation
4. Update parameters using gradient descent

## Gradient Operations

### Creating Trainable Parameters

Use the `param()` function to create tensors with `requires_grad=True`:

```basic
10 let w = param(3, 2)     rem Create 3x2 weight matrix
20 let b = param(1, 2)     rem Create 1x2 bias vector
```

### Tensor Creation Functions

```basic
10 let x = zeros(3, 3)     rem Create 3x3 zero tensor
20 let y = randn(2, 4)     rem Create 2x4 random normal tensor
30 let z = param(5, 1)     rem Create 5x1 parameter tensor (requires_grad=True)
40 let t = tensor([[1,2],[3,4]])  rem Create tensor from list
```

### Forward Pass

Build your model using standard tensor operations:

```basic
10 let hidden = x @ w1 + b1     rem Linear layer
20 let output = hidden @ w2      rem Second layer
```

### Computing Gradients

Use `.backward()` to compute gradients:

```basic
10 let loss = diff.mean()        rem Must be scalar
20 let dummy = loss.backward()   rem Compute all gradients
```

**Important:** The tensor you call `.backward()` on must be a scalar (single value).

### Accessing Gradients

Use `.grad` property to access computed gradients:

```basic
10 let w_gradient = w.grad
20 let b_gradient = b.grad
```

### Parameter Updates

To update parameters, use the `detach()` function to break the computation graph:

```basic
10 let w_new = w - lr * w.grad   rem Compute update
20 let w = detach(w_new)         rem Apply update and re-enable gradients
```

**Why detach?** When you compute `w - lr * w.grad`, you create a new tensor in the computation graph. Using `detach()` removes it from the graph and re-enables gradient tracking for the next iteration.

### Zeroing Gradients

Clear gradients before the next backward pass:

```basic
10 let dummy = w.zero_grad()
20 let dummy = b.zero_grad()
```

## Complete Training Example

Here's a complete linear regression example:

```basic
10 rem Linear Regression with Gradient Descent
20 rem
30 rem Load data
40 dim x_data(10, 1)
50 let x_data = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
60 dim y_data(10, 1)
70 let y_data = [[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]]
80 rem
90 rem Initialize parameters
100 let w = param(1, 1)
110 let b = param(1, 1)
120 rem
130 rem Training loop
140 let lr = 0.01
150 for epoch = 1 to 100 step 1
160   rem Forward pass
170   let y_pred = x_data @ w + b
180   rem
190   rem Compute loss
200   let diff = y_pred - y_data
210   let loss = diff.mean()
220   rem
230   rem Backward pass
240   let dummy = loss.backward()
250   rem
260   rem Update parameters
270   let w_new = w - lr * w.grad
280   let b_new = b - lr * b.grad
290   let w = detach(w_new)
300   let b = detach(b_new)
310   rem
320   rem Zero gradients
330   let dummy2 = w.zero_grad()
340   let dummy3 = b.zero_grad()
350   rem
360 next epoch
370 rem
380 print "Final w:"
390 print w
400 print "Final b:"
410 print b
420 end
```

## Tensor Methods

TensorBASIC supports calling methods on tensors:

- `.backward()` - Compute gradients (scalar tensors only)
- `.zero_grad()` - Zero out gradients
- `.mean()` - Compute mean (returns scalar)
- `.sum()` - Compute sum
- `.detach()` - Detach from computation graph (use `detach(tensor)` function instead)
- Any other PyTorch tensor method

## Tensor Properties

Access tensor properties using dot notation:

- `.grad` - Access gradient tensor
- `.data` - Access underlying data
- Any other PyTorch tensor property

## Example Programs

The `examples/` directory contains these gradient-based ML examples:

1. **linear_regression.bas** - Simple linear regression (y = wx + b)
2. **logistic_regression.bas** - Binary classification with logistic regression
3. **neural_net.bas** - Forward pass example (no gradients)

## Requirements

Gradient operations require PyTorch to be installed:

```bash
pip install torch
```

If PyTorch is not available, tensorBASIC will fall back to NumPy backend (no gradient support).

## Training Tips

1. **Always use scalar loss**: Call `.mean()` or `.sum()` on your loss before `.backward()`
2. **Zero gradients**: Always call `.zero_grad()` before each backward pass
3. **Detach updates**: Use `detach()` when updating parameters
4. **Check convergence**: Print loss periodically to monitor training

## Limitations

- No built-in optimizers (SGD, Adam, etc.) - manual gradient descent only
- No activation functions built-in (sigmoid, relu, etc.) - must implement manually
- No automatic batching
- Checkpointing may not preserve gradient state perfectly

## Future Enhancements

Potential additions:
- Built-in activation functions
- Optimizer implementations
- Learning rate scheduling
- Gradient clipping
- Automatic mixed precision
