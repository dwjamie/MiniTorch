# MiniTorch

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

MiniTorch is a pure Python re-implementation of the Torch API, aiming for clarity and simplicity.

It incorporates core features like auto-differentiation, backpropagation, tensor operations, broadcasting, along with GPU support and parallel computing capabilities.

Docs: https://minitorch.github.io/

## Setting Up

### Create a Virtual Environment

```
python3 -m venv minitorch-env
```

### Activate the Virtual Environment

```
source minitorch-env/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

## Getting Started

### Clone the Codebase

```
git clone https://github.com/dwjamie/MiniTorch.git
```

### Install or Develop

To install:

```
python setup.py install
```

Or for development:

```
python setup.py develop
```

## Examples

### 1. Create a Tensor

```python
from minitorch import Tensor

t1 = Tensor(2.0)
t2 = Tensor(3.0)
t3 = t1 + t2
print(t3)  # Output: Tensor(5.0, requires_grad=False)
```

### 2. Autograd

```python
from minitorch import Tensor

t1 = Tensor(2.0, requires_grad=True)
t2 = Tensor(3.0)
t3 = t1 + t2
t4 = t1 * t3
t4.backward()
print(f"t1 grad: {t1.grad}")  # Output: t1 grad: Tensor(5.0, requires_grad=False)
print(f"t2 grad: {t2.grad}")  # Output: t2 grad: None
```

### 3. Gradient for Broadcast

```python
from minitorch import Tensor

t1 = Tensor([1.0, 2.0], requires_grad=True)
t2 = Tensor(2.0, requires_grad=True)
t3 = t1 + t2
t3.backward(Tensor([1.0, 1.0]))
print(f"t1 grad: {t1.grad}")  # Output: t1 grad: Tensor([1., 1.], requires_grad=False)
print(f"t2 grad: {t2.grad}")  # Output: t2 grad: Tensor(2.0, requires_grad=False)
```

### 4. Create a Neural Network

```python
import minitorch
import minitorch.nn as nn

input = minitorch.rand(2, 3)
linear = nn.Linear(3, 5, bias=True)
output = linear(input)
print(f"output: {output}")

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(3, 5, bias=True)
        self.linear_2 = nn.Linear(5, 6)

    def forward(self, input):
        output = self.linear_1(input)
        output = self.linear_2(output)
        return output

input = minitorch.rand(2, 3)
model = Model()
output = model(input)
print(f"output: {output}")

for name, module in model.named_modules(prefix='model'):
    print(f"{name}: {module}")
```

## Tools

- **mypy:** A static type checker for Python.
- **Flake8:** Your tool for style guide enforcement.
- **unittest:** For unit testing.

## References

- [PyTorch](https://pytorch.org/)
- [autograd](https://github.com/HIPS/autograd)
- [tinygrad](https://github.com/geohot/tinygrad)
