# Introduction

[PyTorch](https://pytorch.org/) is the go-to Python framework for developing deep learning applications.
Most of the ML Research community uses PyTorch due to its ease of use, clear APIs, and reasonable learning curve (although it is steep at the start!).

Before PyTorch, Tensorflow (from Google) was the champion framework of ML Research community, but it had many issues. One of the main issues with Tensorflow at the time was its rigidity: you had to construct your network architecture ahead of time.

This was in contradiction with the dynamic nature of the Python programming language, and also prevented researchers and ML developers from quickly iterating on their code.

Things have somewhat evolved with recent versions of TensorFlow, which introduces more dynamic mechanisms for creating and updating models. However it seems too late for TensorFlow: PyTorch has gained unaissalable momentum within the ML Research community.

One area where TensorFlow remains strong though is around deploying and running models in a production environment.

# Building Blocks

Let's cover some of the building blocks in PyTorch.

## Tensor

Everything starts with tensors in PyTorch.

> A tensor in PyTorch is a special data structure that represents a single or multi-dimensional collection of numbers. Tensors could be a vector (rank 1), a matrix (rank 2), or more advanced multi-dimensional arrays.

Tensors are used for the input, output, and parameters of a model.

The example below shows how to create a tensor in PyTorch
```
import torch

t = torch.tensor([
    [1,2,3]
    [4,5,6]
    [7,8,9]], dtype=torch.float32)

# the below prints "Torch.Size([3, 3])"
print(t.size())

# the below prints the rank of the tensor which is 2
print(len(t.shape))
```

You can perform a wide range of CPU/GPU optimised operations on tensors, such as multiplication, addition, clipping, etc.

By default, tensors are created on the CPU. However, it is much faster to perform usual tensor operations on GPUs than CPUs.
Therefore, if you have a GPU you move your tensors to this device and get massive performance boosts. The snippet below shows how to move your tensors to GPU:

```
# Check if GPU is available
if torch.cuda.is_available():
    t = t.to("cuda")
```

