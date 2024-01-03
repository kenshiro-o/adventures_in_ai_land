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

### Useful Operations

`tensor.squeeze()`: Removes dimensions of length 1 from a N-rank vector. Essentially all dimensions of 1 get _squeezed_ out of the resulting tensor (e.g. [28, 28, 1] tensor gets squeezed into a [28, 28] tensor).

`tensor.unsqueeze(d): Performs the opposite of `squeeze` by adding an extra dimension at the specified index. In most cases `d=0`. This operation is useful when performing inference on single items converted to tensors because the network expects the input tensor's first dimension to be the number of items in the batch.
So for a tensor `t` with shape (3, 64, 64), doing the operation `t_unsqueezed = tensor.unsqueeze(0)` will create a tensor of shape `(1, 3, 64, 64)`.

`tensor.view(-1, n)`: Returns a new, reshaped tensor of n columns. The `-1` let's Pytorch determine the appropriate number of rows. For example, if x originally has a shape of (4, 4, 20), which totals 320 elements, using x.view(-1, 320) will reshape it into a tensor of shape (1, 320) – a 2D tensor with 1 row and 320 columns.

`tensor.permute(...)`: This operation is often used to change the order of the dimensions in a PyTorch tensor. PyTorch uses a channel-first approach (e.g. `[batch_size, channels, height, width]` for images), but other libraries like numpy expect a _channel-last_ input (e.g. `[batch_size, height, width, channels]`).
Assuming we have a rank 4 tensor of an image, we may write the code `tensor.permute(0, 2, 3, 1).numpy()` to re-order the dimensions of the tensor and then convert it to a numpy array. We can then use show the image using matplotlib for instance.

## Dataset

A `torch.utils.data.Dataset` stores information about a given dataset. By dataset we mean both the data itself and the labels from the dataset.

PyTorch comes with a set of pre-defined datasets, but you can also create your own `Dataset` class if needed.


## DataLoader

A `torch.utils.data.DataLoader` wraps a `Dataset` to allow iteration over it.

This enables us to decouple the data itself (i.e. `Dataset`) from the way we access it (i.e. `DataLoader`)

⚠️ We therefore feed a `DataLoader` to a model and not a `Dataset`.

The example code below shows how we can load a `Dataset`:

```
torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
```

Notice that we must specify a `Dataset` and `batch_size` . We can optionally specify whether we would like the data to be shuffled.


# Building a Neural Network From Scratch

## Enabling Gradient Descent

We need to tell Pytorch to "record" forward passes on our tensors so that it can automatically perform differentiation and thus find the gradients that we will use to adjustour weight.

You can do so with the `requires_grad_()` method on a tensor. Remember that in PyTorch appending the `_` suffix to a method means that the operation is done in place (i.e. the internals of the object are modified and no new object is returned)

```
weights.requires_grad_()
```

To get the gradients, you must first:
1. compute the loss
2. call `backward()` on the loss to compute the gradient of all concerned variables
3. access the gradients of the weights from the `.grad` field

Example below:
```
loss = compute_loss(weights, inputs, targets)
loss.backward()
weights.grad
```

⚠️ Make sure to that when you're adjusting the weights by the gradient (and learning rate) PyTorch is not trying to perform differentiation. At this stage we are not doing a forward pass but instead re-adjusting our model so that it gets closer to the target in the next forward pass.

Basically, the part where you adjust the weights should look like this...
```
loss = compute_loss(weights, inputs, targets)
loss.backward()

# ensure Pytorch is not trying to perform auto-differentiation in the block below
with torch.no_grad():
    # adjust the weights - do this in-place thanks to the "_" suffix of sub_(...)
    weights.sub_(weights.grad * learning_rate)

    # reset the gradient - also in-place
    weights.grad.zero_()
```

