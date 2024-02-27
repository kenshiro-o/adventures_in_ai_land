**Loss Function**: Think of it as as way to measure how far off from the actual results our model's predictions are. Popular loss functions include [_mean squared error_](https://en.wikipedia.org/wiki/Mean_squared_error) or the [_negative log likelihood_](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/). Sometimes the loss function is also referred to as the **cost function** or the **objective**.

**Regularization**: Regularization helps prevent our model from overfitting by introducing additional penalty terms to the loss function that are solely dependant on the weights (and sometimes the bias too). Additionally regularization helps us constrain the model so that the weights remain as small as possible.

**Softmax**: Softmax is a _classifier_ (not a loss function!).  It computes the pseudo-probabilities for each class, which allows us to interpret the model's confidence for each predicted outcome. Assuming K classes, and an input x_i, the Softmax for the predicted output z_i given x_i is calculated as follows:
$$
z_i = f(x_i)
$$

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Optimization**: Optimization is the process of finding the set of parameters that minimize the loss function.
