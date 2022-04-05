# Handling long sequences

Why nonsaturating activation functions (e.g., RELU) may lead the RNN to be even more unstable during training? How to prevent that?

A nonsaturating function does not prevent that outputs explode.
Suppose Gradient Descent updates the weights in a way that increases the outputs slightly at the first time step. At the 2nd time step, the outputs may also be slightly increased (because the same weights are used every time step), and so on and so forth.

You can reduce the risk by using a smaller learning rate or using a saturating activation function (e.g., hyperbolic tangent — the default).
However the gradients may also explode.
If training is unstable:

- Monitor the size of the gradients (e.g., TensorBoard)
- Use gradient clipping
