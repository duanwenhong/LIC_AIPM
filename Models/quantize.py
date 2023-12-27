import torch
from torch.autograd import Function
from jax import numpy as jnp

def quantize(x, is_training, offset=0):
    if is_training:
        y = QuantizeFunction.apply(x)
    else:
        y = torch.round(x - offset) + offset

    return y
# temperature default 1/7 it is corresponding to the 1/alpha in paper.
def soft_round(x, temperature):

  if temperature < 1e-4:
    return jnp.around(x)
  if temperature > 1e4:
    return x
  m = jnp.floor(x) + .5
  z = 2 * jnp.tanh(.5 / temperature)
  r = jnp.tanh((x - m) / temperature) / z
  return m + r

def quantize_mod(x, is_training):
  if is_training:
    y = soft_round(x, 1/7)
  else:
    y = torch.round(x)
  return y

class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, inputs):
        noise = torch.zeros(1, dtype=inputs.dtype, device=inputs.device).uniform_(-0.5, 0.5)
        #another method different from directly add the noise
        outputs = torch.round(inputs + noise) - noise

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
