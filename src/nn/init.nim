# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import
  ../tensor/tensor,
  math

# ############################################################
#
#                         Common helpers
#
# ############################################################

# TODO: random seed
# TODO: Test variance and mean

type FanMode = enum
  FanAvg,
  FanIn
  FanOut

type Distribution = enum
  Uniform
  Normal
  # TODO truncated_normal
  #   - https://github.com/keras-team/keras/issues/8048
  #   - https://github.com/tensorflow/tensorflow/issues/18706

func compute_fans(shape: varargs[int]): tuple[fan_in, fan_out: int] =
  # Definitions:
  #   - fan_in: the numbers of inputs a logic gate can handle
  #   - fan_out: the numbers of output a logic gate can handle
  #     ⚠️ Warning - this is the number of inputs in the backward pass
  #
  # Returns the (fan_in, fan_out) pair for the input layer shape.
  #
  # Input:
  #   - A shape [features_out, features_in] for a linear layer
  #   - or a shape [C_out, C_in, kernel_height, kernel_width] for a convolution 2D layer

  assert shape.len in {2..5}, "Only Linear and Convolutions are supported"

  result.fan_out = shape[0]
  result.fan_in = shape[1]

  # Linear
  if shape.len == 2:
    return
  # Convolution
  let receptive_field_size = block:
    var product = 1
    for i in 2 ..< shape.len:
      product *= shape[i]
    product
  result.fan_out *= receptive_field_size
  result.fan_in *= receptive_field_size

proc variance_scaled(
        shape: varargs[int],
        T: type,
        scale: static[T] = 1,
        mode: static FanMode = FanIn,
        distribution: static Distribution = Normal
      ): Tensor[T] =
  let (fan_in, fan_out{.used.}) = shape.compute_fans
  when mode == FanIn:
    let std = sqrt(scale / fan_in.T)
  elif mode == FanOut:
    let std = sqrt(scale / fan_out.T)
  else:
    let std = sqrt(scale * 2.T / T(fan_in + fan_out))

  when distribution == Uniform:
    let limit = sqrt(3.T) * std
    result = randomTensor(shape, -limit .. limit)
  else:
    result = randomNormalTensor(shape, 0'f32, std)

# ############################################################
#
#                  Kaiming He initialisations
#
# ############################################################

# Initialisations from
# Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
#     2015, He et al, https://arxiv.org/abs/1502.01852

proc kaiming_uniform*(shape: varargs[int], T: type): Tensor[T] =
  ## Kaiming He initialisation for trainable layers
  ## preceding a ReLU activation.
  ## Kaiming initialization is recommended for relu activated layers.
  ##
  ## Weight is sampled from an uniform distribution
  ## of range [-√3 * √(2/fan_in), √3 * √(2/fan_in)]
  ## with fan_in the number of input unit in the forward pass.
  ##
  ## This preserves the magnitude of the variance of the weight
  ## during the forward pass
  ##
  ## Paper:
  ##   - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
  ##     ImageNet Classification](http://arxiv.org/abs/1502.01852)
  result = variance_scaled(shape, T, scale = 2.T, mode = FanIn, distribution = Uniform)

proc kaiming_normal*(shape: varargs[int], T: type): Tensor[T] =
  ## Kaiming He initialisation for trainable layers
  ## preceding a ReLU activation.
  ## Kaiming initialization is recommended for relu activated layers.
  ##
  ## Weight is sampled from a normal distribution
  ## of mean 0 and standard deviation √(2/fan_in)
  ## with fan_in the number of input unit in the forward pass.
  ##
  ## This preserves the magnitude of the variance of the weight
  ## during the forward pass
  ##
  ## Paper:
  ##   - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
  ##     ImageNet Classification](http://arxiv.org/abs/1502.01852)
  result = variance_scaled(shape, T, scale = 2.T, mode = FanIn, distribution = Normal)

# ############################################################
#
#                  Xavier Glorot initialisations
#
# ############################################################

# Initialisations from
# Understanding the difficulty of training deep feedforward neural networks
#     2010, Glorot et al, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

proc xavier_uniform*(shape: varargs[int], T: type): Tensor[T] =
  ## Xavier Glorot initialisation for trainable layers
  ## preceding a linear activation (sigmoid, tanh).
  ## Xavier initialization is recommended for sigmoid, tanh
  ## and softsign activated layers.
  ##
  ## Weight is sampled from an uniform distribution
  ## of range [-√3 * √(2/(fan_in+fan_out)), √3 * √(2/(fan_in+fan_out))]
  ## with fan_in the number of input units in the forward pass.
  ## and fan_out the number of input units during the backward pass
  ## (and not output units during the forward pass).
  ##
  ##
  ## This provides a balance between preserving
  ## the magnitudes of the variance of the weight during the forward pass,
  ## and the backward pass.
  ##
  ## Paper:
  ##   - [Understanding the difficulty of training deep feedforward neural
  ##     networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  result = variance_scaled(shape, T, scale = 1.T, mode = FanAvg, distribution = Uniform)

proc xavier_normal*(shape: varargs[int], T: type): Tensor[T] =
  ## Xavier Glorot initialisation for trainable layers
  ## preceding a linear activation (sigmoid, tanh).
  ## Xavier initialization is recommended for sigmoid, tanh
  ## and softsign activated layers.
  ##
  ## Weight is sampled from a normal distribution
  ## of mean 0 and standard deviation √(2/(fan_in+fan_out))
  ## with fan_in the number of input units in the forward pass.
  ## and fan_out the number of input units during the backward pass
  ## (and not output units during the forward pass).
  ##
  ##
  ## This provides a balance between preserving
  ## the magnitudes of the variance of the weight during the forward pass,
  ## and the backward pass.
  ##
  ## Paper:
  ##   - [Understanding the difficulty of training deep feedforward neural
  ##     networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  result = variance_scaled(shape, T, scale = 2.T, mode = FanAvg, distribution = Normal)

# ############################################################
#
#                      Lecun initialisation
#
# ############################################################

# Initialisations from
# Efficient Backprop
#     1998, Lecun et al, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
#
# Self-Normalizing Neural Networks
#     2017, Klambauer et al, https://arxiv.org/abs/1706.02515

proc yann_uniform*(shape: varargs[int], T: type): Tensor[T] =
  ## Yann Lecun initialisation for trainable layers
  ##
  ## Weight is sampled from an uniform distribution
  ## of range [√(3/fan_in), √(3/fan_in)]
  ## with fan_in the number of input unit in the forward pass.
  ##
  ## This preserves the magnitude of the variance of the weight
  ## during the forward pass
  ##
  ## Paper:
  ##   - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  result = variance_scaled(shape, T, scale = 1.T, mode = FanIn, distribution = Uniform)

proc yann_normal*(shape: varargs[int], T: type): Tensor[T] =
  ## Yann Lecun initialisation for trainable layers
  ##
  ## Weight is sampled from a normal distribution
  ## of mean 0 and standard deviation √(1/fan_in)
  ## with fan_in the number of input unit in the forward pass.
  ##
  ## This preserves the magnitude of the variance of the weight
  ## during the forward pass
  ##
  ## Paper:
  ##   - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  result = variance_scaled(shape, T, scale = 1.T, mode = FanIn, distribution = Normal)

# ############################################################
#
#                    Orthogonal initialisation
#
# ############################################################

# Initialisations from
# Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
#     2013, Saxe et al, https://arxiv.org/abs/1312.6120
