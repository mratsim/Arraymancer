# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor,
  ../private/p_activation, ../nnp_linear

proc gru_cell_inference*[T: SomeReal](
  input, hidden,
  W3, U3,
  bW3, bU3: Tensor[T],
  result: var Tensor[T]) =

  ## Input:
  ##   - input tensor of shape [batch_size, features]
  ##   - hidden state of shape [batch_size, hidden_size]
  ##   - weight of input  [3 * hidden_size, features]
  ##   - weight of hidden [3 * hidden_size, hidden_size]
  ##   - biases of input and hidden state [1, 3 * hidden_size]
  ##
  ## Output:
  ##   - y == h'(t): The next hidden state of the GRU Cell.
  ##     (GRU output and next hidden state are the same)
  ##
  ## This is an optimized function when backpropagation is not needed.

  # For compatibility with CuDNN and allow loading CPU/Cuda weights interchangeably,
  # we use the following equations,
  #
  #   - h is hidden state at t-1, h' at t
  #   - input == x, hidden == h
  #   - n = h~ (the candidate hidden state)
  #   - r is the reset gate
  #   - z is the update gate
  #   - h' is a linear interpolation
  #   - w_input  == W, the concatenation of [Wr, Wz, Wn]
  #   - w_recur  == R, the concatenation of [Rr, Rz, Rn]
  #   - bW and bR are the corresponding bias
  #   - R is called U in the original paper
  #
  # r  =    σ(Wr * x + bWr +       Rr * h + bRr)
  # z  =    σ(Wz * x + bWz +       Rz * h + bRz)
  # n  = tanh(Wn * x + bWn + r .* (Rn * h + bRn))
  # h' = (1 - z) .* n + z .* h
  #
  # Those differs from the original paper for n and h'
  #   - The pointwise multiplication by r is after the matrix multiplication
  #   - The linear interpolation has the terms switched

  let
    H = hidden.shape[1]
    # Slices
    sr = (0 ..< H)|1
    sz = (H ..< 2*H)|1
    srz = (0 ..< 2*H)|1
    sn = (2*H ..< 3*H)|1

  var W3x, U3h: Tensor[T] # TODO, pass those as parameter to allow buffer reuse
  # Resulting shape [batch_size, 3*H]
  linear(input, W3, bW3, W3x)
  linear(hidden, U3, bU3, U3h)

  # To reduce allocations, we compute reset gate r
  # and update gate z in the previous buffers
  # We keep them concatenated to improve throughput
  var W2ru = W3x[_, srz] # shape [batch_size, 2*H]
  apply2_inline(W2ru, U3h[_, srz]):
    sigmoid(x + y)

  # We also reuse the previous buffer for the candidate hidden state n
  # shape [H, batch_size]
  var n = W3x[_, sn] # shape [batch_size, H]
  apply3_inline(n, W2ru[_, sr], U3h[_, sn]):
    tanh(x + y * z)

  # Compute the next hidden state
  result = map3_inline(W3x[_, sz], n, hidden):
    (1 - x) * y + x * z
