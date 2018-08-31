# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor,
  ../private/p_activation, ../nnp_linear,
  ../nnp_activation

# For compatibility with CuDNN and allow loading CPU/Cuda weights interchangeably,
# we use the following equations,
#
#   - h is hidden state at t-1, h' at t
#   - input == x, hidden == h
#   - n = h~ (the candidate hidden state)
#   - r is the reset gate
#   - z is the update gate
#   - h', the final output, is a linear interpolation
#
# r  =    σ(Wr * x + bWr +       Ur * h + bUr)
# z  =    σ(Wz * x + bWz +       Uz * h + bUz)
# n  = tanh(W  * x + bW  + r .* (U  * h + bU ))
# h' = (1 - z) .* n + z .* h
#
# Those differs from the original paper for n and h'
#   - The pointwise multiplication by r is after the matrix multiplication
#   - The linear interpolation has the terms switched

# TODO: after the 2 "linear" in forward prop and before the linear
#       in backprop, everything is elementwise
# we could use a giant loop-fusion to avoid intermediate tensors
#
# Note that the CPU prefetcher might not work as well, because
# between the use of U3h.data[i] and U3h.data[i+1]
# there will be a lot of intermediate computation.
#
# Also see here for counterarg: https://software.intel.com/en-us/forums/intel-moderncode-for-parallel-architectures/topic/635075
# Intel CPUs prefetcher can maintain 32 streams

proc gru_cell_inference*[T: SomeReal](
  input, hidden: Tensor[T],
  W3, U3,
  bW3, bU3: Tensor[T],
  next_hidden: var Tensor[T]) =
  ## Input:
  ##   - input tensor of shape [batch_size, features]
  ##   - hidden state of shape [batch_size, hidden_size]
  ##   - weight of input  W3 [3 * hidden_size, features]
  ##   - weight of hidden U3 [3 * hidden_size, hidden_size]
  ##   - biases of input and hidden state [1, 3 * hidden_size]
  ##
  ## Output:
  ##   - y == h'(t): The next hidden state of the GRU Cell.
  ##     (GRU output and next hidden state are the same)
  ##
  ## This is an optimized function when backpropagation is not needed.

  let
    H = hidden.shape[1]
    # Slices
    sr = (0 ..< H)|1
    sz = (H ..< 2*H)|1
    srz = (0 ..< 2*H)|1
    s = (2*H ..< 3*H)|1


  # Step 1 - U*h and W*x - Resulting shape [batch_size, 3*H]
  var W3x, U3h: Tensor[T] # TODO, pass those as parameter to allow buffer reuse

  linear(input, W3, bW3, W3x)
  linear(hidden, U3, bU3, U3h)

  # Step 2 - Computing reset (r) and update (z) gate
  var W2ru = W3x[_, srz] # shape [batch_size, 2*H] - we reuse the previous buffer
  apply2_inline(W2ru, U3h[_, srz]):
    sigmoid(x + y)

  # Step 3 - Computing candidate hidden state ñ
  var n = W3x[_, s] # shape [batch_size, H] - we reuse the previous buffer
  apply3_inline(n, W2ru[_, sr], U3h[_, s]):
    tanh(x + y * z)

  # Step 4 - Compute the next hidden state
  next_hidden = map3_inline(W3x[_, sz], n, hidden):
    (1 - x) * y + x * z

proc gru_cell_forward*[T: SomeReal](
  input, hidden,
  W3, U3,
  bW3, bU3: Tensor[T],
  r, z, n, Uh,
  next_hidden: var Tensor[T]
) =
  ## Input:
  ##   - input tensor of shape [batch_size, features]
  ##   - hidden state of shape [batch_size, hidden_size]
  ##   - weight of input  W3 [3 * hidden_size, features]
  ##   - weight of hidden U3 [3 * hidden_size, hidden_size]
  ##   - biases of input and hidden state [1, 3 * hidden_size]
  ##
  ## Output:
  ##   - r, z, n, Uh: intermediate tensors saved for backpropagation.
  ##     of size [batch_size, hidden_size]
  ##   - y == h'(t): The next hidden state of the GRU Cell.
  ##     (GRU output and next hidden state are the same)
  ##

  let
    H = hidden.shape[1]
    # Slices
    sr = (0 ..< H)|1
    sz = (H ..< 2*H)|1
    s = (2*H ..< 3*H)|1

  # Step 1 - U*h and W*x - Resulting shape [batch_size, 3*H]
  var W3x, U3h: Tensor[T] # TODO, pass those as parameter to allow buffer reuse

  linear(input, W3, bW3, W3x)
  linear(hidden, U3, bU3, U3h)

  # # Saving for backprop
  Uh = U3h[_, s].clone()

  # Step 2 - Computing reset (r) and update (z) gate
  apply3_inline(r, W3x[_, sr], U3h[_, sr]):
    sigmoid(y + z)

  apply3_inline(z, W3x[_, sz], U3h[_, sz]):
    sigmoid(y + z)

  # Step 3 - Computing candidate hidden state ñ
  n = map3_inline(W3x[_, s], r, U3h[_, s]):
    tanh(x + y * z)

  # Step 4 - Compute the next hidden state
  next_hidden = map3_inline(z, n, hidden):
    (1 - x) * y + x * z

proc gru_cell_backward*[T: SomeReal](
  dx, dh, dW3, dU3,          # input and weights gradients
  dbW3, dbU3: var Tensor[T], # bias gradient
  dnext: Tensor[T],          # gradient flowing back from the next hidden state
  x, h, W3, U3: Tensor[T],   # input parameters saved from forward
  r, z, n, Uh: Tensor[T]     # Intermediate tensors saved from forward
) =

  # Backprop of step 4 - z part
  let dz = (h - n) .* dnext
  let dn = (1.0 .- z) .* dnext

  # Backprop of step 3.
  let dWx = tanh_backward(dn, n)
  let dr = Uh .* dWx
  let dUh = r .* dWx

  # Backprop of step 2 - update gate z
  let dWzx = sigmoid_backward(dz, z)
  let dUzh = dWzx

  # Backprop of step 2 - reset gate r
  let dWrx = sigmoid_backward(dr, r)
  let dUrh = dWrx

  # Concat
  let dW3x = concat(dWrx, dWzx, dWx, axis = 1)
  let dU3h = concat(dUrh, dUzh, dUh, axis = 1)

  # Backprop of step 1
  linear_backward(x, W3, dW3x, dx, dW3, dbW3)
  linear_backward(h, U3, dU3h, dh, dU3, dbU3)

  # Backprop of step 4 - h part
  apply3_inline(dh, dnext, z):
    x + y * z

proc gru_inference*[T: SomeReal](
  input, hidden0: Tensor[T],
  W3s, U3s,
  bW3s, bU3s: Tensor[T],
  output, hiddenN: var Tensor[T]
) =
  ## Inputs:
  ##   - `input`: Input tensor of shape [sequence/timesteps, batch, features]
  ##   - `hidden0`: Initial hidden state for each element in the batch of shape
  ##     [num_stacked_layers * num_directions, batch, hidden_size]
  ##   - A series of input weights `W3s` of shape [num_stacked_layers, 3 * hidden_size, features]
  ##   - A series of hidden state weights `U3s` of shape [num_stacked_layers, 3 * hidden_size, hidden_size]
  ##   - A series of biases for input and hidden state weights of shape [num_stacked_layers, 1, 3 * hidden_size]
  ##
  ## Outputs:
  ##   - `output` of shape [sequence/timesteps, batch, num_directions * hidden_size].
  ##     `output` contains the output features `hiddenT` for each T (timesteps)
  ##   - `hiddenN` of shape [num_stacked_layers * num_directions, batch, hidden_size].
  ##     `hiddenN` contains the hidden state for timestep T == sequence/timesteps length of `input`

  # 0. Retrieve the metadata and validate it
  let
    seq_len = input.shape[0]
    batch_size = input.shape[1]
    num_features = input.shape[2]
    hidden_size = hidden0.shape[2]
    num_stacked_layers = W3s.shape[0]
    num_directions = hidden0.shape[0] div num_stacked_layers

  doAssert:
    hidden0.shape == [num_stacked_layers * num_directions, batch, hidden_size]
    hiddenN.shape == hidden0.shape
    W3s.shape == [num_stacked_layers, 3 * hidden_size, features]
    U3s.shape == [num_stacked_layers, 3 * hidden_size, hidden_size]
    bW3s.shape == [num_stacked_layers, 1, 3 * hidden_size]
    bU3s.shape == bW3s.shape

  # Preallocate work buffers
  var Wx = newTensorUninit[T](seq_len, batch_size, 3 * hidden_size) # Alloc across time for batch matmul
  var U3h: Tensor[T] # Time dependency so we re-use this buffer across time

  hiddenN = hidden0.clone()

  # TODO: directions?
  let direction = 1 # stub

  for layer in 0 ..< num_stacked_layers:
    let
      W3l = W3s[layer, _, _].squeeze
      U3l = U3s[layer, _, _].squeeze
      bU3l = bU3s[layer, _, _].squeeze

    # 1. Precompute Wx across all timesteps
    for timestep in 0 ..< seq_len:
      let input_ts = input[timestep, _, _].squeeze
      let W3x_ts = Wx[timestep, _, _].squeeze

      # Ideally we should use batch_matmul here but it's only implemented in Intel MKL
      # https://github.com/mratsim/Arraymancer/issues/101
      gemm(input_ts, W3l.transpose, W3x_ts)
    Wx .+= bW3s

    # 2. Timesteps-dependent computation
    for timestep in 0 ..< seq_len:
      let hidden_ts = hiddenN[layer * direction, _, _].squeeze
      let
        H = hidden_ts.shape[1]
        # Slices
        sr = (0 ..< H)|1
        sz = (H ..< 2*H)|1
        srz = (0 ..< 2*H)|1
        s = (2*H ..< 3*H)|1

      # 2.1 -- U*h, shape [batch_size, 3*H]
      linear(hidden_ts, U3l, bU3l, U3h)

      # Before that point, everything can be started in parallel with step 1.
      let W3x_ts = Wx[timestep, _, _].squeeze

      # 2.2 - Computing reset (r) and update (z) gate
      var W2ru = W3x_ts[_, srz] # shape [batch_size, 2*H] - we reuse the previous buffer
      apply2_inline(W2ru, U3h[_, srz]):
        sigmoid(x + y)

      # Step 3 - Computing candidate hidden state ñ
      var n = W3x_ts[_, s] # shape [batch_size, H] - we reuse the previous buffer
      apply3_inline(n, W2ru[_, sr], U3h[_, s]):
        tanh(x + y * z)

      # Step 4 - Compute the next hidden state
      apply3_inline(hidden_ts, W3x[_, sz], n):
        (1 - y) * z + y * x
