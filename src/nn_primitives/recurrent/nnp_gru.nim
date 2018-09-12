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
  input: Tensor[T],
  W3, U3,
  bW3, bU3: Tensor[T],
  hidden: var Tensor[T]) =
  ## Input:
  ##   - input tensor of shape [batch_size, features]
  ##   - weight of input  W3 [3 * hidden_size, features]
  ##   - weight of hidden U3 [3 * hidden_size, hidden_size]
  ##   - biases of input and hidden state [1, 3 * hidden_size]
  ##
  ## Output (in-place):
  ##   - y == h'(t): The next hidden state of the GRU Cell.
  ##     (GRU output and next hidden state are the same)
  ##
  ## ⚠️ Input/Output updated in-place:
  ##   - h(t) -> h'(t), the hidden state of shape [batch_size, hidden_size]
  ##     is both an input and output
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

  # Step 4 - Update the hidden state
  apply3_inline(hidden, W3x[_, sz], n):
    (1 - y) * z + y * x

proc gru_cell_forward*[T: SomeReal](
  input,
  W3, U3,
  bW3, bU3: Tensor[T],
  r, z, n, Uh,
  hidden: var Tensor[T]
) =
  ## Input:
  ##   - input tensor of shape [batch_size, features]
  ##   - hidden state of shape [batch_size, hidden_size]
  ##   - gates weights of input W3 [3 * hidden_size, features]
  ##   - recurrent weights of hidden state U3 [3 * hidden_size, hidden_size]
  ##   - biases of input and hidden state [1, 3 * hidden_size]
  ##
  ## Output:
  ##   - r, z, n, Uh: intermediate tensors saved for backpropagation.
  ##     of shape [batch_size, hidden_size]
  ##   - y == h'(t): The next hidden state of the GRU Cell.
  ##     (GRU output and next hidden state are the same)
  ##
  ## ⚠️ Input/output updated in place:
  ##   - h(t) -> h'(t), the hidden state of shape [batch_size, hidden_size]
  ##     is both an input and output

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
  apply2_inline(Uh, U3h[_, s]):
    y

  # Step 2 - Computing reset (r) and update (z) gate
  apply3_inline(r, W3x[_, sr], U3h[_, sr]):
    sigmoid(y + z)

  apply3_inline(z, W3x[_, sz], U3h[_, sz]):
    sigmoid(y + z)

  # Step 3 - Computing candidate hidden state ñ
  # TODO: need apply4 / loopfusion for efficient
  # buffer passing in Stacked GRU implementation
  n = map3_inline(W3x[_, s], r, U3h[_, s]):
    tanh(x + y * z)

  # Step 4 - Update the hidden state
  apply3_inline(hidden, z, n):
    (1 - y) * z + y * x

proc gru_cell_backward*[T: SomeReal](
  dx, dh, dW3, dU3,          # input and weights gradients
  dbW3, dbU3: var Tensor[T], # bias gradient
  dnext: Tensor[T],          # gradient flowing back from the next hidden state
  x, h, W3, U3: Tensor[T],   # input parameters saved from forward
  r, z, n, Uh: Tensor[T]     # Intermediate tensors saved from forward
) =
  ## Input:
  ##   - dx, dh, dW3, dU3: respectively gradients of
  ##     - x, input tensor during the forward pass. Shape [batch_size, features]
  ##     - h, hidden state during the forward pass. Shape [batch_size, hidden_size]
  ##     - W3, gate input weights (multiplied by x) during the forward pass. Shape [3 * hidden_size, features]
  ##     - U3, recurrent weights (multiplied by h) during the forward pass. Shape [3 * hidden_size, features]
  ##   - dbW3 and dbU3: gradients of the biases for W3 and U3 weights
  ##   - dnext: gradient floowing back from the next layer
  ##   - x, h, W3, U3: inputs saved from the forward pass
  ##   - r, z, n, Uh: intermediate results saved from the forward pass of shape [batch_size, hidden_size]
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
  input: Tensor[T],
  W3s: openarray[Tensor[T]],
  U3s, bW3s, bU3s: Tensor[T],
  output, hidden: var Tensor[T]
) =
  ## Bidirectional support is not implemented
  ##
  ## Inputs:
  ##   - `input`: Input tensor of shape [sequence/timesteps, batch, features]
  ##   - An array of `num_stacked_layers` input weights `W3s` of shapes:
  ##       - [3 * hidden_size, features] for the first layer
  ##       - [3 * hidden_size, num_directions * hidden_size] for the following layers
  ##   - A series of hidden state weights `U3s` of shape [num_stacked_layers, 3 * hidden_size, hidden_size]
  ##   - A series of biases for input and hidden state weights of shape [num_stacked_layers, 1, 3 * hidden_size]
  ##
  ## Outputs:
  ##   - `output` of shape [sequence/timesteps, batch, num_directions * hidden_size].
  ##     `output` contains the output features `hiddenT` for each T (timesteps)
  ##   - `hidden` of shape [num_stacked_layers * num_directions, batch, hidden_size].
  ##     `hidden` contains the hidden state for timestep T == sequence/timesteps length of `input`
  ##
  ## ⚠️ Input/Output updated in-place:
  ##   - h(t) -> h'(t), the hidden state of shape [batch_size, hidden_size]
  ##     is both an input and output

  # 0. Retrieve the metadata and validate it
  let
    seq_len = input.shape[0]
    batch_size = input.shape[1]
    num_features = input.shape[2]
    hidden_size = hidden.shape[2]
    num_stacked_layers = W3s.len
    num_directions = hidden.shape[0] div num_stacked_layers

  doAssert hidden.shape == [num_stacked_layers * num_directions, batch_size, hidden_size]
  doAssert W3s[0].shape == [3 * hidden_size, num_features]
  for k in 1 ..< num_stacked_layers:
    doAssert W3s[k].shape == [3 * hidden_size, num_directions * hidden_size]
  doAssert U3s.shape == [num_stacked_layers, 3 * hidden_size, hidden_size]
  doAssert bW3s.shape == [num_stacked_layers, 1, 3 * hidden_size]
  doAssert bU3s.shape == bW3s.shape

  let directions = 1 # stub

  # Initialize output
  output = newTensorUninit[T](seq_len, batch_size, directions * hidden_size)

  block: # 1. Initial layer
    let
      W3l = W3s[0]
      U3l = U3s[0, _, _].squeeze(0)
      bW3l = bW3s[0, _, _].squeeze(0)
      bU3l = bU3s[0, _, _].squeeze(0)
    var hiddenl = hidden[0, _, _].squeeze(0)

    for timestep in 0 ..< seq_len:
      let input_ts = input[timestep, _, _].squeeze
      # TODO: reuse buffers
      gru_cell_inference(
        input_ts,
        W3l, U3l, bW3l, bU3l,
        hiddenl
      )
      output[timestep, _, _] = hiddenl.unsqueeze(0)

  # 2. Subsequent layers
  for layer in 1 ..< num_stacked_layers:
    let
      W3l = W3s[layer]
      U3l = U3s[layer, _, _].squeeze(0)
      bW3l = bW3s[layer, _, _].squeeze(0)
      bU3l = bU3s[layer, _, _].squeeze(0)
    var hiddenl = hidden[layer * directions, _, _].squeeze

    for timestep in 0 ..< seq_len:
      # TODO: reuse more than the output buffer
      let output_ts = output[timestep, _, _].squeeze
      gru_cell_inference(
        output_ts,
        W3l, U3l, bW3l, bU3l,
        hiddenl
      )
      output[timestep, _, _] = hiddenl.unsqueeze(0)

proc gru_forward*[T: SomeReal](
  input: Tensor[T],
  W3s: openarray[Tensor[T]],
  U3s, bW3s, bU3s: Tensor[T],
  rs, zs, ns, Uhs: var Tensor[T],
  output, hidden: var Tensor[T]
) =
  ## Bidirectional support is not implemented
  ##
  ## Inputs:
  ##   - `input`: Input tensor of shape [sequence/timesteps, batch, features]
  ##   - An array of `num_stacked_layers` input weights `W3s` of shapes:
  ##       - [3 * hidden_size, features] for the first layer
  ##       - [3 * hidden_size, num_directions * hidden_size] for the following layers
  ##   - A series of hidden state weights `U3s` of shape [num_stacked_layers, 3 * hidden_size, hidden_size]
  ##   - A series of biases for input and hidden state weights of shape [num_stacked_layers, 1, 3 * hidden_size]
  ##
  ## Outputs:
  ##   - rs, zs, ns, Uhs: intermediate tensors saved for backpropagation.
  ##     Shape [num_stacked_layers, batch_size, hidden_size]. They must be preallocated (but it can be with random values).
  ##   - `output` of shape [sequence/timesteps, batch, num_directions * hidden_size].
  ##     `output` contains the output features `hiddenT` for each T (timesteps)
  ##   - `hidden` of shape [num_stacked_layers * num_directions, batch, hidden_size].
  ##     `hidden` contains the hidden state for timestep T == sequence/timesteps length of `input`
  ##
  ## ⚠️ Input/Output updated in-place:
  ##   - h(t) -> h'(t), the hidden state of shape [batch_size, hidden_size]
  ##     is both an input and output

  # 0. Retrieve the metadata and validate it
  let
    seq_len = input.shape[0]
    batch_size = input.shape[1]
    num_features = input.shape[2]
    hidden_size = hidden.shape[2]
    num_stacked_layers = W3s.len
    num_directions = hidden.shape[0] div num_stacked_layers

  doAssert hidden.shape == [num_stacked_layers * num_directions, batch_size, hidden_size]
  doAssert W3s[0].shape == [3 * hidden_size, num_features]
  for k in 1 ..< num_stacked_layers:
    doAssert W3s[k].shape == [3 * hidden_size, num_directions * hidden_size]
  doAssert U3s.shape == [num_stacked_layers, 3 * hidden_size, hidden_size]
  doAssert bW3s.shape == [num_stacked_layers, 1, 3 * hidden_size]
  doAssert bU3s.shape == bW3s.shape

  doAssert rs.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert zs.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert ns.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert Uhs.shape == [num_stacked_layers, batch_size, hidden_size]

  let directions = 1 # stub

  # Initialize output
  output = newTensorUninit[T](seq_len, batch_size, directions * hidden_size)

  block: # 1. Initial layer
    let
      W3l = W3s[0]
      U3l = U3s[0, _, _].squeeze(0)
      bW3l = bW3s[0, _, _].squeeze(0)
      bU3l = bU3s[0, _, _].squeeze(0)
    var
      rl = rs[0, _, _].squeeze(0)
      zl = zs[0, _, _].squeeze(0)
      nl = ns[0, _, _].squeeze(0)
      Uhl = Uhs[0, _, _].squeeze(0)
      hiddenl = hidden[0, _, _].squeeze(0)

    # TODO: gru_cell_forward will detach `nl``
    # due to a missing apply4/loop-fusion operation
    var n_tmp = nl

    for timestep in 0 ..< seq_len:
      let input_ts = input[timestep, _, _].squeeze
      # TODO: reuse buffers
      gru_cell_forward(
        input_ts,
        W3l, U3l, bW3l, bU3l,
        rl, zl, n_tmp, Uhl,
        hiddenl
      )
      output[timestep, _, _] = hiddenl.unsqueeze(0)
      # TODO: apply/loop-fusion
      # copy n_tmpl back to nl
      apply2_inline(nl, n_tmp):
        y

  # 2. Subsequent layers
  for layer in 1 ..< num_stacked_layers:
    let
      W3l = W3s[layer]
      U3l = U3s[layer, _, _].squeeze(0)
      bW3l = bW3s[layer, _, _].squeeze(0)
      bU3l = bU3s[layer, _, _].squeeze(0)
    var
      rl = rs[layer, _, _].squeeze(0)
      zl = zs[layer, _, _].squeeze(0)
      nl = ns[layer, _, _].squeeze(0)
      Uhl = Uhs[layer, _, _].squeeze(0)
      hiddenl = hidden[layer, _, _].squeeze(0)

    # TODO: gru_cell_forward will detach `nl``
    # due to a missing apply4/loop-fusion operation
    var n_tmp = nl

    for timestep in 0 ..< seq_len:
      let output_ts = output[timestep, _, _].squeeze
      # TODO: reuse buffers
      gru_cell_forward(
        output_ts,
        W3l, U3l, bW3l, bU3l,
        rl, zl, n_tmp, Uhl,
        hiddenl
      )
      output[timestep, _, _] = hiddenl.unsqueeze(0)
      # TODO: apply/loop-fusion
      # copy n_tmpl back to nl
      apply2_inline(nl, n_tmp):
        y
