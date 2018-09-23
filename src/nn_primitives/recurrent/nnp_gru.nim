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
  ##   - dnext: gradient flowing back from the next layer
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

  # Backprop of step 1 - TODO this detaches gradients if they are slices
  linear_backward(x, W3, dW3x, dx, dW3, dbW3)
  linear_backward(h, U3, dU3h, dh, dU3, dbU3)

  # Backprop of step 4 - h part
  apply3_inline(dh, dnext, z):
    x + y * z

proc gru_inference*[Layers: static[int], T: SomeReal](
  input: Tensor[T],
  W3s: array[Layers, Tensor[T]],
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
    num_directions = hidden.shape[0] div num_stacked_layers # Always 1 at the moment

  doAssert hidden.shape == [num_stacked_layers * num_directions, batch_size, hidden_size]
  doAssert W3s[0].shape == [3 * hidden_size, num_features]
  for k in 1 ..< num_stacked_layers:
    doAssert W3s[k].shape == [3 * hidden_size, num_directions * hidden_size]
  doAssert U3s.shape == [num_stacked_layers, 3 * hidden_size, hidden_size]
  doAssert bW3s.shape == [num_stacked_layers, 1, 3 * hidden_size]
  doAssert bU3s.shape == bW3s.shape

  # Initialize output
  output = newTensorUninit[T](seq_len, batch_size, num_directions * hidden_size)

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
    var hiddenl = hidden[layer * num_directions, _, _].squeeze

    for timestep in 0 ..< seq_len:
      # TODO: reuse more than the output buffer
      let output_ts = output[timestep, _, _].squeeze
      gru_cell_inference(
        output_ts,
        W3l, U3l, bW3l, bU3l,
        hiddenl
      )
      output[timestep, _, _] = hiddenl.unsqueeze(0)

proc gru_forward*[Layers, Timesteps: static[int], T: SomeReal](
  input: Tensor[T],
  W3s: array[Layers, Tensor[T]],
  U3s, bW3s, bU3s: Tensor[T],
  rs, zs, ns, Uhs: var Tensor[T],
  output, hidden: var Tensor[T],
  cached_inputs: var array[Layers, Tensor[T]],
  cached_hiddens: var array[Layers, array[Timesteps, Tensor[T]]]
) =
  ## ⚠️ API subject to change to match CuDNNs
  ##
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
  ##   - `cached_inputs`, a sequence of length num_stacked_layers containing
  ##     - the first layer input of shape [sequence/timesteps, batch, features]
  ##     - the following layer inputs of shape [sequence/timesteps, batch, num_directions * hidden_size]
  ##   - `cached_hiddens`, a sequence of sequences of length [num_stacked_layers, sequence/timesteps]
  ##     - containing all intermediate hidden states for each timesteps for each stacked layers.
  ##       Hidden states are of tensors of shape [3 * hidden_size, hidden_size]
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

  # doAssert cached_inputs.len == num_stacked_layers
  # doAssert cached_hidden.len == num_stacked_layers
  # doAssert num_stacked_layers[0].len == seq_len

  let directions = 1 # stub

  # Initialize output
  output = newTensorUninit[T](seq_len, batch_size, directions * hidden_size)

  for layer in 0 ..< num_stacked_layers:
    if layer == 0:
      cached_inputs[0] = input
    else:
      cached_inputs[layer] = output.clone()
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
      cached_hiddens[layer][timestep] = hiddenl.clone()
      let input_ts = block:
            if layer == 0:
              input[timestep, _, _].squeeze(0)
            else:
              output[timestep, _, _].squeeze(0)
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

proc gru_backward*[Layers, Timesteps: static[int], T: SomeReal](
  dinput, dhidden: var Tensor[T],                             # Input and starting hidden state gradient
  dW3s: var array[Layers, Tensor[T]],                         # Weight tensor
  dU3s, dbW3s, dbU3s: var Tensor[T],                          # Weights & biases gradients
  doutput: Tensor[T],                                         # Gradient flowing back from the output/next hidden state
  cached_inputs: array[Layers, Tensor[T]],                    # Input params saved from forward
  cached_hiddens: array[Layers, array[Timesteps, Tensor[T]]], # Input params saved from forward
  W3s: array[Layers, Tensor[T]], U3s,                         # Input params saved from forward
  rs, zs, ns, Uhs: Tensor[T]                                  # Intermediate tensors saved from forward
) =
  ## ⚠️ API subject to change to match CuDNNs

  ## Outputs:
  ##   - dinput, dhidden, dW3s, dU3s:
  ##     Gradient tensors, will hold the results corresponding to the respective gradients of:
  ##     - `input`: Input tensor during the forward pass of shape [sequence/timesteps, batch, features]
  ##     - `hidden`: Hidden states during the forward pass of shape [num_stacked_layers * num_directions, batch, hidden_size]
  ##     - `W3s`: An array of `num_stacked_layers` input weights of shapes:
  ##       - [3 * hidden_size, features] for the first layer
  ##       - [3 * hidden_size, num_directions * hidden_size] for the following layers
  ##     - `U3s`: A series of hidden state weights of shape [num_stacked_layers, 3 * hidden_size, hidden_size]
  ##   - dbW3s and dbU3s: gradients of the biases. Shape [num_stacked_layers, 1, 3 * hidden_size]
  ##
  ## Inputs:
  ##   - doutput: gradient flowing back from the next layer.
  ##     Shape: [sequence/timesteps, batch, num_directions * hidden_size]
  ##   - cached_inputs, cached_hiddens, W3s, U3s: saved from the forward pass
  ##   - rs, zs, ns, Uhs: intermediate results saved from the forward pass
  ##     Shape [num_stacked_layers, batch_size, hidden_size]

  # 0. Retrieve the metadata and validate it
  let
    seq_len = cached_inputs[0].shape[0]
    batch_size = cached_inputs[0].shape[1]
    num_features = cached_inputs[0].shape[2]
    hidden_size = cached_hiddens[0][0].shape[1]
    num_stacked_layers = int(Layers) # Workaround for Nim 0.18.0. Not needed in devel
    num_directions = 1 # stub

  doAssert W3s[0].shape == [3 * hidden_size, num_features]
  for k in 1 ..< num_stacked_layers:
    doAssert W3s[k].shape == [3 * hidden_size, num_directions * hidden_size]
  doAssert U3s.shape == [num_stacked_layers, 3 * hidden_size, hidden_size]

  doAssert rs.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert zs.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert ns.shape == [num_stacked_layers, batch_size, hidden_size]
  doAssert Uhs.shape == [num_stacked_layers, batch_size, hidden_size]

  # 1. Preallocate the results (TODO: separate alloc from compute so that users can pass buffers)
  dhidden = newTensorUninit[T](num_stacked_layers, batch_size, hidden_size)
  for i, t in dW3s.mpairs:
    t = zeros_like(W3s[i])
  dU3s = zeros_like(U3s)
  dbW3s = zeros[T]([num_stacked_layers, 1, 3 * hidden_size])
  dbU3s = zeros[T]([num_stacked_layers, 1, 3 * hidden_size])

  # 2. Proceed from last layer to initial layer
  var gFlowBack = doutput # gradient flowing back
  dInput = newTensorUninit[T](seq_len, batch_size, num_features)

  for layer in countdown(num_stacked_layers - 1, 0):
    let
      W3l = W3s[layer]
      U3l = U3s[layer, _, _].squeeze(0)

      rl = rs[layer, _, _].squeeze(0)
      zl = zs[layer, _, _].squeeze(0)
      nl = ns[layer, _, _].squeeze(0)
      Uhl = Uhs[layer, _, _].squeeze(0)

      inputl = cached_inputs[layer]

    var dht1 = zeros[T](batch_size, hidden_size)
    var dht: Tensor[T]

    for timestep in countdown(seq_len - 1, 0):
      let input_lts = inputl[timestep, _, _].squeeze(0)
      let hidden_lts = cached_hiddens[layer][timestep]
      var gFlowBack_ts = gFlowBack[timestep, _, _].squeeze(0)

      # gradients of hidden state and hidden state (t+1)
      var dx: Tensor[T]
      dht1 += gFlowBack_ts

      # Contribution of weights for this timestep
      var dW3s_lts, dU3s_lts, dbW3s_lts, dbU3s_lts: Tensor[T]

      gru_cell_backward(
        dx, dht, dW3s_lts, dU3s_lts,
        dbW3s_lts, dbU3s_lts,
        dht1,
        input_lts, hidden_lts, W3l, U3l,
        rl, zl, nl, Uhl
      )

      # Update gradient flowing back at timestep to pass to next layer
      if layer != 0:
        apply2_inline(gFlowBack_ts, dx): y
      else:
        dInput[timestep, _, _] = dx.unsqueeze(0)
      if timestep != 0:
        dht1 = dht

      # Accumulate the contribution of weights
      dW3s[layer] = dW3s_lts
      dU3s[layer, _, _] = dU3s_lts.unsqueeze(0)
      dbW3s[layer, _, _] = dbW3s_lts.unsqueeze(0)
      dbU3s[layer, _, _] = dbU3s_lts.unsqueeze(0)

    dhidden[layer, _, _] = dht.unsqueeze(0)
