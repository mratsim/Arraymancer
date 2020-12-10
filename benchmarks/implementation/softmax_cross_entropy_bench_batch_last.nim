# Copyright 2017 the Arraymancer contributors
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


import  ../../src/tensor/tensor,
        math, times

proc streaming_max_sumexp[T](t: Tensor[T]): tuple[max:T, sumexp: T] {.noSideEffect, inline.}=
  # One pass but with branching
  result.max = -Inf.T   # will store the streaming max of the tensor
  result.sumexp = 0.T   # will store the streaming sum of exp of the tensor

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp *= exp(result.max - x)
      result.sumexp += 1
      result.max = x

proc classic_max_sumexp[T](t: Tensor[T]): tuple[max:T, sumexp: T] =
  # 2 pass no branching

  result.max = t.max
  result.sumexp = t.fold_inline() do:
    x = exp(y - result.max)
  do:
    x += exp(y - result.max)
  do: x += y

proc streaming_max_sumexp[T](t: Tensor[T], axis: int): Tensor[tuple[max:T, sumexp: T]] {.noInit.}=
  # Only 2D tensor are supported for now. (i.e. no 3D Softmax)
  assert axis in {0, 1}

  result = newTensorUninit[tuple[max:T, sumexp: T]](t.shape[axis])

  for i in `||`(0, t.shape[axis]-1, "simd"):
    result.unsafe_raw_buf[i] = t.atAxisIndex(axis, i).streaming_max_sumexp

  # Reexpand the tensor to be consistent with fold_axis/reduce_axis
  if axis == 0:
    result = result.unsqueeze(1)
  else:
    result = result.unsqueeze(0)

proc classic_max_sumexp[T](t: Tensor[T], axis: int): Tensor[tuple[max:T, sumexp: T]] {.noInit.}=
  # Only 2D tensor are supported for now. (i.e. no 3D Softmax)
  assert axis in {0, 1}

  result = newTensorUninit[tuple[max:T, sumexp: T]](t.shape[axis])

  for i in `||`(0, t.shape[axis]-1, "simd"):
    result.unsafe_raw_buf[i] = t.atAxisIndex(axis, i).classic_max_sumexp

  # Reexpand the tensor to be consistent with fold_axis/reduce_axis
  if axis == 0:
    result = result.unsqueeze(1)
  else:
    result = result.unsqueeze(0)


proc streaming_logsumexp[T: SomeFloat](t: Tensor[T]): T =
  # 1 pass but branching
  let (max, sumexp) = t.streaming_max_sumexp
  result = max + ln(sumexp)

proc classic_logsumexp[T: SomeFloat](t: Tensor[T]): T =
  # 2 pass but no branching
  let max = t.max # first loop over data

  result = t.fold_inline() do: # second loop over data
    # Init first element
    x = exp(y - max)
  do:
    # Process next elements
    x += exp(y - max)
  do:
    # Merge the partial folds
    x += y

  result = max + ln(result)

proc classic_logsumexp_v2[T: SomeFloat](t: Tensor[T]): T =
  # 2 pass no branching
  let (max, sumexp) = t.classic_max_sumexp
  result = max + ln(sumexp)

proc softmax_cross_entropy1*[T](input, target: Tensor[T]): T =
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])
  var i = 0
  for sample_input, sample_target in zipAxis(input, target, 1):
    let lse = sample_input.classic_logsumexp_v2 # y.streaming_log_sumexp
    sample_softmax_xentropy[0, i] = sum:
      map2_inline(sample_input, sample_target):
        y * (lse - x)
    inc i
  result = sample_softmax_xentropy.mean

proc frobenius_inner_prod*[T](a,b: Tensor[T]): T =
  sum:
    map2_inline(a,b):
      x * y

proc softmax_cross_entropy2[T](input, target: Tensor[T]): T =
  result = frobenius_inner_prod(input, target)

  let sum_logsumexp = fold_axis_inline(input, T, fold_axis=1) do:
    x = y.streaming_log_sumexp
  do:
    x += y.streaming_log_sumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(input.shape[1])

####### Sparse
proc sparse_softmax_cross_entropy1[T](input: Tensor[T], target: Tensor[int]): T =
  for i in 0||(input.shape[1]-1):
    let lse = input[_,i].streaming_log_sumexp
    when not declared(openmp):
      result += lse - input[target[i], i]
    else:
      let tmp = lse - input[target[i], i]
      {.emit:"#pragma omp atomic".}
      {.emit:"`result` += `tmp`;".}
  result /= T(input.shape[1])

proc sparse_softmax_cross_entropy2[T](input: Tensor[T], target: Tensor[int]): T =
  let batch_size = input.shape[1]

  for i in 0||(batch_size-1):
    # Unfortunately we can't use `result` in a parallel for reduction declaration so we need atomic
    when not declared(openmp):
      result = input[target[i], i]
    else:
      let tmp = input[target[i], i]
      {.emit:"#pragma omp atomic".}
      {.emit:"`result` += `tmp`;".}

  let sum_logsumexp = fold_axis_inline(input, T, fold_axis=1) do:
    x = y.streaming_log_sumexp
  do:
    x += y.streaming_log_sumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(batch_size)

###### Backprop bench
proc stable_softmax[T](x, max, sumexp: T): T {.noSideEffect, inline.}=
  # Numerically stable streaming softmax helper
  result = exp(x - max) / sumexp

proc softmax_cross_entropy_backward1[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  let batch_size = cached_tensor.shape[1]

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  result = zeros_like(cached_tensor)

  for i in 0||(batch_size-1): # Can't use OpenMP - SIGSEGV Illegal Address
    let (max, sumexp) = cached_tensor[_,i].streaming_max_sumexp

    var res_slice = result[_, i]

    apply3_inline(res_slice, cached_tensor[_,i], target[_,i]):
      grad * (stable_softmax(y, max, sumexp) - z) / T(batch_size)

proc sparse_softmax_cross_entropy_backward1[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[int]
        ): Tensor[T] {.noInit.}=

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  let batch_size = cached_tensor.shape[1]

  result = zeros_like(cached_tensor)

  for i, truth_idx in enumerate(target):
    result[truth_idx, i] = -1

  for i in 0||(batch_size-1):
    let (max, sumexp) = cached_tensor[_,i].streaming_max_sumexp

    var res_slice = result[_,i]

    apply2_inline(res_slice, cached_tensor[_,i]):
      grad * (stable_softmax(y, max, sumexp) + x) / T(batch_size)

############ New optimized
proc softmax_cross_entropy_backward2[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  let batch_size = cached_tensor.shape[1]

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  let axis_max_sumexp = cached_tensor.streaming_max_sumexp(axis = 1).broadcast(cached_tensor.shape)
  # let axis_max_sumexp = cached_tensor.classic_max_sumexp(axis = 1).broadcast(cached_tensor.shape)

  result = map3_inline(cached_tensor, target, axis_max_sumexp):
      grad * (stable_softmax(x, z.max, z.sumexp) - y) / T(batch_size)


proc sparse_softmax_cross_entropy_backward2[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[int]
        ): Tensor[T] {.noInit.}=
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  let batch_size = cached_tensor.shape[1]

  result = zeros_like(cached_tensor)
  for i, truth_idx in enumerate(target):
    result[truth_idx, i] = -1

  let axis_max_sumexp = cached_tensor.streaming_max_sumexp(axis = 1).broadcast(cached_tensor.shape)
  # let axis_max_sumexp = cached_tensor.classic_max_sumexp(axis = 1).broadcast(cached_tensor.shape)


  apply3_inline(result, cached_tensor, axis_max_sumexp):
      grad * (stable_softmax(y, z.max, z.sumexp) + x) / T(batch_size)


####### Bench:

let batch_size = 200
let nb_classes = 100000

# Create a sparse label tensor of shape: [batch_size]
let sparse_labels = randomTensor(batch_size, nb_classes)

# Create the corresponding dense label tensor of shape [nb_classes, batch_size]
var labels = zeros[float64](nb_classes, batch_size)

# Fill in the non-zeros values
for sample_id, nonzero_idx in enumerate(sparse_labels):
  labels[nonzero_idx, sample_id] = 1

# Create a random tensor with predictions:
let pred = randomTensor(nb_classes, batch_size, -1.0..1.0)

# Display
# echo "### Pred & Labels"
# echo pred
# echo labels

echo "### Reference"
let sce_loss = pred.softmax_cross_entropy1(labels)
echo sce_loss
echo "### Challenger"
echo pred.softmax_cross_entropy2(labels)
echo "### Sparse reference"
echo pred.sparse_softmax_cross_entropy1(sparse_labels)
echo "### Sparse challenger"
echo pred.sparse_softmax_cross_entropy2(sparse_labels) # Warning it's only accurate at 1e-3 and precision is at 1e-2 with OpenMP

## Warmup for OpenMP threadpool and CPU on "on-demand" governor
discard pred *. pred

var start = epochTime()

start = epochTime()
for i in 0..<20:
  discard pred.softmax_cross_entropy1(labels)
echo "Softmax xentropy zipAxis, mean(sum <- map2): ", epochTime() - start


start = epochTime()
for i in 0..<20:
  discard pred.softmax_cross_entropy2(labels)
echo "Softmax xentropy Frobenius fold: ", epochTime() - start


start = epochTime()
for i in 0..<20:
  discard pred.sparse_softmax_cross_entropy1(sparse_labels)
echo "Sparse softmax naive loop: ", epochTime() - start

start = epochTime()
for i in 0..<20:
  discard pred.sparse_softmax_cross_entropy2(sparse_labels)
echo "Sparse softmax simplified loop + fold: ", epochTime() - start


echo "###### Backpropagation"

# We can't display those insane sized tensors
# echo "### Reference"
# echo softmax_cross_entropy_backward1(sce_loss, pred, labels)
# echo "### Sparse reference"
# echo sparse_softmax_cross_entropy_backward1(sce_loss, pred, sparse_labels)
# echo "### Dense Challenger"
# echo softmax_cross_entropy_backward2(sce_loss, pred, labels)
# echo "### Sparse Challenger"
# echo sparse_softmax_cross_entropy_backward2(sce_loss, pred, sparse_labels)


## Warmup for OpenMP threadpool and CPU on "on-demand" governor
# for i in 0..<5:
#   discard softmax_cross_entropy_backward1(loss, pred, labels)

start = epochTime()
for i in 0..<20:
  discard softmax_cross_entropy_backward1(sce_loss, pred, labels)
echo "Softmax xentropy backward: ", epochTime() - start

start = epochTime()
for i in 0..<20:
  discard sparse_softmax_cross_entropy_backward1(sce_loss, pred, sparse_labels)
echo "Sparse softmax xentropy backward: ", epochTime() - start

start = epochTime()
for i in 0..<20:
  discard softmax_cross_entropy_backward2(sce_loss, pred, labels)
echo "Backprop SCE optimized: ", epochTime() - start

start = epochTime()
for i in 0..<20:
  discard sparse_softmax_cross_entropy_backward2(sce_loss, pred, sparse_labels)
echo "Backprop Sparse SCE optimized: ", epochTime() - start


###### Streaming logsumexp

# on i5-5257 - single Threaded measure with cpuTime
# Softmax xentropy zipAxis, mean(sum <- map2): 29.35189
# Softmax xentropy Frobenius fold: 29.50572204589844
# Sparse softmax naive loop: 16.23663592338562              # Warning non-deterministic with OpenMP sometimes accurate sometimes 1e-2 (result returned too fast?)
# Sparse softmax simplified loop + fold: 15.22145700454712  # Warning it's only accurate at 1e-3 and precision is at 1e-2 with OpenMP
# ###### Backpropagation
# Softmax xentropy backward: 41.92376613616943
# Sparse softmax xentropy backward: 38.69393587112427
# Backprop SCE optimized: 19.85112118721008
# Backprop Sparse SCE optimized: 19.77226686477661

# on i5-5257 - OpenMP (dual core) measure with epochTIme
# Softmax xentropy zipAxis, mean(sum <- map2): 20.8852322101593
# Softmax xentropy Frobenius fold: 9.38280200958252
# Sparse softmax naive loop: 8.613812923431396                 # Warning non-deterministic with OpenMP sometimes accurate sometimes 1e-2 (result returned too fast?)
# Sparse softmax simplified loop + fold: 8.595002889633179      # Warning it's only accurate at 1e-3 and precision is at 1e-2 with OpenMP
# ###### Backpropagation
# Softmax xentropy backward: 26.3615939617157
# Sparse softmax xentropy backward: 24.02109813690186
# Backprop SCE optimized: 11.17088794708252
# Backprop Sparse SCE optimized: 11.63259792327881


# ###### 2-pass logsumexp
# #### No OpenMP
# Softmax xentropy zipAxis, mean(sum <- map2): 32.44389295578003
# Softmax xentropy Frobenius fold: 22.51684999465942
# Sparse softmax naive loop: 22.35019779205322
# Sparse softmax simplified loop + fold: 21.44888496398926
# ###### Backpropagation
# Softmax xentropy backward: 48.69741106033325
# Sparse softmax xentropy backward: 47.51089406013489
# Backprop SCE optimized: 28.69979810714722
# Backprop Sparse SCE optimized: 29.08899593353271

# #### OpenMP
# Softmax xentropy zipAxis, mean(sum <- map2): 21.78942489624023
# Softmax xentropy Frobenius fold: 12.66531205177307
# Sparse softmax naive loop: 12.26658082008362
# Sparse softmax simplified loop + fold: 14.28971004486084
# ###### Backpropagation
# Softmax xentropy backward: 31.89623808860779
# Sparse softmax xentropy backward: 27.4044349193573
# Backprop SCE optimized: 14.16854381561279
# Backprop Sparse SCE optimized: 15.79159712791443


# ###### 2-pass logsumexp_v2

# Softmax xentropy zipAxis, mean(sum <- map2): 34.91657114028931
# Softmax xentropy Frobenius fold: 24.60891890525818
# Sparse softmax naive loop: 24.37962102890015
# Sparse softmax simplified loop + fold: 23.77377796173096
# ###### Backpropagation
# Softmax xentropy backward: 48.49996495246887
# Sparse softmax xentropy backward: 46.18426609039307
# Backprop SCE optimized: 25.64767813682556
# Backprop Sparse SCE optimized: 25.24655890464783

# ### OpenMP

# Softmax xentropy zipAxis, mean(sum <- map2): 21.07627701759338
# Softmax xentropy Frobenius fold: 12.68725204467773
# Sparse softmax naive loop: 12.00046896934509
# Sparse softmax simplified loop + fold: 12.31421494483948
# ###### Backpropagation
# Softmax xentropy backward: 30.81571578979492
# Sparse softmax xentropy backward: 29.58900713920593
# Backprop SCE optimized: 14.53171896934509
# Backprop Sparse SCE optimized: 14.58920097351074



################ Dec 2017, with new Nim allocator
# unfortunately there is allocation issue see: issue #179

# No OpenMP

# Browser/Slack also using CPU
# Softmax xentropy zipAxis, mean(sum <- map2): 46.21152997016907
# Softmax xentropy Frobenius fold: 23.82361888885498
# Sparse softmax naive loop: 23.38949584960938
# Sparse softmax simplified loop + fold: 23.20574307441711
# ###### Backpropagation
# Softmax xentropy backward: 59.66550493240356
# Sparse softmax xentropy backward: 56.26315188407898
# Backprop SCE optimized: 29.80728793144226
# Backprop Sparse SCE optimized: 27.75954103469849

# CPU not busy
# Softmax xentropy zipAxis, mean(sum <- map2): 35.61256313323975
# Softmax xentropy Frobenius fold: 19.71887993812561
# Sparse softmax naive loop: 19.89032196998596
# Sparse softmax simplified loop + fold: 21.1429181098938
# ###### Backpropagation
# SIGSEGV: Illegal storage access. (Attempt to read from nil?)
