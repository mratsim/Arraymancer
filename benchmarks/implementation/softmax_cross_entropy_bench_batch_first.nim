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
  var sample_softmax_xentropy = zeros[T](input.shape[0], 1)
  var i = 0
  for sample_input, sample_target in zipAxis(input, target, 0):
    let lse = sample_input.classic_logsumexp_v2 # y.streaming_log_sumexp
    sample_softmax_xentropy[i, 0] = sum:
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

  let sum_logsumexp = fold_axis_inline(input, T, fold_axis=0) do:
    x = y.streaming_log_sumexp
  do:
    x += y.streaming_log_sumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(input.shape[0])

####### Sparse
proc sparse_softmax_cross_entropy1[T](input: Tensor[T], target: Tensor[int]): T =
  for i in 0||(input.shape[0]-1):
    let lse = input[i,_].streaming_log_sumexp
    when not declared(openmp):
      result += lse - input[i, target[i]]
    else:
      let tmp = lse - input[i, target[i]]
      {.emit:"#pragma omp atomic".}
      {.emit:"`result` += `tmp`;".}
  result /= T(input.shape[0])

proc sparse_softmax_cross_entropy2[T](input: Tensor[T], target: Tensor[int]): T =
  let batch_size = input.shape[0]

  for i in 0||(batch_size-1):
    # Unfortunately we can't use `result` in a parallel for reduction declaration so we need atomic
    when not declared(openmp):
      result = input[i, target[i]]
    else:
      let tmp = input[i, target[i]]
      {.emit:"#pragma omp atomic".}
      {.emit:"`result` += `tmp`;".}

  let sum_logsumexp = fold_axis_inline(input, T, fold_axis=0) do:
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
  let batch_size = cached_tensor.shape[0]

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  result = zeros_like(cached_tensor)

  for i in 0||(batch_size-1):
    let (max, sumexp) = cached_tensor[i,_].streaming_max_sumexp

    var res_slice = result[i,_]

    apply3_inline(res_slice, cached_tensor[i,_], target[i,_]):
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

  let batch_size = cached_tensor.shape[0]

  result = zeros_like(cached_tensor)

  for i, truth_idx in enumerate(target):
    result[i, truth_idx] = -1

  for i in 0||(batch_size-1):
    let (max, sumexp) = cached_tensor[i, _].streaming_max_sumexp

    var res_slice = result[i, _]

    apply2_inline(res_slice, cached_tensor[i, _]):
      grad * (stable_softmax(y, max, sumexp) + x) / T(batch_size)

############ New optimized
proc softmax_cross_entropy_backward2[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  let batch_size = cached_tensor.shape[0]

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  let axis_max_sumexp = cached_tensor.streaming_max_sumexp(axis = 0).broadcast(cached_tensor.shape)
  # let axis_max_sumexp = cached_tensor.classic_max_sumexp(axis = 0).broadcast(cached_tensor.shape)

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

  let batch_size = cached_tensor.shape[0]

  result = zeros_like(cached_tensor)
  for i, truth_idx in enumerate(target):
    result[i, truth_idx] = -1

  let axis_max_sumexp = cached_tensor.streaming_max_sumexp(axis = 0).broadcast(cached_tensor.shape)
  # let axis_max_sumexp = cached_tensor.classic_max_sumexp(axis = 0).broadcast(cached_tensor.shape)


  apply3_inline(result, cached_tensor, axis_max_sumexp):
      grad * (stable_softmax(y, z.max, z.sumexp) + x) / T(batch_size)


####### Bench:

let batch_size = 200
let nb_classes = 100000

# Create a sparse label tensor of shape: [batch_size]
let sparse_labels = randomTensor(batch_size, nb_classes)

# Create the corresponding dense label tensor of shape [batch_size, nb_classes]
var labels = zeros[float64](batch_size, nb_classes)


# Fill in the non-zeros values
for sample_id, nonzero_idx in enumerate(sparse_labels):
  labels[sample_id, nonzero_idx] = 1

# Create a random tensor with predictions:
let pred = randomTensor(batch_size, nb_classes, -1.0..1.0)

# Display
# echo "### Pred & Labels"
# echo pred
# echo sparse_labels
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


################ Dec 2017, with new Nim allocator

# No OpenMP
# Softmax xentropy zipAxis, mean(sum <- map2): 4.697981119155884
# Softmax xentropy Frobenius fold: 4.604862928390503
# Sparse softmax naive loop: 3.006110191345215
# Sparse softmax simplified loop + fold: 3.005247116088867
# ###### Backpropagation
# Softmax xentropy backward: 7.50258207321167
# Sparse softmax xentropy backward: 7.282567977905273
# Backprop SCE optimized: 8.412425994873047
# Backprop Sparse SCE optimized: 8.289819955825806

# OpenMP
# Softmax xentropy zipAxis, mean(sum <- map2): 3.449564933776855
# Softmax xentropy Frobenius fold: 2.707981824874878
# Sparse softmax naive loop: 1.304688930511475
# Sparse softmax simplified loop + fold: 1.330923080444336
# ###### Backpropagation
# Softmax xentropy backward: 3.666075944900513
# Sparse softmax xentropy backward: 3.913578033447266
# Backprop SCE optimized: 4.060986995697021
# Backprop Sparse SCE optimized: 4.227570056915283
