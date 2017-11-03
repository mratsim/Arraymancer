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

proc streaming_max_sumexp*[T](t: Tensor[T]): tuple[max:T, sumexp: T] {.noSideEffect, inline.}=
  result.max = -Inf.T   # will store the streaming max of the tensor
  result.sumexp = 0.T   # will store the streaming sum of exp of the tensor

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp *= exp(result.max - x)
      result.sumexp += 1
      result.max = x

proc logsumexp*[T: SomeReal](t: Tensor[T]): T =
  let (max, sumexp) = t.streaming_max_sumexp
  result = max + ln(sumexp)


proc softmax_cross_entropy1*[T](input, target: Tensor[T]): T =
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])
  var i = 0
  for sample_input, sample_target in zipAxis(input, target, 1):
    let lse = sample_input.logsumexp
    sample_softmax_xentropy[0, i] = sum:
      map2_inline(sample_input, sample_target):
        y * (lse - x)
    inc i
  result = sample_softmax_xentropy.mean

proc frobenius_inner_prod*[T](a,b: Tensor[T]): T =
  sum:
    map2_inline(a,b):
      x * y

proc softmax_cross_entropy2*[T](input, target: Tensor[T]): T =
  result = frobenius_inner_prod(input, target)

  let sum_logsumexp = fold_axis_inline(input, T, 1) do:
    x = y.logsumexp
  do:
    x += y.logsumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(input.shape[1])

####### Sparse
proc sparse_softmax_cross_entropy1*[T](input: Tensor[T], target: Tensor[int]): T =
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])
  for i in 0..<input.shape[1]:  # Can't use OpenMP here, Illegal storage access
    let lse = input[_,i].logsumexp
    sample_softmax_xentropy[0, i] = lse - input[target[i], i]
  result = sample_softmax_xentropy.mean

proc sparse_softmax_cross_entropy2*[T](input: Tensor[T], target: Tensor[int]): T =
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
    x = y.logsumexp
  do:
    x += y.logsumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(batch_size)

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
echo softmax_cross_entropy1(pred, labels)
echo "### Challenger"
echo softmax_cross_entropy2(pred, labels)
echo "### Sparse reference - super slow due to generalSeqAssign"
echo sparse_softmax_cross_entropy1(pred, sparse_labels)
echo "### Sparse challenger"
echo sparse_softmax_cross_entropy1(pred, sparse_labels)

## Warmup for OpenMP threadpool and CPU on "on-demand" governor
for i in 0..<5:
  discard softmax_cross_entropy1(pred, labels)

var start = epochTime()
# for i in 0..<20:
#   discard softmax_cross_entropy1(pred, labels)
# echo " Softmax xentropy zipAxis, mean(sum <- map2): ", epochTime() - start


start = epochTime()
for i in 0..<20:
  discard softmax_cross_entropy2(pred, labels)
echo " Softmax xentropy Frobenius fold: ", epochTime() - start


# start = epochTime()
# for i in 0..<20:
#   discard sparse_softmax_cross_entropy1(pred, sparse_labels)
# echo " Sparse softmax naive loop: ", epochTime() - start

start = epochTime()
for i in 0..<20:
  discard sparse_softmax_cross_entropy2(pred, sparse_labels)
echo " Sparse softmax simplified loop + fold: ", epochTime() - start


### On i5-5257U - no OpenMP
#  Softmax xentropy zipAxis, mean(sum <- map2): 30.71215105056763
#  Softmax xentropy Froebenius fold: 18.18636608123779
#  Sparse softmax naive loop: 113.4928979873657
#  Sparse softmax simplified loop + fold: 16.42902994155884

### On i5-5257U - with OpenMP
#  Softmax xentropy zipAxis, mean(sum <- map2): 25.62056088447571
#  Softmax xentropy Froebenius fold: 10.49299907684326
#  Sparse softmax naive loop: 102.5834209918976
#  Sparse softmax simplified loop + fold: 9.73446798324585



###### Backprop bench
proc stable_softmax[T](x, max, sumexp: T): T {.noSideEffect, inline.}=
  # Numerically stable streaming softmax helper
  result = exp(x - max) / sumexp

proc softmax_cross_entropy_backward1*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  let batch_size = cached_tensor.shape[1]

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.data[gradient.offset]

  result = zeros_like(cached_tensor)

  for i in 0 ..< batch_size: # Can't use OpenMP here, Illegal storage access
    let (max, sumexp) = cached_tensor[_,i].streaming_max_sumexp

    result[_,i] = map2_inline(cached_tensor[_,i], target[_,i]):
      grad * (stable_softmax(x, max, sumexp) - y) / T(batch_size)

proc sparse_softmax_cross_entropy_backward1*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[int]
        ): Tensor[T] {.noInit.}=

  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.data[gradient.offset]

  let batch_size = cached_tensor.shape[1]

  result = zeros_like(cached_tensor)

  for i, truth_idx in enumerate(target):
    result[truth_idx, i] = -1

  for i in 0 ..< batch_size: # Can't use OpenMP here, Illegal storage access
    let (max, sumexp) = cached_tensor[_,i].streaming_max_sumexp

    var res_slice = result.unsafeSlice(_, i)

    apply2_inline(res_slice, cached_tensor[_,i]):
      grad * (stable_softmax(y, max, sumexp) + x) / T(batch_size)

######

echo "###### Backpropagation"

let loss = softmax_cross_entropy1(pred, labels)

echo "### Reference - too slow ... more than 5min"
# echo softmax_cross_entropy_backward1(loss, pred, labels)
echo "### Sparse reference - too slow ... more than 5min"
# echo sparse_softmax_cross_entropy_backward1(loss, pred, sparse_labels)

## Warmup for OpenMP threadpool and CPU on "on-demand" governor
# for i in 0..<5:
#   discard softmax_cross_entropy_backward1(loss, pred, labels)

# start = epochTime()
# for i in 0..<20:
#   discard softmax_cross_entropy_backward1(loss, pred, labels)
# echo " Softmax xentropy backward: ", epochTime() - start

# start = epochTime()
# for i in 0..<20:
#   discard sparse_softmax_cross_entropy_backward1(loss, pred, sparse_labels)
# echo " Sparse softmax xentropy backward: ", epochTime() - start