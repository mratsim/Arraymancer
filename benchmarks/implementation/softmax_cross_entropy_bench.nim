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


## Warmup for OpenMP threadpool and CPU on "on-demand" governor
for i in 0..<5:
  discard softmax_cross_entropy1(pred, labels)

var start = epochTime()
for i in 0..<20:
  discard softmax_cross_entropy1(pred, labels)
echo " Softmax xentropy zipAxis, mean(sum <- map2): ", epochTime() - start


start = epochTime()
for i in 0..<20:
  discard softmax_cross_entropy2(pred, labels)
echo " Softmax xentropy Froebenius fold: ", epochTime() - start


### On i5-5257U - no OpenMP
# ### Reference
# 11.67763585880838
# ### Challenger
# 11.67763585880838
#  Softmax xentropy zipAxis, mean(sum <- map2): 30.27224016189575
#  Softmax xentropy Froebenius fold: 17.71994495391846

### On i5-5257U - with OpenMP
# ### Reference
# 11.67763585880838
# ### Challenger
# 11.67763585880837
#  Softmax xentropy zipAxis, mean(sum <- map2): 28.51023387908936
#  Softmax xentropy Froebenius fold: 10.95725297927856