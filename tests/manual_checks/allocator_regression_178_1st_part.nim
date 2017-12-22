import  ../../src/tensor/tensor,
        math

# GC_disableMarkAndSweep() # This only delays the crash, it seems like using setters `data=` in the init_cpu procs prevent memory freeing in certain circumstances

proc streaming_max_sumexp[T](t: Tensor[T]): T {.noSideEffect, inline.}=
  return -Inf.T # Plot twist, a function is needed

proc softmax_cross_entropy1*[T](input, target: Tensor[T]): T =
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])
  var i = 0
  for sample_input, sample_target in zipAxis(input, target, 1):
    let lse = 10.T
    sample_softmax_xentropy[0, i] = sum:  # Plt twist, this is needed as well
      map2_inline(sample_input, sample_target):
        y * (lse - x)
    inc i
  result = sample_softmax_xentropy.mean

###### Backprop bench

proc softmax_cross_entropy_backward1[T](
        gradient: T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  let batch_size = cached_tensor.shape[1]

  result = zeros_like(cached_tensor)

  for i in 0..<batch_size:
    let test = cached_tensor[_,i].streaming_max_sumexp

    var res_slice = result[_, i]

    apply3_inline(res_slice, cached_tensor[_,i], target[_,i]):
      gradient * ((y + y + y) - z) / T(batch_size) # Plot twist: y + y + y needed, / batch_size needed

############ New optimized
proc softmax_cross_entropy_backward2[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=

  result = map3_inline(cached_tensor, target, cached_tensor): # Plot twist: this is also needed
      0.T

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

echo "### Reference"
let sce_loss = softmax_cross_entropy1(pred, labels)
echo sce_loss


import times # Plot twist - this is needed
var start = epochTime()

echo "###### Backpropagation"

for i in 0..<20:
  discard softmax_cross_entropy_backward1(sce_loss, pred, labels)
echo "Softmax xentropy backward: ", epochTime() - start # Plot twist - this is needed

for i in 0..<20:
  discard softmax_cross_entropy_backward2(sce_loss, pred, labels)
echo "Backprop SCE optimized: ", epochTime() - start
