import ../../src/arraymancer

# https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/

# Reminder, for now batch_size is the innermost index
let predicted = [-3.44'f32, 1.16, -0.81, 3.91].toTensor.reshape(4,1)
let truth = [0'f32, 0, 0, 1].toTensor.reshape(4,1)

echo softmax_cross_entropy(predicted, truth) # Should be 0.0709

let sparse_truth = [3].toTensor

echo sparse_softmax_cross_entropy(predicted, sparse_truth)  # Should be 0.0709