import ../../src/arraymancer

let batch_size = 5
let nb_classes = 3

# Create a sparse label tensor of shape: [batch_size]
let sparse_labels = randomTensor(batch_size, nb_classes)

# Create the corresponding dense label tensor of shape [nb_classes, batch_size]
var labels = zeros[float64](nb_classes, batch_size)

# Fill in the non-zeros values
for sample_id, nonzero_idx in enumerate(sparse_labels):
  labels[nonzero_idx, sample_id] = 1

echo sparse_labels
echo labels

# Create a random tensor with predictions:
let pred = randomTensor(nb_classes, batch_size, -1.0..1.0)

echo pred

let sce_loss = softmax_cross_entropy(pred, labels)
let sparse_sce_loss = sparse_softmax_cross_entropy(pred, sparse_labels)

echo sce_loss
echo sparse_sce_loss


## Test the gradient, create closures first:
proc sce(pred: Tensor[float]): float =
  pred.softmax_cross_entropy(labels)

proc sparse_sce(pred: Tensor[float]): float =
  pred.sparse_softmax_cross_entropy(sparse_labels)

let expected_grad = sce_loss * numerical_gradient(pred, sce)
let expected_sparse_grad = sparse_sce_loss * numerical_gradient(pred, sparse_sce)

echo "Expected dense and sparse"
echo expected_grad
echo expected_sparse_grad

echo "Computed dense"
let grad = softmax_cross_entropy_backward(sce_loss, pred, labels)
echo grad

echo "Computed sparse"
let sparse_grad = sparse_softmax_cross_entropy_backward(sparse_sce_loss, pred, sparse_labels)
echo sparse_grad
