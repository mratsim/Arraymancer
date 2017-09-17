| Language/lib      | Normal matmul | element-wise  matmul (Hadamard) | vec-vec dot product | mat-vec multiplication|
| ------------- | ---------------------------- | --- | --- | --- |
| Arraymancer  | A * B | \|*\| | A * B | A * B |
| neo/linalg  | A * B | \|*\| | A * B | A * B |
| Julia  | A * B | .* | | dot(A, B) | A * B |
| Numpy ndarray| np.dot(A, B) or np.matmul(A, B) or A @ B| np.multiply(A, B) or A * B | np.dot(A, B) or np.inner(A, B) | np.dot(A, B) |
| R | A %*% B | A * B | A %*% B or dot(A, B)| A %*% B |
| Tensorflow | tf.matmul(A, B) or A @ B | tf.multiply(A, B) | tf.matmul(a, b, transpose_a=False, transpose_b=True) or tf.tensordot(a, b, 1) or tf.einsum('i,i->', x, y) | same reshape/transpose/einsum shenanigans as vec-vec|
| Torch/PyTorch | torch.mm(A,B) or torch.matmul(A,B) | torch.cmul(A, B) | torch.dot(A, B) or torch.matmul(A, B) | torch.mv(A, B) or torch.dot(A, B)
| Theano | theano.tensor.dot(A, B) | A * B | dot(A, B) or vdot(A, B) ?| dot(A, B) or tensordot(A,B) ? |
| Common math |