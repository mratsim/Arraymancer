import ../../src/arraymancer


let imgs = read_mnist_images("bin/train-images-idx3-ubyte")
let labels = read_mnist_labels("bin/train-labels-idx1-ubyte")


for i in 0..10:
  echo "\n ############## "
  echo imgs[i, `...`]
  echo labels[i, `...`]