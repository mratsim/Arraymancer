import ../../src/arraymancer

# Note: dowload MNIST here: http://yann.lecun.com/exdb/mnist/
# uncompress and place it in the "arraymancer_root/bin/" folder
let imgs = read_mnist_images("bin/train-images-idx3-ubyte")
let labels = read_mnist_labels("bin/train-labels-idx1-ubyte")


for i in 0..10:
  echo "\n ############## "
  echo imgs[i, `...`]
  echo labels[i, `...`]