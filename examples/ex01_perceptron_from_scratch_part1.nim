# Example Perceptron in Arraymancer.

# We will use as examples the OR function similar to this article:
# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/

# We can get the perceptron mathematical definition from my previous Nim library:
# https://github.com/mratsim/nim-rmad/blob/c0ca3729c2d0c857ff4a983b98a1f0cfded8be1c/examples/ex04_2_hidden_layers_neural_net.nim#L41-L48

# The full code without comment is 35 lines (with comment it's 235+)
# I hope to give you a feel of working with tensors and what you can use
# to deal with your own problems.
# In the future, I hope to reduce such a trivial example to 5 lines using built-in functions.

# Without further ado, here is the code we will comment together

############################################
### import arraymancer
###
### var model = randomTensor([3,3], -1.0..1)
### let data = randomTensor([100,2], 2)
###
### var X = data.astype(float)
### var y = X[_, 0] + X[_, 1]
###
### let batch_size = 8
### var i = 0
###
### var batch_X = X[i ..< i + batch_size]
### var batch_y = y[i ..< i + batch_size]
###
### proc min_1_and_x(x: float): float =
###   if x >= 1:
###     return 1
###   return x
###
### batch_y = y.fmap(min_1_and_x)
###
### let vec_ones = ones([batch_size,1], float)
### batch_X = batch_X.concat(vec_ones, axis=1)
###
### import future
###
### let s_weights = randomTensor([4], -1.0..1)
###
### proc score(n: Tensor[float], weights: Tensor[float]): Tensor[float] =
###   result = n.fmap(x => max(0,x))
###   result = concat(result, ones([n.shape[0],1], float), axis=1)
###   result = result * weights
###
### let s1 = score(batch_X * model, s_weights).reshape([batch_size,1])
######################################################################

# Okay let's start
# The perceptron equation is
# n1 = relu(a1 * x + b1 * y + c1) # First neuron + relu activation
# n2 = relu(a2 * x + b2 * y + c2) # 2nd neuron + relu activation
# n3 = relu(a3 * x + b3 * y + c3) # 3nd neuron + relu activation
# score = a4 * n1 + b4 * n2 + c4 * n3 + d4 # score

# x and y are the inputs (0, 0), (0, 1), (1, 0) or (1, 1) for the OR function
# a, b, c are the weights of each neurons, they will be adjusted over several iterations
# so that the perceptron learns and can reproduce the OR function.

# `relu` is an activation function, mathematically is the maximum between 0 and the input
# `relu(x) = max(0, x)`.

# `score` will be our way to give a feedback to the perceptron on how it performed for the current iteration
# Note the constant c1, c2, c3, d4 that are aded for each neurons and the score
# It will give the model more wiggle rooms to adjust.

# Let's generate our model, arbitrarily initialized with random weights between -1 and 1

import ../src/arraymancer

var model = randomTensor([3,3], -1.0..1)
# We create atensor filled with random values between -1 and 1 (exclusive)
# Random seed can be set by importing ``random`` and ``randomize(seed)``

echo model
# Tensor of shape 3x3 of type "float" on backend "Cpu"
# |-0.5346077004780883    0.6456934023082188      -0.3333241059604588|
# |0.9789923693794811     -0.1125470106621034     -0.8483679531473696|
# |0.1873127270016446     -0.608930613694278      0.05177817001150142|

# Nice, now we need data, how about 100 examples, 42 used for training and the rest for testing?
# Inputs are only 0 or 1 which makes it easier

let data = randomTensor([100,2], 2)
# Here we create a randomTensor with int values between 0 (inclusive) and 2 (exclusive)
# meaning either 0 or 1
# `let` declares a variable that can't be modified.
# It is useful to prevent mistakes (and also it's easier to optimize)

echo data
# Tensor of shape 100x2 of type "int" on backend "Cpu"
# |0      0|
# |0      0|
# |0      0|
# |1      0|
# |0      0|
# |1      1|
# |0      1|
# |0      1|
# ...

# Actually we will work with floats and not integers, so let's convert data.
# This is also an excellent time to introduce `var`
# `var`declares a variable that you will be able to modify after
var X = data.astype(float)

# Note:
# Be aware that Nim is strictly typed, here X is a float,
# If you do `X = X.astype(string)` it will tell you that
# you are mixing oil (X is a float) and water (X.astype(string) is a string)

# Now we need the truth values: the or function is
# Convention 0 corresponds to false and 1 to true
# 0 0 ==> 0
# 0 1 ==> 1
# 1 0 ==> 1
# 1 1 ==> 1

# So we need to check each row if there is at least a 1.
# In Python / R you would use dataframe row operations, we don't have that (yet)
# so I'm going to cheat a little.
# I will first add the 2 columns ...
# _ is a joker to take the whle dimension

var y = X[_, 0] + X[_, 1]

# We only display the first 8 values
echo y[0..<8, _]
# Tensor of shape 8x1 of type "float" on backend "Cpu"
#         0.0|
#         0.0|
#         0.0|
#         1.0|
#         0.0|
#         2.0|
#         1.0|
#         1.0|
#         ...

# We will process the input by batches of 8, then update the weights
# Then process the next 8, then update again, etc.
# When we finish processing all the examples, we will have done 1 epoch in deep learning jargon
# Processing by mini-batch allows to go faster.
# Using bigger batch will "dilute" the correction too much, the move will not be big enough.

let batch_size = 8
var i = 0  # current iteration, well we won't use a for loop in this example
           # so it's just for show

var batch_X = X[i ..< i + batch_size]
var batch_y = y[i ..< i + batch_size]

# Now 0 is both x and y are false ==> OR retruns false
# and 1 or 2 corresponds to at least one is true ==> OR returns true
# so we modify y accordingly

proc min_1_and_x(x: float): float =
  if x >= 1:
    return 1

  return x # This is only reached if x = 0

# fmap applies a function to each element of a tensor
batch_y = y.fmap(min_1_and_x)

echo batch_y
# Tensor of shape 8x1 of type "float" on backend "Cpu"
#         0.0|
#         0.0|
#         0.0|
#         1.0|
#         0.0|
#         1.0|
#         1.0|
#         1.0|


# Now remember, the equation of each neuron is a*x + b*y + c
# We are going to add a dummy column filled with 1 so we can do a*x + b*y + c*1
# which is a Matrix-Vector product.

# We create a column vector of ones
let vec_ones = ones([batch_size,1], float)

# And we concatenate along the columns
batch_X = batch_X.concat(vec_ones, axis=1)


# Now let's recap
echo batch_X
# Tensor of shape 8x3 of type "float" on backend "Cpu"
# |0.0    0.0     1.0|
# |0.0    0.0     1.0|
# |0.0    0.0     1.0|
# |1.0    0.0     1.0|
# |0.0    0.0     1.0|
# |1.0    1.0     1.0|
# |0.0    1.0     1.0|
# |0.0    1.0     1.0|
# ...

echo model
# Tensor of shape 3x3 of type "float" on backend "Cpu"
# |-0.5346077004780883    0.6456934023082188      -0.3333241059604588|
# |0.9789923693794811     -0.1125470106621034     -0.8483679531473696|
# |0.1873127270016446     -0.608930613694278      0.05177817001150142|

# We will do X * model, meaning each neurons is actually in column not in row
# n1 = relu(a1 * x + b1 * y + c1) # First neuron + relu activation
# n2 = relu(a2 * x + b2 * y + c2) # 2nd neuron + relu activation
# n3 = relu(a3 * x + b3 * y + c3) # 3nd neuron + relu activation

# So weights and X * model have the following form
#                       |a1           a2  a3|
#                       |b1           b2  b3|
#                       |c1           c2  c3|
#
# |0.0    0.0     1.0|  |a1*0+b1*0+c1 ... ...
# |0.0    0.0     1.0|  ...           ... ...
# |0.0    0.0     1.0|  ...           ... ...
# |1.0    0.0     1.0|  |a1*1+b1*0+c1 ... ...
# |0.0    0.0     1.0|  ...           ... ...
# ...

# The values of the tensor can be "activated" by the relu function
# which is the max between 0 and x
# Lets import the "future" library for the very nice syntax shortcut "=>"
# like this "output.fmap(x => max(0,x))"
import future

# Now we can build the score function.
# score = a4 * n1 + b4 * n2 + c4 * n3 + d4 # score
# We can also represent a4 and co by a tensor
let s_weights = randomTensor([4], -1.0..1)

# And the score function
proc score(n: Tensor[float], weights: Tensor[float]): Tensor[float] =

  # n is the output of batch_X * model

  result = n.fmap(x => max(0,x))
  # result is the implicit value returned in nim

  # We also concatenate a dummy vector filled with ones for d4
  # dummy vector has the same number of rows as n (n.shape[0])
  # Note that we can use interchangeably result.concat(...) and concat(result, ...)
  result = concat(result, ones([n.shape[0],1], float), axis=1)

  # And finally we compute the score.
  # Remember, result will be implicitly returned at the end

  result = result * weights

# Now we want our network to output -1 if the correct result is 0 (false)
# and 1 if the correct result is 1 (true)
# Why -1 and 1? Later to get the contribution of each weights for adjustment
# we will take the gradient and use it to adjust the weight.
# Gradient of a*x w.r.t. x is a, if a is 0, very small adjustments will be done

# First score, we use s1 to a column vector (matrix of shape batch_size,1)
let s1 = score(batch_X * model, s_weights).reshape([batch_size,1])

echo s1
# Tensor of shape 8x1 of type "float" on backend "Cpu"
#         -0.9839423601184261|
#         -0.9839423601184261|
#         -0.9839423601184261|
#         -0.8304378825658082|
#         -0.9839423601184261|
#         -1.347142576688641|
#         -1.788149588470766|
#         -1.788149588470766|

# Remember the truth values? our network will have lots of learning to do
# to recognize (0, 1), (1, 1), (1, 0)

echo batch_y
# Tensor of shape 8x1 of type "float" on backend "Cpu"
#         0.0|
#         0.0|
#         0.0|
#         1.0|
#         0.0|
#         1.0|
#         1.0|
#         1.0|

# Well, this is the end for this example.
# We don't have the facilities yet to do the backpropagation directly with tensors.

# Thanks for reading