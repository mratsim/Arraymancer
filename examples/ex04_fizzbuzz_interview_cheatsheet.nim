# A port to Arraymancer of Joel Grus hilarious FizzBuzz in Tensorflow:
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

# Interviewer: Welcome, can I get you a coffee or anything? Do you need a break?
# ...
# Interviewer: OK, so I need you to print the numbers from 1 to 100,
#              except that if the number is divisible by 3 print "fizz",
#              if it's divisible by 5 print "buzz", and if it's divisible by 15 print "fizzbuzz".

# Let's start with standard imports
import ../src/arraymancer, math, strformat

# We want to input a number and output the correct "fizzbuzz" representation
# ideally the input is a represented by a vector of real values between 0 and 1
# One way to do that is by using the binary representation of number
proc binary_encode(i: int, num_digits: int): Tensor[float32] =
  result = newTensor[float32](1, num_digits)
  for d in 0 ..< num_digits:
    result[0, d] = float32(i shr d and 1)

# For the input, we distinguish 4 cases: nothing, fizz, buzz and fizzbuzz.
func fizz_buzz_encode(i: int): int =
  if   i mod 15 == 0: return 3 # fizzbuzz
  elif i mod  5 == 0: return 2 # buzz
  elif i mod  3 == 0: return 1 # fizz
  else              : return 0

# Next, let's generate training data, we don't want to train on 1..100, that's our test values
# We can't tell the neural net the truth values it must discover the logic by itself.
# so we use values between 101 and 1024 (2^10)
const NumDigits = 10

var x_train = newTensor[float32](2^NumDigits - 101, NumDigits)
var y_train = newTensor[int](2^NumDigits - 101)

for i in 101 ..< 2^NumDigits:
  x_train[i - 101, _] = binary_encode(i, NumDigits)
  y_train[i - 101] = fizz_buzz_encode(i)

# How many neurons do we need to change a light bulb, sorry do a division? let's pick ...
const NumHidden = 100

# Let's setup our neural network context, variables and model
let
  ctx = newContext Tensor[float32]
  X   = ctx.variable x_train

network FizzBuzzNet:
  layers:
    hidden: Linear(NumDigits, NumHidden)
    output: Linear(NumHidden, 4)
  forward x:
    x.hidden.relu.output

let model = ctx.init(FizzBuzzNet)
let optim = model.optimize(SGD, 0.05'f32)

func fizz_buzz(i: int, prediction: int): string =
  [$i, "fizz", "buzz", "fizzbuzz"][prediction]

# Phew, finally ready to train, let's pick the batch size and number of epochs
const BatchSize = 128
const Epochs    = 2500

# And let's start training the network
for epoch in 0 ..< Epochs:
  # Here I should probably shuffle the input data.
  for start_batch in countup(0, x_train.shape[0]-1, BatchSize):

    # Pick the minibatch
    let end_batch = min(x_train.shape[0]-1, start_batch + BatchSize)
    let X_batch = X[start_batch ..< end_batch, _]
    let target = y_train[start_batch ..< end_batch]

    # Go through the model
    let clf = model.forward(X_batch)

    # Go through our cost function
    let loss = clf.sparse_softmax_cross_entropy(target)

    # Backpropagate the errors and let the optimizer fix them.
    loss.backprop()
    optim.update()

  # Let's see how we fare:
  ctx.no_grad_mode:
    echo &"\nEpoch #{epoch} done. Testing accuracy"

    let y_pred = model
                  .forward(X)
                  .value
                  .softmax
                  .argmax(axis = 1)
                  .squeeze

    let score = y_pred.accuracy_score(y_train)
    echo &"Accuracy: {score:.3f}%"
    echo "\n"


# Our network is trained, let's see if it's well behaved

# Now let's use what we really want to fizzbuzz, numbers from 1 to 100
var x_buzz = newTensor[float32](100, NumDigits)
for i in 1 .. 100:
  x_buzz[i - 1, _] = binary_encode(i, NumDigits)

# Wrap them for neural net
let X_buzz = ctx.variable x_buzz

# Pass it through the network
ctx.no_grad_mode:
  let y_buzz = model
                .forward(X_buzz)
                .value
                .softmax
                .argmax(axis = 1)
                .squeeze

# Extract the answer
var answer: seq[string] = @[]

for i in 1..100:
  answer.add fizz_buzz(i, y_buzz[i - 1])

echo answer
# @["1", "fizzbuzz", "fizz", "4", "buzz", "fizz", "7", "8", "fizz", "buzz",
#   "11", "fizz", "13", "14", "fizzbuzz", "16", "17", "fizz", "19", "buzz",
#   "21", "22", "23", "24", "buzz", "26", "fizz", "28", "29", "fizzbuzz",
#   "31", "buzz", "33", "34", "buzz", "fizz", "37", "buzz", "fizz", "buzz",
#   "41", "fizz", "43", "44", "fizzbuzz", "46", "47", "fizz", "49", "buzz",
#   "51", "52", "53", "fizz", "buzz", "56", "fizz", "58", "59", "fizzbuzz",
#   "61", "62", "fizz", "64", "65", "66", "67", "68", "69", "70",
#   "71", "fizz", "73", "74", "fizzbuzz", "76", "77", "fizz", "79", "80",
#   "81", "82", "83", "fizz", "buzz", "86", "fizz", "88", "89", "fizzbuzz",
#   "91", "92", "93", "94", "buzz", "96", "97", "98", "fizz", "100"]

# I guess 100 neurons are not enough to learn multiplication :/.
