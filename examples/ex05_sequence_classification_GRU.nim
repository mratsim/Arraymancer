# Classify a sequence of number if increasing, decreasing or non-monotonic

import
  ../src/arraymancer,
  random, sequtils, strformat

# Make the results reproducible by initializing a random seed
randomize(42)

type SeqKind = enum
  Increasing, Decreasing, NonMonotonic

const DataSize = 1000

func classify(input: Tensor[float32], id: int): SeqKind =
  if input[id, 0] < input[id, 1] and input[id, 1] < input[id, 2]:
    result = Increasing
  elif input[id, 0] > input[id, 1] and input[id, 1] > input[id, 2]:
    result = Decreasing
  else:
    result = NonMonotonic

proc gen3(): array[3, float32] =
  # Generate monotonic sequence with more than 25% probability
  # Note that if NonMonotonic is drawn, it's just plain random
  let kind = rand([Increasing, Decreasing, NonMonotonic, NonMonotonic])

  result[0] = rand(1.0)
  for i in 1..2:
    case kind
    of Increasing:
      result[i] = rand(result[i-1]..1'f32)
    of Decreasing:
      result[i] = rand(0'f32..result[i-1])
    else:
      result[i] = rand(0'f32 .. 1'f32)

var dataset_X = toTensor newSeqWith(DataSize, gen3())
var dataset_y = newTensor[SeqKind](DataSize)

for i in 0 ..< DataSize:
  dataset_y[i] = classify(dataset_X, i)

echo dataset_X[0 ..< 10, _]
echo dataset_y[0 ..< 10]

# How many neurons do we need to change a light bulb, sorry compare 3 numbers? let's pick ...
const
  HiddenSize = 100
  BatchSize = 10
  Epochs = 2500

# Let's setup our neural network context, variables and model
let
  ctx = newContext Tensor[float32]
  # GRU needs this shape[sequence, batch, features]
  X = ctx.variable dataset_X.transpose.unsqueeze(2)
  y = dataset_y.astype(int)

# Check our shape
doAssert X.value.shape == [3, DataSize, 1]

network ctx, TheGreatSequencer:
  layers:
    # Note input_shape will only require the number of features in the future
    gru1: GRU([3, Batch_size, 1], HiddenSize, 1) # (input_shape, hidden_size, stacked_layers)
    classifier: Linear(HiddenSize, 3) # With GRU we need some reshape magic to move the batch_size/seq_len around
  forward x, hidden0:
    let (output, hiddenN) = gru1(x, hidden0)
    # hiddenN of shape [num_stacked_layers * num_directions, batch_size, hidden_size]
    # We discard the output and consider that the hidden layer stores
    # our monotonic info
    result = classifier(hiddenN.squeeze(0))

# Allocate the model
let model = ctx.init(TheGreatSequencer)
let optim = model.optimizerSGD(0.001'f32)

# And let's start training the network
for epoch in 0 ..< Epochs:
  for start_batch in countup(0, DataSize-1, BatchSize):
    # Deal with last batch being smaller
    let end_batch = min(X.value.shape[1]-1, start_batch + BatchSize)
    let X_batch = X[_, start_batch ..< end_batch, _]
    let target = y[start_batch ..< end_batch]

    let this_batch_size = end_batch - start_batch

    # Go through the model
    let hidden0 = ctx.variable zeros[float32](1, this_batch_size, HiddenSize)
    let clf = model.forward(X_batch, hidden0)

    # Go through our cost function
    let loss = clf.sparse_softmax_cross_entropy(target)

    # Backpropagate the errors and let the optimizer fix them.
    loss.backprop()
    optim.update()

  # Let's see how we fare:
  ctx.no_grad_mode:
    echo &"\nEpoch #{epoch} done. Testing accuracy"
    let hidden0 = ctx.variable zeros[float32](1, DataSize, HiddenSize)
    let y_pred = model
                  .forward(X, hidden0)
                  .value
                  .softmax
                  .argmax(axis = 1)
                  .squeeze

    let score = accuracy_score(y, y_pred)
    echo &"Accuracy: {score:.3f}%"
    echo "\n"
