# Classify a sequence of number if increasing, decreasing or non-monotonic

import
  ../src/arraymancer,
  random, sequtils, strformat

# Make the results reproducible by initializing a random seed
randomize(42)

type SeqKind = enum
  Increasing, Decreasing, NonMonotonic

const DataSize = 30000

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
  let kind = sample([Increasing, Decreasing, NonMonotonic, NonMonotonic])

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

echo "Example dataset"
echo dataset_X[0 ..< 10, _]
echo "Corresponding labels"
echo dataset_y[0 ..< 10]
echo "\n"

# How many neurons do we need to change a light bulb, sorry compare 3 numbers? let's pick ...
const
  HiddenSize = 256
  BatchSize = 512
  Epochs = 8
  Layers = 4

# Let's setup our neural network context, variables and model
let
  ctx = newContext Tensor[float32]
  # GRU needs this shape[sequence, batch, features]
  X = ctx.variable dataset_X.transpose.unsqueeze(2)
  y = dataset_y.astype(int)

# Check our shape
doAssert X.value.shape == [3, DataSize, 1]

network TheGreatSequencer:
  layers:
    gru1: GRULayer(1, HiddenSize, 4) # (num_input_features, hidden_size, stacked_layers)
    fc1: Linear(HiddenSize, 32)                  # 1 classifier per GRU layer
    fc2: Linear(HiddenSize, 32)
    fc3: Linear(HiddenSize, 32)
    fc4: Linear(HiddenSize, 32)
    classifier: Linear(32 * 4, 3)                # Stacking a classifier which learns from the other 4
  forward x, hidden0:
    let
      (output, hiddenN) = gru1(x, hidden0)
      clf1 = hiddenN[0, _, _].squeeze(0).fc1.relu
      clf2 = hiddenN[1, _, _].squeeze(0).fc2.relu
      clf3 = hiddenN[2, _, _].squeeze(0).fc3.relu
      clf4 = hiddenN[3, _, _].squeeze(0).fc4.relu

    # Concat all
    # Since concat backprop is not implemented we cheat by stacking
    # Then flatten
    result = stack(clf1, clf2, clf3, clf4, axis = 2)
    result = classifier(result.flatten)

# Allocate the model
let model = ctx.init(TheGreatSequencer)
var optim = model.optimizerAdam(0.01'f32)

# And let's start training the network
for epoch in 0 ..< Epochs:
  for start_batch in countup(0, DataSize-1, BatchSize):
    # Deal with last batch being smaller
    let end_batch = min(X.value.shape[1]-1, start_batch + BatchSize)
    let X_batch = X[_, start_batch ..< end_batch, _]
    let target = y[start_batch ..< end_batch]

    let this_batch_size = end_batch - start_batch

    # Go through the model
    let hidden0 = ctx.variable zeros[float32](Layers, this_batch_size, HiddenSize)
    let clf = model.forward(X_batch, hidden0)

    # Go through our cost function
    let loss = clf.sparse_softmax_cross_entropy(target)

    # Backpropagate the errors and let the optimizer fix them.
    loss.backprop()
    optim.update()

  # Let's see how we fare:
  ctx.no_grad_mode:
    let hidden0 = ctx.variable zeros[float32](Layers, DataSize, HiddenSize)
    let y_pred = model
                  .forward(X, hidden0)
                  .value
                  .softmax
                  .argmax(axis = 1)
                  .squeeze

    let score = y_pred.accuracy_score(y)
    echo &"Epoch #{epoch:> 04}. Accuracy: {score*100:00.3f}%"

###################
# Output

# Example dataset
# Tensor[system.float32] of shape [10, 3] of type "float32" on backend "Cpu"
# |0.08715851604938507	0.6252052187919617	0.8734603524208069|
# |0.4635309278964996	0.1152218133211136	0.6088221073150635|
# |0.4754987359046936	0.7151913642883301	0.7708750367164612|
# |0.3764243125915527	0.3795507848262787	0.9351327419281006|
# |0.6993147730827332	0.733343780040741	0.8100541830062866|
# |0.4297148883342743	0.09527183324098587	0.01486776024103165|
# |0.875207245349884	0.2490521669387817	0.1578131020069122|
# |0.02143412455916405	0.0222312156111002	0.7928663492202759|
# |0.07909850776195526	0.1905942112207413	0.4293616414070129|
# |0.04384680092334747	0.7198637723922729	0.2911368310451508|

# Corresponding labels
# Tensor[ex05_sequence_classification_GRU.SeqKind] of shape [10] of type "SeqKind" on backend "Cpu"
# 	Increasing	NonMonotonic	Increasing	Increasing	Increasing	Decreasing	Decreasing	Increasing	Increasing	NonMonotonic

# Epoch # 000. Accuracy: 95.163%
# Epoch # 001. Accuracy: 97.377%
# Epoch # 002. Accuracy: 97.740%
# Epoch # 003. Accuracy: 96.940%
# Epoch # 004. Accuracy: 97.380%
# Epoch # 005. Accuracy: 98.010%
# Epoch # 006. Accuracy: 98.700%
# Epoch # 007. Accuracy: 98.370%


###################
## Let's give our model some handcrafted tests
block:
  let exam = ctx.variable([
      [float32 0.10, 0.20, 0.30], # increasing
      [float32 0.10, 0.90, 0.95], # increasing
      [float32 0.45, 0.50, 0.55], # increasing
      [float32 0.10, 0.30, 0.20], # non-monotonic
      [float32 0.20, 0.10, 0.30], # non-monotonic
      [float32 0.98, 0.97, 0.96], # decreasing
      [float32 0.12, 0.05, 0.01], # decreasing
      [float32 0.95, 0.05, 0.07]  # non-monotonic
    ].toTensor.transpose.unsqueeze(2))

  let hidden0 = ctx.variable zeros[float32](Layers, 8, HiddenSize)

  let answer = model
                .forward(exam, hidden0)
                .value
                .softmax
                .argmax(axis = 1)
                .squeeze
                .astype(SeqKind)

  echo "\nTesting the model with:"
  echo exam.value.squeeze(2).transpose()
  echo "Answers:"
  echo answer.unsqueeze(1)
  # Tensor[ex05_sequence_classification_GRU.SeqKind] of shape [8, 1] of type "SeqKind" on backend "Cpu"
  # 	  Increasing|
  # 	  Increasing|
  # 	  Increasing|
  # 	  NonMonotonic|
  # 	  NonMonotonic|
  # 	  Increasing| <----- Wrong!
  # 	  Decreasing|
  # 	  NonMonotonic|

  # Almost there!
  # Next step: financial markets, let's collar those bears.
