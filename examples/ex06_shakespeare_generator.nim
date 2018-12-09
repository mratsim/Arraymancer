# Create a Shakespeare AI
# Inspired by Andrej Karpathy http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# and https://github.com/karpathy/char-rnn

import
  streams, os, random, times, strformat,
  ../src/arraymancer

# ################################################################
#
#                     Environment constants
#
# ################################################################

const
  HiddenSize = 100
  BatchSize = 100
  Epochs = 2000
  Layers = 2
  LearningRate = 0.01'f32
  VocabSize = 255
  EmbedSize = 100
  SeqLen = 200        # Characters sequences will be split in chunks of 200
  StatusReport = 200  # Report training status every x batches

# ################################################################
#
#                           Helpers
#
# ################################################################

func strToTensor(x: TaintedString): Tensor[char] =
  ## By default unsafe string (read from disk)
  ## are protected by TaintedString and need explicit string conversion (no runtime cost)
  ## before you can handle them. This is to remind you that you must vlidate the string
  ##
  ## Here we will take several shortcuts, we assume that the string is safe.
  ## and we will also cast it to a sequence of characters
  ##
  ##     (Don't do this at home, this rely on Nim sequence of chars having the same representation as strings
  ##      in memory and the garbage collector
  ##      which is approximately true, there is an extra hidden '\0' at the end of Nim strings)
  ##
  ## before converting it to a Tensor of char.
  result = cast[seq[char]](x).toTensor()

# ################################################################
#
#                        Neural network model
#
# ################################################################

# Create our model and the weights to train
#
#     Due too much convenience, the neural net declaration mini-language
#     used in examples 2 to 5
#     only accepts Variable[Tensor[float32]] (for a Tensor[float32] context)
#     but we also need a Tensor[char] input for embedding.
#     So much for trying to be too clever ¯\_(ツ)_/¯.
#
#     Furthermore, you don't have flexibility in the return variables
#     while we need to also return the hidden state of our text generation model.
#
#     So we need to do everything manually...

# We use a classic Encoder-Decoder architecture, with text encoded into an internal representation
# and then decoded back into text.
# So we need to train the encoder, the internal representation and the decoder.

type
  ## The following is normally unnecessary when using the NN mini-lang
  LinearLayer[TT] = object
    weight: Variable[TT]
    bias: Variable[TT]
  GRULayer[TT] = object
    W3s0, W3sN: Variable[TT]
    U3s: Variable[TT]
    bW3s, bU3s: Variable[TT]
  EmbeddingLayer[TT] = object
    weight: Variable[TT]

  ShakespeareNet[TT] = object
    # Embedding weight = Encoder
    encoder: EmbeddingLayer[TT]
    # GRU RNN = Internal representation
    gru: GRULayer[TT]
    # Linear layer weight and bias = Decoder
    decoder: LinearLayer[TT]

template weightInit(shape: varargs[int]): untyped {.dirty.} =
  ## Even though we need to do the initialisation manually
  ## let's not repeat ourself too much.
  ctx.variable(
    randomTensor(shape, -0.5'f32 .. 0.5'f32),
    requires_grad = true
  )

proc newShakespeareNet[TT](ctx: Context[TT]): ShakespeareNet[TT] =
  ## Initialise a model with random weights.
  ## Normally this is done for you with the `network` macro

  # Embedding layer
  #   Input: [SeqLen, BatchSize, VocabSize]
  #   Output: [SeqLen, BatchSize, EmbedSize]
  result.encoder.weight = weightInit(VocabSize, EmbedSize)

  # GRU layer
  #   Input:   [SeqLen, BatchSize, EmbedSize]
  #   Hidden0: [Layers, BatchSize, HiddenSize]
  #
  #   Output:  [SeqLen, BatchSize, HiddenSize]
  #   HiddenN: [Layers, BatchSize, HiddenSize]

  # GRU have 5 weights/biases that can be trained. This initialisation is normally hidden from you.
  result.gru.W3s0 = weightInit(            3 * HiddenSize,     EmbedSize)
  result.gru.W3sN = weightInit(Layers - 1, 3 * HiddenSize,     HiddenSize)
  result.gru.U3s  = weightInit(    Layers, 3 * HiddenSize,     HiddenSize)
  result.gru.bW3s = weightInit(    Layers,              1, 3 * HiddenSize)
  result.gru.bU3s = weightInit(    Layers,              1, 3 * HiddenSize)

  # Linear layer
  #   Input: [BatchSize, HiddenSize]
  #   Output: [BatchSize, VocabSize]
  result.decoder.weight = weightInit(VocabSize, HiddenSize)
  result.decoder.bias   = weightInit(        1, VocabSize)

# Some wrappers to pass the layer weights
proc encode[TT](model: ShakespeareNet[TT], x: Tensor[char]): Variable[TT] =
  embedding(x, model.encoder.weight)

proc gru_forward(model: ShakespeareNet, x, hidden0: Variable): tuple[output, hiddenN: Variable] =
  gru(
    x, hidden0,
    Layers,
    model.gru.W3s0, model.gru.W3sN,
    model.gru.U3s,
    model.gru.bW3s, model.gru.bU3s
  )

proc decode(model: ShakespeareNet, x: Variable): Variable =
  linear(x, model.decoder.weight, model.decoder.bias)

proc forward[TT](
        model: ShakespeareNet[TT],
        input: Tensor[char],
        hidden0: Variable[TT]
      ): tuple[output, hidden: Variable[TT]] =

  let encoded = model.encode(input)
  let (output, hiddenN) = model.gru_forward(encoded, hidden0)

  # result.output is of shape [Sequence, BatchSize, HiddenSize]
  # In our case the sequence is 1 so we can simply flatten
  let flattened = output.reshape(output.value.shape[1], HiddenSize)

  result.output = model.decode(flattened)
  result.hidden = hiddenN

# ################################################################
#
#                        Training
#
# ################################################################

proc gen_training_set(
        data: Tensor[char],
        seq_len, batch_size: int,
        rng: var Rand
      ): tuple[input, target: Tensor[char]] =
  ## Generate a set of input sequences of length `seq_len`
  ## and the immediate following `seq_len` characters to predict
  ## Sequence are extracted randomly from the whole text.
  ## i.e. If we have ABCDEF input data
  ##         we can have ABC input
  ##                 and BCD target

  # For input we order in [seq_len, batch_size] as this is faster with RNNs
  result.input = newTensor[char](seq_len, batch_size)

  # For target we order with [batch_size, seq_len] as this is the natural
  # result from the model, and also what is expected by loss functions
  result.target = newTensor[char](batch_size, seq_len)

  let length = data.shape[0]
  for batch_id in 0 ..< batch_size:
    let start_idx = rng.rand(0 ..< (length - seq_len))
    let end_idx = start_idx + seq_len + 1
    # [seq_len, batch_size]
    result.input[_, batch_id] =  data[start_idx ..< end_idx - 1]
    # [batch_size, seq_len]
    result.target[batch_id, _] = data[start_idx + 1 ..< end_idx].transpose()

proc train[TT](
        ctx: Context[TT],
        model: ShakespeareNet[TT],
        optimiser: Sgd[TT],
        input, target: Tensor[char]): float32 =
  ## Train a model with an input and the corresponding characters to predict.
  ## Return the loss after the training session

  let seq_len = input.shape[0]
  let hidden0 = ctx.variable zeros[float32](Layers, BatchSize, HiddenSize)

  # We will cumulate the loss before backpropping at once
  # to avoid teacher forcing bias. (Adjusting weights just before the next char)
  var loss = ctx.variable(zeros[float32](1), requires_grad = true)

  for char_pos in 0 ..< seq_len:
    let (output, hidden) = model.forward(input[char_pos, _], hidden0)
    loss = loss + output.sparse_softmax_cross_entropy(target) # In-place operations are tricky in an autograd

  loss.backprop()
  optimiser.update()

  result = loss.value[0] / seq_len.float32

# ################################################################
#
#                     User interaction
#
# ################################################################

proc main() =
  # Parse the input file
  let filePath = paramStr(1).string
  let txt_raw = readFile(filePath).strToTensor

  echo "Checking the Tensor of the first hundred characters of your file"
  echo txt_raw[0 .. 100]

  # For our need in gen_training_set, we reshape it from [nb_chars] to [nb_chars, 1]
  let txt = txt_raw.unsqueeze(1)

  # Make the results reproducible
  randomize(0xDEADBEEF) # Changing that will change the weight initialisation

  # Create our autograd context that will track deep learning operations applied to tensors.
  let ctx = newContext Tensor[float32]

  # Build our model and initialize its weights
  let model = ctx.newShakespeareNet()

  # Stochastic Gradient Descent (API will change)
  let optim = model.optimizerSGD(learning_rate = LearningRate)

  # We use a different RNG for seq split
  var split_rng = initRand(42)

  # Start our time counter
  let start = epochTime()

  for epoch in 0 ..< Epochs:
    let (input, target) = gen_training_set(txt, SeqLen, BatchSize, split_rng)
    let loss = ctx.train(model, optim, input, target)

    if epoch mod StatusReport == 0:
      let elapsed = epochTime() - start
      echo &"Time: {elapsed:>4.4f} s, Epoch: {epoch}/{Epochs}, Loss: {loss:>2.4f}"
      # TODO: example sentence generated

main()
