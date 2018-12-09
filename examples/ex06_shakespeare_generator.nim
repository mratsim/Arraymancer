# Create a Shakespeare AI
# Inspired by Andrej Karpathy http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# and https://github.com/karpathy/char-rnn

import
  streams, os, random,
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

type ShakespeareNet[TT] = object
  # Embedding weight = Encoder
  encoder_w: Variable[TT]

  # GRU RNN = Internal representation
  # GRU weights, normally this is hidden from you (see example 5), RNNs are very involved
  # but without the NN mini-lang I unfortunately need to expose this.
  gru_W3s0, gru_W3sN, gru_U3s, gru_bW3s, bU3s: Variable[TT]

  # Linear layer weight and bias = Decoder
  decoder_w: Variable[TT]
  decoder_b: Variable[TT]

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
  result.encoder_w = weightInit(VocabSize, EmbedSize)

  # GRU layer
  #   Input:   [SeqLen, BatchSize, EmbedSize]
  #   Hidden0: [Layers, BatchSize, HiddenSize]
  #
  #   Output:  [SeqLen, BatchSize, HiddenSize]
  #   HiddenN: [Layers, BatchSize, HiddenSize]

  # GRU have 5 weights/biases that can be trained. This initialisation is normally hidden from you.
  result.gru_W3s0 = weightInit(3 * HiddenSize, EmbedSize)
  result.gru_W3sN = weightInit(Layers - 1, 3 * HiddenSize, HiddenSize)
  result.gru_U3s = weightInit(Layers, 3 * HiddenSize, HiddenSize)
  result.gru_bW3s = weightInit(Layers, 1, 3 * HiddenSize)
  result.gru_bU3s = weightInit(Layers, 1, 3 * HiddenSize)

  # Linear layer
  #   Input: [BatchSize, HiddenSize]
  #   Output: [BatchSize, VocabSize]
  result.decoder_w = weightInit(VocabSize, HiddenSize)
  result.decoder_b = weightInit(        1, VocabSize)

proc main() =
  # Parse the input file
  let filePath = paramStr(1).string
  let txt = readFile(filePath)

  echo "Checking the Tensor of the first hundred characters of your file"
  echo txt.strToTensor[0 .. 100]

  # Make the results reproducible
  randomize(0xDEADBEEF) # Changing that will change the text generated

  # Create our autograd context that will track deep learning operations applied to tensors.
  let ctx = newContext Tensor[float32]


main()
