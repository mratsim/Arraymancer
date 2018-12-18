# Create a Shakespeare AI
#
# Inspired by Andrej Karpathy http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# and https://github.com/karpathy/char-rnn

# This can learn anything text based, including code and LaTe paper ;).

# Note: training is quite slow on CPU, 30 min for my i5-5257U (2.7GHz dual-core Broadwell from 2015)
#
# Also parallelizing via OpenMP will slow down computation so don't use it.
# there is probably false sharing in the GRU layer, reshape layer or flatten_idx from Embedding.

# Remember that the network
#   - must learn, not to use !?;. everywhere
#   - must learn how to use spaces and new lines
#   - must learn capital letters
#   - must learn that character form words

# TODO: save/reload trained weights

import
  streams, os, random, times, strformat, algorithm, sequtils, tables,
  ../src/arraymancer

# ################################################################
#
#                     Environment constants
#
# ################################################################

# Printable chars - from Python: import string; string.printable
const IxToChar = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c"
type PrintableIdx = uint8

func genCharToIx(): Table[char, PrintableIdx] =
  result = initTable[char, PrintableIdx]()
  for idx, ch in IxToChar:
    result[ch] = PrintableIdx idx

const
  CharToIx = genCharToIx()
  VocabSize = IxToChar.len # Cardinality of the set of PrintableChars
  BatchSize = 100
  Epochs = 2000            # This take a long long time, I'm not even sure it converges
  Layers = 2
  HiddenSize = 100
  LearningRate = 0.01'f32
  EmbedSize = 100
  SeqLen = 200             # Characters sequences will be split in chunks of 200
  StatusReport = 200       # Report training status every x batches

# ################################################################
#
#                           Helpers
#
# ################################################################

func strToTensor(str: string|TaintedString): Tensor[PrintableIdx] =
  result = newTensor[PrintableIdx](str.len)

  # For each x in result, map the corresponding char index
  for i, val in result.menumerate:
    val = CharToIx[str[i]]

# Weighted random sampling / multinomial sampling
#   Note: during text generation we only work with
#         a batch size of 1 so for simplicity we use
#         seq and openarrays instead of tensors

func cumsum[T](x: openarray[T]): seq[T] =
  ## Cumulative sum of a 1D array/seq
  #
  # Note: this will have a proper and faster implementation for tensors in the future
  result = newSeq[T](x.len)
  result[0] = x[0]
  for i in 1 ..< x.len:
    result[i] = x[i] + result[i-1]

proc searchsorted[T](x: openarray[T], value: T, leftSide: static bool = true): int =
  ## Returns the index corresponding to where the input value would be inserted at.
  ## Input must be a sorted 1D seq/array.
  ## In case of exact match, leftSide indicates if we put the value
  ## on the left or the right of the exact match.
  ##
  ## This is equivalent to Numpy and Tensorflow searchsorted
  ## Example
  ##    [0, 3, 9, 9, 10] with value 4 will return 2
  ##    [1, 2, 3, 4, 5]             2 will return 1 if left side, 2 otherwise
  #
  # Note: this will have a proper and faster implementation for tensors in the future

  when leftSide:
    result = x.lowerBound(value)
  else:
    result = x.upperBound(value)

proc sample[T](probs: Tensor[T]): int =
  ## Returns a weighted random sample (multinomial sampling)
  ## from a 1D Tensor of probabilities.
  ## Probabilities must sum to 1 (normalised)
  ## For example:
  ##    - a Tensor of [0.1, 0.4, 0.2, 0.3]
  ##      will return 0 in 10% of cases
  ##                  1 in 40% of cases
  ##                  2 in 20% of cases
  ##                  3 in 30% of cases
  assert probs.rank == 1
  assert probs.is_C_contiguous
  assert probs.sum - 1.T < T(1e-5)

  # We use a separate RNG for our sampling
  var rng {.global.} = initRand(0xDEADBEEF)

  # We pass our 1D Tensor as an openarray to avoid copies
  let p = cast[ptr UncheckedArray[T]](probs.get_data_ptr)

  # Get a sample from an uniform distribution
  let u = T(rng.rand(1.0))

  # Get the Cumulative Distribution Function of our probabilities
  let cdf = cumsum p.toOpenArray(0, probs.shape[0] - 1)
  result = cdf.searchsorted(u, leftSide = false)

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

template weightInit(shape: varargs[int], init_kind: untyped): Variable =
  ## Even though we need to do the initialisation manually
  ## let's not repeat ourself too much.
  ctx.variable(
    init_kind(shape, float32),
    requires_grad = true
  )

template gruInit(shape: varargs[int]): Variable =
  let std = 1'f32 / sqrt(HiddenSize.float32)
  ctx.variable(
    randomTensor(shape, -std .. std),
    requires_grad = true
  )

proc newShakespeareNet[TT](ctx: Context[TT]): ShakespeareNet[TT] =
  ## Initialise a model with random weights.
  ## Normally this is done for you with the `network` macro

  # Embedding layer
  #   Input: [SeqLen, BatchSize, VocabSize]
  #   Output: [SeqLen, BatchSize, EmbedSize]
  result.encoder.weight = ctx.variable(
    # initialisation bench https://arxiv.org/pdf/1711.09160.pdf
    # Convergence is **VERY** sensitive, I can't reproduce the paper.
    # Best in our case is mean = 0, std = 1.
    randomNormalTensor(VocabSize, EmbedSize, 0'f32, 1'f32),
    requires_grad = true
  )

  # GRU layer
  #   Input:   [SeqLen, BatchSize, EmbedSize]
  #   Hidden0: [Layers, BatchSize, HiddenSize]
  #
  #   Output:  [SeqLen, BatchSize, HiddenSize]
  #   HiddenN: [Layers, BatchSize, HiddenSize]

  # GRU have 5 weights/biases that can be trained.
  # This initialisation is normally hidden from you.
  result.gru.W3s0 = gruInit(            3 * HiddenSize,      EmbedSize)
  result.gru.W3sN = gruInit(Layers - 1, 3 * HiddenSize,     HiddenSize)
  result.gru.U3s  = gruInit(    Layers, 3 * HiddenSize,     HiddenSize)

  result.gru.bW3s = ctx.variable(zeros[float32](Layers, 1, 3 * HiddenSize), requires_grad = true)
  result.gru.bU3s = ctx.variable(zeros[float32](Layers, 1, 3 * HiddenSize), requires_grad = true)

  # Linear layer
  #   Input: [BatchSize, HiddenSize]
  #   Output: [BatchSize, VocabSize]
  result.decoder.weight = weightInit(VocabSize, HiddenSize, kaiming_normal)
  result.decoder.bias   = ctx.variable(zeros[float32](1, VocabSize), requires_grad = true)

# Some wrappers to pass the layer weights
proc encode[TT](model: ShakespeareNet[TT], x: Tensor[PrintableIdx]): Variable[TT] =
  embedding(x, model.encoder.weight)

proc gru_forward(model: ShakespeareNet, x, hidden0: Variable): tuple[output, hiddenN: Variable] =
  gru(
    x, hidden0,
    model.gru.W3s0, model.gru.W3sN,
    model.gru.U3s,
    model.gru.bW3s, model.gru.bU3s
  )

proc decode(model: ShakespeareNet, x: Variable): Variable =
  linear(x, model.decoder.weight, model.decoder.bias)

proc forward[TT](
        model: ShakespeareNet[TT],
        input: Tensor[PrintableIdx],
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
        data: Tensor[PrintableIdx],
        seq_len, batch_size: int,
        rng: var Rand
      ): tuple[input, target: Tensor[PrintableIdx]] =
  ## Generate a set of input sequences of length `seq_len`
  ## and the immediate following `seq_len` characters to predict
  ## Sequence are extracted randomly from the whole text.
  ## i.e. If we have ABCDEF input data
  ##         we can have ABC input
  ##                 and BCD target

  result.input = newTensor[PrintableIdx](seq_len, batch_size)
  result.target = newTensor[PrintableIdx](seq_len, batch_size)

  let length = data.shape[0]
  for batch_id in 0 ..< batch_size:
    let start_idx = rng.rand(0 ..< (length - seq_len))
    let end_idx = start_idx + seq_len + 1
    result.input[_, batch_id] =  data[start_idx ..< end_idx - 1]
    result.target[_, batch_id] = data[start_idx + 1 ..< end_idx]

proc train[TT](
        ctx: Context[TT],
        model: ShakespeareNet[TT],
        optimiser: var Optimizer[TT],
        input, target: Tensor[PrintableIdx]): float32 =
  ## Train a model with an input and the corresponding characters to predict.
  ## Return the loss after the training session

  let seq_len = input.shape[0]
  var hidden = ctx.variable zeros[float32](Layers, BatchSize, HiddenSize)

  # We will cumulate the loss before backpropping at once
  # to avoid teacher forcing bias. (Adjusting weights just before the next char)
  var seq_loss = ctx.variable(zeros[float32](1), requires_grad = true)

  for char_pos in 0 ..< seq_len:
    var output: Variable[TT]
    (output, hidden) = model.forward(input[char_pos, _], hidden)
    let batch_loss = output.sparse_softmax_cross_entropy(target[char_pos, _].squeeze(0))

    seq_loss = seq_loss + batch_loss

  seq_loss.backprop()
  optimiser.update()

  result = seq_loss.value[0] / seq_len.float32

# ################################################################
#
#                     Text generator
#
# ################################################################

proc gen_text[TT](
        ctx: Context[TT],
        model: ShakespeareNet[TT],
        seed_chars = "Wh", # Why, What, Who ...
        seq_len = SeqLen,
        temperature = 0.8'f32
      ): string =
  ## Inputs:
  ##   - Model:       the trained model
  ##   - seed_chars:  A string to initialise the generator state and get it running
  ##   - seq_len:     Generation are done by chunk of `seq_len` length
  ##   - temperature: The conservative <--> diversity scale of the generator.
  ##                  Value between 0 and 1, near 0 it will be conservative,
  ##                  near 1 it will take liberties but make more mistakes.

  ctx.no_grad_mode:
    var
      hidden = ctx.variable zeros[float32](Layers, 1, HiddenSize) # batch_size is now 1
    let primer = seed_chars.strToTensor().unsqueeze(1) # Shape [seq_len, 1]

    # Create a consistent hidden state
    for char_pos in 0 ..< primer.shape[0] - 1:
      var (_, hidden) = model.forward(primer[char_pos, _], hidden)


    result = seed_chars

    # And start from the last char!
    var input = primer[^1, _]
    var output: Variable[TT]

    for _ in 0 ..< seq_len:
      (output, hidden) = model.forward(input, hidden)
      # output is of shape [BatchSize, VocabSize] with BatchSize = 1

      # Go back in the tensor domain
      var preds = output.value

      # We scale by the temperature first.
      preds ./= temperature
      # Get a probability distribution.
      let probs = preds.softmax().squeeze(0)

      # Sample and append to the generated chars
      let ch_ix = probs.sample().PrintableIdx
      result &= IxToChar[ch_ix]

      # Next char
      input = newTensor[PrintableIdx](1, 1)
      input[0, 0] = ch_ix

# ################################################################
#
#                     User interaction
#
# ################################################################

proc main() =
  # Parse the input file
  let filePath = paramStr(1).string
  let txt_raw = readFile(filePath)

  echo "Checking the first hundred characters of your file"
  echo txt_raw[0 .. 100]
  echo "\n####\nStarting training\n"

  # For our need in gen_training_set, we reshape it from [nb_chars] to [nb_chars, 1]
  let txt = txt_raw.strToTensor.unsqueeze(1)

  # Make the results reproducible
  randomize(0xDEADBEEF) # Changing that will change the weight initialisation

  # Create our autograd context that will track deep learning operations applied to tensors.
  let ctx = newContext Tensor[float32]

  # Build our model and initialize its weights
  let model = ctx.newShakespeareNet()

  # Optimizer
  # let optim = model.optimizerSGD(learning_rate = LearningRate)
  var optim = model.optimizerAdam(learning_rate = LearningRate)

  # We use a different RNG for seq split
  var split_rng = initRand(42)

  # Start our time counter
  let start = epochTime()

  for epoch in 0 ..< Epochs:
    let (input, target) = gen_training_set(txt, SeqLen, BatchSize, split_rng)
    let loss = ctx.train(model, optim, input, target)

    if epoch mod StatusReport == 0:
      let elapsed = epochTime() - start
      echo &"\n####\nTime: {elapsed:>4.4f} s, Epoch: {epoch}/{Epochs}, Loss: {loss:>2.4f}"
      echo "Sample: "
      echo ctx.gen_text(model, seq_len = 100)

  echo "\n##########\nTraining end. Generating 4000 characters Shakespeare masterpiece in 3. 2. 1...\n\n"
  echo ctx.gen_text(model, seq_len = 4000)

main()

# ##################################################
# Output

# $  ./build/ex06 examples/ex06_shakespeare_input.txt
# Checking the first hundred characters of your file
# First Citizen:
# Before we proceed any further, hear me speak.
#
# All:
# Speak, speak.
#
# First Citizen:
# You

# ####
# Starting training


# ####
# Time: 0.8500 s, Epoch: 0/2000, Loss: 4.6268
# Sample:
# Whwno\<^,[
# [;c@HmN,FMf-DMZd7rSTM|C'PfuMlW7
#               Hhy;
# dM<v
# |8Z<%}Sv/X[\6sA.>GSNSB
#                       TReNt<>%>x`

# ####
# Time: 150.7535 s, Epoch: 200/2000, Loss: 1.5315
# Sample:
# Whess! and
# Whily cannamed in the lastider, you good
# this think
# What, ere to you? To me alfatio' worst

# ####
# Time: 313.9455 s, Epoch: 400/2000, Loss: 1.4023
# Sample:
# Whe in joy
# call hears an arring,
# For the depart:
# Whilst dye find that the while a thou here, a heartio

# ####
# Time: 467.2250 s, Epoch: 600/2000, Loss: 1.4058
# Sample:
# Wheer and words thence to sir,
# Even I do notgices the numbers to awe of him?
#
# PETRUCHIO:
# Moor of our l

# ####
# Time: 625.6501 s, Epoch: 800/2000, Loss: 1.4030
# Sample:
# Whic deglinnes; as fings old,
# Whilstress.
#
# CURTIS:
# Some, company, though of God's heart breath.
#
# Sheph

# ####
# Time: 772.9566 s, Epoch: 1000/2000, Loss: 1.3489
# Sample:
# Whang hild, pult, my two thou
# diecin-started leave to-morroved: my might,
# For Fonce; who change you ar

# ####
# Time: 919.4623 s, Epoch: 1200/2000, Loss: 1.3602
# Sample:
# Whis mole: then,
# I know her, soul to report in souls, transporther:
# As with thy other coyll'd speak't.

# ####
# Time: 1071.9547 s, Epoch: 1400/2000, Loss: 1.3084
# Sample:
# Wher I have stand
# hod untires for thy brother that many unmoon?
#
# Nurse:
# Gentle live, the temple of fig

# ####
# Time: 1238.2013 s, Epoch: 1600/2000, Loss: 1.3803
# Sample:
# When, but hollow been he, ladoming
# Whinks my state to noth'd it o' the hist:
# I snoos be to your master

# ####
# Time: 1394.6621 s, Epoch: 1800/2000, Loss: 1.3630
# Sample:
# Wh were cannot caniders are well him:
# he half and light: Thank you will my sea;
# Lest you do know you a

# ##########
# Training end. Generating 4000 characters Shakespeare masterpiece in 3. 2. 1...


# Whollever: and forbid shall be dew
# Gto coffents untendred foul charge.
#
# AUFIDIUS:
# But forgher; an heart to seizing to suppose,
# I can hown to provide's of your daughter,
# Whink Richard me, good heard why too furthen,
# To stay and considents?
#
# PERDITA:
# Let him fee of children and England's good father
# Town's crowling-contigns call with his king:
# Alas, in sound against the majesty.
#
# SICINIUS:
# No; and proud all the villain of Yorket,
# With a thing bear obey, thou canst it comes:
# Let me not underneme usested old country.
# Sy in Bament, your face honour'd up all begetake you,
# For I her very truth, I shall comes falm only;
# And not shield to him of priless to the Tower:
# Yes in the honour.
#
# First Murderer:
# Hark. Now, if you conseings they do? O, come;
# Or shall pack?
#
# BRUTUS:
# It one noble:
# That mother Trobleing! O, the seet, for else
# But then speak mistaked the court-tray'd,
# Course to strike.
#
# RATCLIFF:
# A divide gatler, for there he such sorrow from strong.
#
# KATHARINA:
# Whether will go well, no doubt is English heart:
# But seem to thigh is my soul, nigh, that loir,
# Who will possessed sore homeft things with a powry
# Of a wretched the heart sullaff, do your faired all.
# Ah, whose power cord hither is the chemies, of the holy Roman:
# But no ream, else carries me deteet! Bold Neassens,
# Tybals, I'll come are, we are raven their newsion
# The cancation, even swear amanother:
# Aso faithly married sorrow.
#
# GONZALO:
# I post caseditities to so feard that I said
# My lord. They little to your voice:
# I do bid love me on the condition to
# me a fathers sacred still death,
# Dusty fair, therefore whom I loves hober it hang the father's dear house:
# I say for he speak on the old duiting.
# I have det-both a shall give, this dost from this?
#
# LUCENTIO:
# Nay, how'' mine pardon.
#
# ISABELLA:
# Who mea her of Lancaster from say
# That have put a criptest me no thinghofs of their face;
# Was this all in sum seen such a fled under some father,
# That accusate.
# Where imend rare, senall's obey.
#
# First Senator:
# Good most or this addle they have anquer tell me in:
# A heart the writ't think honour durinder;
# Farewell and Some strange,
# He hath some hilless watched from one facts:
# Thereboes, a Roman: but seem thine; one wife
# Couts from his hearthd.
#
# LUCIO:
# Good my lord,
# Hath say; and to me, we gosted to a posfrlought'st aigess agre mouded poison:
# Grom father! while Blifts he look as good father's
# hear not hear him; go; if he done! let me resign sold they
# condead good provides for she say.
#
# KING RICHARD III:
# How now!
#
# TRANIO:
# Her save just that due we would, my lord,
# I would in bring from us these twife free;
# For I be gone! the mortal both, in last,--
# That's enmitation here yares my greets
# Which when me to Henry what thou shalt contempling.
# Some orly, and to the call of the sufficer
# Yet worthins, surned to give the geneing of death.
#
# BUCKINGHAM:
# Farecch, you title consumion, like Plantage!
#
# GREMIO:
# He is a ratch that a more to high loved it,
# And I am be good unpleasune, all all
# From from him to his mother and protest
# How call as offer spit amazemore should sught.
#
# ELBOW:
# They nature to thus now ye,
# And Oxford, and farm to epther it is sagly:
# Inch, thus ready strengthen sure' to consul,
# You am whifty queel the thing affections. Is Let why many gaunt
# The good many mother honour,
# Inch whom his woman! and you are regard strike
# What to to the fear the slipe passate no offence bear
# And your chargaretion; who should run but uncarreless?
# If thou shalt I the most devil in a grefence!
# Though I'll know it out of the king lie?
#
# TRANIO:
# She! hear?
#
# LEONTES:
# Break the masters awn is grief I live to perfectic is fright at me;
# And dear worthy food carnat in the still not
# will seal bite fier match shuk're and were.
#
# DUCHESS OF YORK:
# A Norfolk; thou wilt thy spirit; which I do not
# Happiness.
#
# POLINE:
# Inkence it, and gentleman number veil thankty on
# Did to fails of your hotsed to go.
#
# BRUTUS:
# Bear me last, I would help obedient--ondery to long your timber
# Thy as a valling to know'scall.
#
# CORIOLANUS:
# My lord stone's dest
