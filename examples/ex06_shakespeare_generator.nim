# Create a Shakespeare AI
#
# Inspired by Andrej Karpathy http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# and https://github.com/karpathy/char-rnn

# Note: training is quite slow on CPU
#       furthermore we probably need a better optimizer than SGD
# I guess time to implement GPU RNNs and Adam/RMSprop
#
# Also parallelizing via OpenMP will slow down computation so don't use it.
# there is probably false sharing in the GRU layer, reshape layer or flatten_idx from Embedding.

# Also I didn't have the patience to tune the parameters so I'm not sure it converges.
# There are plenty of reports of people struggling
# to make Tensorflow and PyTorch char-rnn converge so TODO

# Remember that the network
#   - must learn, not to use !?;. everywhere
#   - must learn how to use spaces and new lines
#   - must learn capital letters
#   - must learn that character form words

# TODO: save/reload trained weights

import
  streams, os, random, times, strformat, algorithm, sequtils,
  ../src/arraymancer

# ################################################################
#
#                     Environment constants
#
# ################################################################

# Printable chars are \n (ASCII code 10)
# and Space (ASCII code 32) to tilde (ASCII code 126)
const PrintableChars = {'\n'} + {' ' .. '~'}
type PrintableIdx = uint8

func genCharsMapping(): tuple[
          charToIx: array[char, PrintableIdx],
          ixToChar: array[PrintableChars.card, char]
        ] =

  # For debug we put ¿ (ASCII code 168) character as default value in charToIx
  result.charToIx.fill(168)

  var idx = 0'u8
  for ch in PrintableChars:
    result.charToIx[ch] = idx
    result.ixToChar[idx] = ch
    inc idx

const
  Mappings = genCharsMapping()
  VocabSize = PrintableChars.card # Cardinality of the set of PrintableChars
  BatchSize = 32
  Epochs = 100_000    # This take a long long time, I'm not even sure it converges
  Layers = 2
  HiddenSize = 128
  LearningRate = 0.01'f32
  EmbedSize = 100
  SeqLen = 200        # Characters sequences will be split in chunks of 200
  StatusReport = 200  # Report training status every x batches

# ################################################################
#
#                           Helpers
#
# ################################################################

func strToTensor(str: string|TaintedString): Tensor[PrintableIdx] =
  result = newTensor[PrintableIdx](str.len)

  # For each x in result, map the corresponding char index
  for i, val in result.menumerate:
    val = Mappings.charToIx[str[i]]

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

  # GRU have 5 weights/biases that can be trained.
  # This initialisation is normally hidden from you.
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
  let hidden0 = ctx.variable zeros[float32](Layers, BatchSize, HiddenSize)

  # We will cumulate the loss before backpropping at once
  # to avoid teacher forcing bias. (Adjusting weights just before the next char)
  var seq_loss = ctx.variable(zeros[float32](1), requires_grad = true)

  for char_pos in 0 ..< seq_len:
    let (output, hidden) = model.forward(input[char_pos, _], hidden0)
    let batch_loss = output.sparse_softmax_cross_entropy(target[char_pos, _].squeeze(0))

    # Inplace `+=` is usually tricky with autograd
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

      # Instead of softmaxing, we sample from the network
      # as if it was a multinomial distribution.

      # We scale by the temperature first.
      preds ./= temperature
      # Get a probability distribution.
      let probs = preds.softmax().squeeze(0)

      # Sample and append to the generated chars
      let ch_ix = probs.sample().PrintableIdx
      result &= Mappings.ixToChar[ch_ix]

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

main()

# ##################################################
# Output - Adam, learning rate 0.01 - Gradients exploded ...

# Checking the first hundred characters of your file
# First Citizen:
# Before we proceed any further, hear me speak.

# All:
# Speak, speak.

# First Citizen:
# You

# ####
# Starting training


# ####
# Time: 0.5359 s, Epoch: 0/100000, Loss: 5.1517
# Sample:
# WhIGTuuuerwH:rrrrD=I;edI;***Z\D8trrrNjjII-uti8rD6DWD\8d3\}22Ps|D^kZzf6[tttttt[kr6nkkkkkkkbzkd5\IqqJq.r

# ####
# Time: 82.5663 s, Epoch: 200/100000, Loss: 2.4458
# Sample:
# Wheaata ttyyootyr eur iufyoorssay  oif rioooootrvanyaott urteryto s?iatoonyfsotoiurrror eaitrrusotitre

# ####
# Time: 158.4338 s, Epoch: 400/100000, Loss: 2.4572
# Sample:
# Wheeaatenrt aarto? ory ay ttorneteeatt, antt ttrytt, t  sy t tr tut ttheaty ttte utennt?nt   't s si t

# ####
# Time: 234.2593 s, Epoch: 600/100000, Loss: 2.4647
# Sample:
# Wheaivatstytraatrrioeyirsttrt eatttryy otrr  oarryrirorureotrrrrd rrriraorervotontrosoirtrrarne, adait

# ####
# Time: 310.1731 s, Epoch: 800/100000, Loss: 2.4936
# Sample:
# Whe eaaiasstatt  ttatstheaisaotortro aouirriveiitnirrandadrry eiarresoattreryf aroottyot renootrttrrrr

# ####
# Time: 386.0483 s, Epoch: 1000/100000, Loss: 2.5014
# Sample:
# Wh aiwar  tiotetooowotwartertvaia rottrerreoildaoddvyaeoriaarldees riaotiyrrarroaoirrsess iairnttr rar

# ####
# Time: 460.4311 s, Epoch: 1200/100000, Loss: 2.4683
# Sample:
# Wheitee eteeaa  wcsatteeaiarsttr ttotoatroyaratrrr ae ty rtrrtrarrry ottrtyyo'toooo eoatyrrrayoraryaye

# ####
# Time: 533.2700 s, Epoch: 1400/100000, Loss: 2.4537
# Sample:
# Whaeitioostiotioeuat rroretriraievoiuoarrssssss atoo rrotososatats s oooonoiootoottorrrrtrerrrororrari

# ####
# Time: 606.2192 s, Epoch: 1600/100000, Loss: 2.4590
# Sample:
# Whe   anoiay nas, a ata t
#  si tttuitttaioutstetfestatofft ofstouateea itatsstost ataitstttuteeiarsat

# ####
# Time: 679.5429 s, Epoch: 1800/100000, Loss: 2.4710
# Sample:
# Whtoea o  oorofa r  feas saatarroetane meafat  oetae,rt ,y aot ttottref nyoo  suofisfto rfoootofyofoo

# ####
# Time: 752.2276 s, Epoch: 2000/100000, Loss: 2.4679
# Sample:
# Whi ofua e  ntratn ooroa nks , aiwtnutono naiteotan s so ceetehaaasaueaa sstsso s atseaoaasaosaf wfaaa

# ####
# Time: 824.9055 s, Epoch: 2200/100000, Loss: 2.4722
# Sample:
# Whetieeeartt  ern etoesaoamat esaftt  tuitttttrerrareyset taot, tra rtrottsuas ss sse trsartosms otsua

# ####
# Time: 897.6905 s, Epoch: 2400/100000, Loss: 2.4571
# Sample:
# Wheo iett  e atorteo  .   iasositaalalllrianyootokfen oe a stmoe  ta tt t astuto  at  tooest  sseotse

# ####
# Time: 970.3663 s, Epoch: 2600/100000, Loss: 2.4791
# Sample:
# Who ur seo aet  e so a iaiicesiolteeeeea koea erieeroeeeeisaasetasasnlte  ssasfmeseost  ee eoteaaeeasn

# ####
# Time: 1043.0339 s, Epoch: 2800/100000, Loss: 2.4477
# Sample:
# Whiae eetroorri oosotirruo,  s etoise toe a    , ti othooosuiantalaito'ienasooaaloosooots,ooo oo'utt

# ####
# Time: 1115.7065 s, Epoch: 3000/100000, Loss: 2.4734
# Sample:
# Whi aooocoertaereeeayte esaaoea ate  e   aoateraea ireo  ae eaat aanttceyeea't  tstt o eitot  tteloar,

# ####
# Time: 1188.4677 s, Epoch: 3200/100000, Loss: 2.4743
# Sample:
# Whiaur  tcaeireoaealaitrelleoio mrorrlereee anreaoorrrreroerwioarrotaiseair inerasenir eessaettee eean

# ####
# Time: 1261.3639 s, Epoch: 3400/100000, Loss: 2.4448
# Sample:
# Whea aio mi  seceaarlekole'erlts slsaos tie t, rttet, 'st ie rott rtreet eatoeares,   ss  trirshaiotsu

# ####
# Time: 1333.9971 s, Epoch: 3600/100000, Loss: 2.4636
# Sample:
# Wh ao  a rt oo  oo -t n   '  m saeooo aauin  fl eta  a ssi s tsitstt u   t lttr, s  otiooueuuseusf, at

# ####
# Time: 1406.7137 s, Epoch: 3800/100000, Loss: 2.4666
# Sample:
# Whe o arieao  it eanado ore os seou e seeeeot ook esnti stle   tt asostcty iet,     tes e  tats  tnhs'

# ####
# Time: 1479.4245 s, Epoch: 4000/100000, Loss: 2.4885
# Sample:
# Wha yeoee  a  ilear ss e , eieoe i eselaot sst  oo tluose mo w n mserawoa msmi aimammmmm siyal eooes e

# ####
# Time: 1552.1464 s, Epoch: 4200/100000, Loss: 2.4736
# Sample:
# Wheeoee i soltveseea ssses,e a,e  t  e  site   heieseiis eceaiaaseeiataanstitciosetaat isatastettiatlo

# ####
# Time: 1624.8478 s, Epoch: 4400/100000, Loss: 2.4688
# Sample:
# Wha;a alorae  e  oa oer riwiaernoenoonotss oem ,t n  tt  ooote otoete ssfoote o'eseeeoulsa,  t sst t

# ####
# Time: 1697.5983 s, Epoch: 4600/100000, Loss: 2.4671
# Sample:
# Whiata tr r oroo toeor o tooooo oouo oieaia ortmomeeaaoouoano aaaoroofss sas, ,oo ooa'soaooooooosnnaoo

# ####
# Time: 1770.3448 s, Epoch: 4800/100000, Loss: 2.4674
# Sample:
# Whe.  iee au  ils  t   t t   rsofirteeai ssalsst t aiteefittee e      i ssttttufl est  istit'aset ofoo

# ####
# Time: 1843.0177 s, Epoch: 5000/100000, Loss: 2.4602
# Sample:
# Wheie-lense  l e  as  tnoos  t t  soee oo  eooosoenogalasigteysatie   'ssssss ,  te   ltt t sgiet  ,s

# ####
# Time: 1916.1867 s, Epoch: 5200/100000, Loss: 2.4519
# Sample:
# Whoieeastt, o
# aea  a  oraoasos ie,   e    ,   eo ioiosmoamtea a y oee  a ia osoaitt oone oo'  sea ut

# ####
# Time: 1988.8650 s, Epoch: 5400/100000, Loss: 2.4603
# Sample:
# Wh iarte oeweeoooeoeroetroo   oooeeenkooeoe oo,  iat'oedoe  noo aooia toofoo yeeofo ooafefoeootaoo s

# ####
# Time: 2061.4777 s, Epoch: 5600/100000, Loss: 2.4519
# Sample:
# Whoyi ceeaarooittoauaucteateeeeteettcettac lsee.ecea  acctleo eateele eeeeeaeteteeeelerateceeeeeeeeaee

# ####
# Time: 2134.1670 s, Epoch: 5800/100000, Loss: 2.4581
# Sample:
# Wheieaeeafreieeeeeeeeaaese, aeialiiril e  oieoootoloailoumn iooooooooaoronaaooooieoeonoromoooooooo aan

# ####
# Time: 2206.8956 s, Epoch: 6000/100000, Loss: 2.5014
# Sample:
# Wheio leoaan d   ,eero looeeaei e y e, oio drotioscathaou  seiawsauiaa aseea essaatl hoctiantloo oacau

# ####
# Time: 2279.5939 s, Epoch: 6200/100000, Loss: 2.4951
# Sample:
# Wheeea ie ol,'e   ear eo o o  nawonsa ee   s aeieeee aaeaa tu   ne  o oe  aus n  e caaeteeheaetteactoe

# ####
# Time: 2352.2956 s, Epoch: 6400/100000, Loss: 2.4517
# Sample:
# Wha ieledaen  orooteonou fe oes  eoe oa o so oe  mteou   e e e e a  toss uosn usosmoeemeo,   s o  umue

# ####
# Time: 2424.9952 s, Epoch: 6600/100000, Loss: 2.4409
# Sample:
# Wheo o eniate nreooar  orioooooooooooooootoooororoerrooooooorrroootoooorooooooeororooooooroormorioao'a

# ####
# Time: 2497.9720 s, Epoch: 6800/100000, Loss: 2.4797
# Sample:
# Wha !iee meo:e ea   se  aw i     ieelmanosro  eoenesm noo saonnoso oammia n  as eseee: asotooonyanieao

# ####
# Time: 2572.7135 s, Epoch: 7000/100000, Loss: 2.4406
# Sample:
# Whiartratrltr  co u rr rtti e ieeelsea eostssscuhao t o e t   rootaeeanru eor u e r s ou   ta  t     t

# ####
# Time: 2645.4656 s, Epoch: 7200/100000, Loss: 2.4820
# Sample:
# Wheieoa r non ndar  ioe'ssssas ssesses sssst ssstt
# en sst otsttstt steetssate ststu mn  sn tsssts slas

# ####
# Time: 2718.2670 s, Epoch: 7400/100000, Loss: 2.4529
# Sample:
# Wh e meee eas easeiaoemsloaaamumiooioeeaaaanelalaelooaieloaou eioooooaaoaonooooaoooneaeoooe oeaaaaaiee

# ####
# Time: 2791.0363 s, Epoch: 7600/100000, Loss: 2.4604
# Sample:
# Wh o iattsst'sst s etsee  os  gsesaaststa's t sstte   es t thees s tstet eass   s et ss t etss se tsts

# ####
# Time: 2863.8652 s, Epoch: 7800/100000, Loss: 2.4646
# Sample:
# Wheae,       e     i ' a  tl e   t   ,       m
#       s      e               e,   e           s

# ####
# Time: 2936.6419 s, Epoch: 8000/100000, Loss: 2.4862
# Sample:
# Whe; a y e a  e o iooo aanomoneooiooiemmtemaeuoae ciaioiiiei seiiniiiiooeaitouo tocooouaeeaeaaaioa  ie

# ####
# Time: 3009.3742 s, Epoch: 8200/100000, Loss: 2.4467
# Sample:
# Whyioo oaamnn,o   ,   ms i    mam's       memseea maseioo eo sd eieiaant  t   o ciaiiises ss ssseciai

# ####
# Time: 3082.3433 s, Epoch: 8400/100000, Loss: 2.4806
# Sample:
# Wh ee e o       y eowerae ieneaoooooloonooarieneooooaounaoeragete  ass s o ssots ceae eaottys ottoaoao

# ####
# Time: 3155.1481 s, Epoch: 8600/100000, Loss: 2.4881
# Sample:
# Wheeeileerrr
# ieiooosuoeeeacaaeeaaeeaee oeaee leeaioeaaa a eaa aiaaeoteaea a ea aal ieaaieeaeaeaaaaita

# ####
# Time: 3227.9119 s, Epoch: 8800/100000, Loss: 2.4599
# Sample:
# Whear se      ,    i   ae yoi  n auen ateemy   t  t      n s        n   ec eeeaaeta ataiaoot o minusau

# ####
# Time: 3300.6949 s, Epoch: 9000/100000, Loss: 2.4538
# Sample:
# Wheeao  o i arr iao ad,avayeieeioaiaaaiamyoeeaya eeao eaooteloeoe. 'y a'  oea      eo  ie e oaa oo o

# ####
# Time: 3373.4095 s, Epoch: 9200/100000, Loss: 2.4557
# Sample:
# Whaioame,     oieoaa aoa ua sm yy  aoe'ssma  seseweea ema ae aods a esa o esmat otls    s s  oa a smos

# ####
# Time: 3446.1548 s, Epoch: 9400/100000, Loss: 2.4801
# Sample:
# Whot oo-aiai utntnnsnmnmnnmitia  m s      afae   itss maa  iee t e  aaa ol i aeoe i,   e s i  n   aioo

# ####
# Time: 3518.8856 s, Epoch: 9600/100000, Loss: 2.4468
# Sample:
# Wheeon w   eoo,  oe    e aamo  i one  e aeiiar s ienneaasneeaeeeo nsnasea   ngaana,    noie a  ,a na

# ####
# Time: 3591.7374 s, Epoch: 9800/100000, Loss: 2.4835
# Sample:
# Whe aaaeatae ae ee,, o iptea toe
# aooe;te
# is
# .
# ,a e cooeat  aa
#  ierit eoo, iaeaaraaeeaiesaiaaaaar elr

# ####
# Time: 3665.4201 s, Epoch: 10000/100000, Loss: 2.4707
# Sample:
# Wheea  ieamr ,e s     s   fey .o   e   e' e,  , d o    e  i  too '  e  s     te   t,     '      e

# ####
# Time: 3739.4768 s, Epoch: 10200/100000, Loss: 2.4598
# Sample:
# Whor e.      irerurrr e eeer  e  rrei e   eetairri,eua orleo oosoeaooeliee e  e    i  o e um   e  e  n

# ####
# Time: 3816.7557 s, Epoch: 10400/100000, Loss: 2.4374
# Sample:
# Whieea r oaere orir i rd teoa  r    ,  i  e    l e          ,        ,    y            a o  ri

# ####
# Time: 3897.0216 s, Epoch: 10600/100000, Loss: 2.4507
# Sample:
# Wh ooiroo oie aaaioeoearr anoeaa  a    oaaaaan osoeaea e o   e     ,      a ane n oinoaronnnsaas nawoo

# ####
# Time: 3976.0119 s, Epoch: 10800/100000, Loss: 2.4592
# Sample:
# Whim  eeeee e aaanooaooaeieaa,atlaro erooesuonal etera alarere  aioao naaiaoeaoeaoeoioeeaou aalieaelel

# ####
# Time: 4054.5111 s, Epoch: 11000/100000, Loss: 2.4444
# Sample:
# Wha ti o use , ,  tir  et   u   t  ? r u ,  ,  cts


# o o




# o




# '


# i
# u
# s
# t

# uoistsotattt
# tttctst

# ####
# Time: 4131.4897 s, Epoch: 11200/100000, Loss: 2.4708
# Sample:
# Wh'yi ;t e ii uroou e al'o amusstlts sn    s e  rst s      os  ss? sl s
# s  ye  et  e   so o     e i

# ####
# Time: 4205.3146 s, Epoch: 11400/100000, Loss: 2.4721
# Sample:
# Whak  ,, i snts' i seiel iseleeesiiesiesi issseisss  sssstu ss'esseissstssssssss
# e  sesss
# ssisses tie

# ####
# Time: 4279.8948 s, Epoch: 11600/100000, Loss: 2.4738
# Sample:
# Wheeee  eae, ss tw ieio ouue
# ,ee, taaiosu uosesuusuyuyneyuo ust ot toy no iue o  o
# ea       s tnanut

# ####
# Time: 4354.7046 s, Epoch: 11800/100000, Loss: 2.4381
# Sample:
# Wheaieoto e.  w  yi i o in orr.'s  ss a iegs  osoeteeoaeooetia   f  a  usn wwa  nslesecawtoutyal
# yi ,s

# ####
# Time: 4430.0820 s, Epoch: 12000/100000, Loss: 2.4448
# Sample:
# Whatton oo rtoot
# crorurrytrrrrrr r curscttouyorsoctrerkcocirtttterrtrrcrirrrrercrrrrrrrcrcr rtcrrrrcrr

# ####
# Time: 4505.4830 s, Epoch: 12200/100000, Loss: 2.4558
# Sample:
# Wheaity,       , e   i   t  u te o ee  s   eo         -   u             e    s        oi  f

# ####
# Time: 4580.7898 s, Epoch: 12400/100000, Loss: 2.4399
# Sample:
# Whamoorueyoo u ea  lo eye e  ee'lml   s ,,   ,      ,    ,,ae e  a e   e ee     aeioeeeo aoaoo aeeeloe

# ####
# Time: 4656.9998 s, Epoch: 12600/100000, Loss: 2.4364
# Sample:
# Wh

# ####
# Time: 4736.2755 s, Epoch: 12800/100000, Loss: 2.4532
# Sample:
# Wh

# ####
# Time: 4816.0056 s, Epoch: 13000/100000, Loss: 2.4373
# Sample:
# Wh

# ####
# Time: 4895.6147 s, Epoch: 13200/100000, Loss: 2.4710
# Sample:
# Wh

# ####
# Time: 4975.5689 s, Epoch: 13400/100000, Loss: 2.4768
# Sample:
# Wh

# ####
# Time: 5055.6982 s, Epoch: 13600/100000, Loss: 2.4454
# Sample:
# Wh

# ####
# Time: 5136.1495 s, Epoch: 13800/100000, Loss: 2.4547
# Sample:
# Wh

# ####
# Time: 5216.3171 s, Epoch: 14000/100000, Loss: 2.4263
# Sample:
# Wh
