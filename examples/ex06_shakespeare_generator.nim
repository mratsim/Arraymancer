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
  Epochs = 10_000
  Layers = 2
  HiddenSize = 128
  LearningRate = 0.001'f32
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
        optimiser: Sgd[TT],
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
      echo &"\n####\nTime: {elapsed:>4.4f} s, Epoch: {epoch}/{Epochs}, Loss: {loss:>2.4f}"
      echo "Sample: "
      echo ctx.gen_text(model, seq_len = 100)

main()

# ##################################################
# Output

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
# Time: 0.5717 s, Epoch: 0/10000, Loss: 5.1517
# Sample:
# Wh605tnnPsv8/brrrrTI6IA#8[88rrrrrjrr>;\I\\uqj0qM4MqMW,j4Uu:5:nr*TjHwo4bn>g8>mXrj.Atja4aaig|d?3TT[[[[Wt

# ####
# Time: 90.0758 s, Epoch: 200/10000, Loss: 2.6763
# Sample:
# Wheaase dswtinnssI fd atatarrearusIordDrrnIII;;;a;aodoot[innartta snrrrrdruIt yrruurrrrII ayrrrruurrrr

# ####
# Time: 187.3876 s, Epoch: 400/10000, Loss: 2.6028
# Sample:
# Wheeeatd,tt ,attri.art??yarrdeddtrrirsIIeDUt?ut;usrdes.X?yItead8ari?routttuitrri.trairurrtrearyrrder.t

# ####
# Time: 285.6617 s, Epoch: 600/10000, Loss: 2.5498
# Sample:
# Wheaiy:sistood.ssisi.sosssiirdeImf ftr atrt;
# o;`dooormiu;??tAsutu!rrsdd?akIetioddrrrurduyrsdodoAdmimar

# ####
# Time: 367.2997 s, Epoch: 800/10000, Loss: 2.5789
# Sample:
# Wheataiiders,et, ttatttteatn etditro artEyoirooooiorr;II?rrry;moirreroIord??rr;?airiruisbpaiomrtrurrrr

# ####
# Time: 450.9833 s, Epoch: 1000/10000, Loss: 2.5497
# Sample:
# Wh ayy e Isiourtrrosryuerterssarc potrrassIire??[;;siairrlo:odoiisdsmIet?trudes? fTorrdos?iddmmtrtat?r

# ####
# Time: 538.6711 s, Epoch: 1200/10000, Loss: 2.5168
# Sample:
# Wheetae;evetai??tredrrnn?add rsiauspiidrrdoddIiroo??i?sy!ttrrtt?atty?ruutuyyrdurrrrI?r syineatrtartaur

# ####
# Time: 617.2467 s, Epoch: 1400/10000, Loss: 2.4899
# Sample:
# Whaitweeissateeedids.eirpesatmdedvimsaPrrtdottr;?uir?rrdsdiirdtrtr?terrr??rdtetaiuuitrlttttttrrtmdreri

# ####
# Time: 694.5187 s, Epoch: 1600/10000, Loss: 2.4839
# Sample:
# Wheadaissicw.rcpe,??;ooopoo??or?ruutarttearsriraurdss?isdiu.ddurrrdrrdrrrtcyrirorr;eraitrsrtusodo??r?r

# ####
# Time: 777.1259 s, Epoch: 1800/10000, Loss: 2.4962
# Sample:
# Whtiaids, oorsio?p5;crer,geIitIssiueee
# rs aand`t ttttuurryirrrartrtyierirurradturrrrrsr?rdtrrtrtutttt?

# ####
# Time: 859.6361 s, Epoch: 2000/10000, Loss: 2.4913
# Sample:
# Whiatsth! I'ttu at?t
#
#
#
#
# ?Xr
# BOBourrtrrar?rdet?rtat?tlutrairtriredretkdd;erurrrrt??rrrdrrdterrdiitrier

# ####
# Time: 940.7449 s, Epoch: 2200/10000, Loss: 2.4965
# Sample:
# Whesispietiwodissdissir??fiisfosoiyrdemmddotsuklttfutyutt?tukreetiaastertutuitrti?irdrrdmairrr?t??tsu?

# ####
# Time: 1022.9619 s, Epoch: 2400/10000, Loss: 2.4786
# Sample:
# Whes,idtub!a
# ctttratart
#
# airrrriuicrerrrtr?iussssimmsbeeaaatttut.;ia6ov6t?`riurriedi??urrriscdrrrttrii

# ####
# Time: 1109.9007 s, Epoch: 2600/10000, Loss: 2.4947
# Sample:
# Whoayt,urt aturrttutarrriredirrrrtirired?rrr?;arrirusr??gtuattrttuttttut??ttttrttttttt?err;arradirtrur

# ####
# Time: 1197.0158 s, Epoch: 2800/10000, Loss: 2.4666
# Sample:
# Whiemaeeetseisiaty?eederut?
#
# u
# dtwssddrolkd addddtr8crrtrrryirmriredicrrrrertt??errrirrrueirdedd?tsw,a

# ####
# Time: 1276.9473 s, Epoch: 3000/10000, Loss: 2.4868
# Sample:
# Whi athoitoovootnoonyyeeste!seeadtt?
# ta;?itattt?eiastss
# !!idooko daivtlly tttutttoorotoorror?ituatt?ir

# ####
# Time: 1428.7132 s, Epoch: 3200/10000, Loss: 2.4850
# Sample:
# Whigtr,ath.
#
# Whpeoik,,tt.faosri?atrtutdurri??rtairrrett?errty?rrreru??urrerarrdt????ru.y rseirrdeddedd

# ####
# Time: 1642.4389 s, Epoch: 3400/10000, Loss: 2.4520
# Sample:
# Whea
# siohosibreere,dedttearttstt.iitist?att
# e
#
# btsoutttyu.tt.eius?rtraetererrrdratirderr;,rrettefi?tuu

# ####
# Time: 1768.0808 s, Epoch: 3600/10000, Loss: 2.4733
# Sample:
# Wh ay hiereeet
#
# dddstsn,adidisstlett??Xitrr,.turrri?at;so??artttuuty.urrdrderurr?turtutrtt?tutiyoa??su

# ####
# Time: 1897.5193 s, Epoch: 3800/10000, Loss: 2.4753
# Sample:
# WheausireeetaIltsorssaidst???t?knrw.??sllulupfthattttttatuttttruttttttuttwetttrt????uru?asbedrs?.tuite

# ####
# Time: 1980.7799 s, Epoch: 4000/10000, Loss: 2.4961
# Sample:
# Whaayisssioooossoooeroirrrirsssidr.?rsiirtI?irarrd?tcurrrdrs?t;dditrtirrierrred?
# ?
# ustroatntd?,rtt?i;r

# ####
# Time: 2067.5314 s, Epoch: 4200/10000, Loss: 2.4826
# Sample:
# Wheeted,t,tttyyrurrd rutrr?ior?rriu??tthutyor??urr?rustt.?aureaatttt?si?cosrpaitutueatstt,ttttatuttutt

# ####
# Time: 2151.6148 s, Epoch: 4400/10000, Loss: 2.4774
# Sample:
# Whaisbbesti
#
# CI att utt
# houteasoo?atttuutt?utustu.uirtr;?erdirrdr??ud
# dddderd;r??rdtuty?e?.; t
# uireddd

# ####
# Time: 2246.4050 s, Epoch: 4600/10000, Loss: 2.4725
# Sample:
# Whidtc
# utbtertyeetoetteretriitraerut?cordi?
# urtcrirordigtiirecassttthesi?seiadittcurrdirderortriter??r

# ####
# Time: 2341.8143 s, Epoch: 4800/10000, Loss: 2.4691
# Sample:
# Whea,atshatweanct
#
# r
# 9Iterea,ttuistt??;?
# ut
# aoorrt??cutrrrutth?ttttatttrriuttuutu?suy?atsuet?eaatlrduu

# ####
# Time: 2423.0604 s, Epoch: 5000/10000, Loss: 2.4635
# Sample:
# Wheiraeelse,!adtt rutrtttuttautuhetutr;ura?ossoutrreededttiurvutt??id??suiratattetuitr?uuisst?eiu
# iatd

# ####
# Time: 2504.9325 s, Epoch: 5200/10000, Loss: 2.4572
# Sample:
# Whotiaioviaeta
# sitpeiidss.,!irr prrtt?ur???ru?rsuateadthoirtreerdw?tutrat?er?cusedtt?tuui?tuX??u?
# Ayr:

# ####
# Time: 2583.1865 s, Epoch: 5400/10000, Loss: 2.4627
# Sample:
# Wh sissbbuew.?rturiatthrustaeerssadicrounrierr???crurrirrrerrtrderte??utt?
# s
# {aittt?ut??turiyry?yuiss?

# ####
# Time: 2665.0941 s, Epoch: 5600/10000, Loss: 2.4524
# Sample:
# Whoyoioyardrssettidnds,tt
# p
# haruttttottuar?uurrdierdrlidatrdt,rtutrti,erreta?tssd???iuuaterrerddeti?do

# ####
# Time: 2744.3063 s, Epoch: 5800/10000, Loss: 2.4640
# Sample:
# Wheeeatk?,ust
#
#
#
#
#
#
# W
# /cyk ondrenirted,??dsr?miirr?i.abyttuirurrrd?iu?rrddddrrdrc?attttuiirtusirri?I?u

# ####
# Time: 2835.9202 s, Epoch: 6000/10000, Loss: 2.4993
# Sample:
# Wheisalitl
# e
# c
# PPeres?!,eteetatre?v???;uor?etssattieteltraatrt?tuiutiedrtrriaryradtt?turyrirtrrtereirt

# ####
# Time: 2918.5590 s, Epoch: 6200/10000, Loss: 2.5049
# Sample:
# Wheeeadts?itissac
# c!!!!a!o!t'!iarssue itte?uatturrrr.iirirauy?autrr.urssiaoytru.
# urrcrrtrrttrruttrtutr

# ####
# Time: 3004.4479 s, Epoch: 6400/10000, Loss: 2.4494
# Sample:
# Wheeeetdeatt; prndsatart??ttusu?!s
# b!uees!tt!illiluatteherataterdiaitsst?irrr;urirrrddier??d?t
# u??y?ud

# ####
# Time: 3084.9506 s, Epoch: 6600/10000, Loss: 2.4409
# Sample:
# Whes
# s
# bilTubaiwotwas, ssjllututututr tsautrttututrty?utuerrrsutdeudiartetardeatt?ilutoriridt;at?trrad

# ####
# Time: 3191.8401 s, Epoch: 6800/10000, Loss: 2.4778
# Sample:
# Whaissbbborteirede,?tt?-tuateec;eradirdirttu??rsisste?eri?uatrettt;reirodouanst.;ttrtt?attututty?urssu

# ####
# Time: 3286.5917 s, Epoch: 7000/10000, Loss: 2.4408
# Sample:
# Whiesseeilatt itt
# uitrrrtyrerertiderri?tututtusucrres?r?ddt
#
#
# d
#
# tspplouyicopew.letltittorerep,tttt?ot

# ####
# Time: 3368.9405 s, Epoch: 7200/10000, Loss: 2.4774
# Sample:
# Wheist;
# i
# ooooooooooooddrrrrdrIIyetn?s;ssirsecisuu
# itmootaitthortririreididsrattttyotttryritrertw?attt

# ####
# Time: 3460.0559 s, Epoch: 7400/10000, Loss: 2.4525
# Sample:
# Wh oiyere
# edvattu?teu?itsse??ittutust??cucrruodrerrri??drr.uu?rrsuorrderdirursd?ttutttuuutr?uts??iurrr

# ####
# Time: 3578.3554 s, Epoch: 7600/10000, Loss: 2.4628
# Sample:
# Wh thiauthet
# ucuirbrrrrre?rr irirtarrtutr?tutrruutr?c?ruitdyrrrdaectrriradctt??iurrtiut?ucryur?urdrrrr

# ####
# Time: 3750.5076 s, Epoch: 7800/10000, Loss: 2.4641
# Sample:
# Wheelsatt
# ichtoibres!aailswi.iucbyripyiooooooooderr!?ir??;y??u rererded?ts?iitrrdutucruerr??utt.udt.?i

# ####
# Time: 3845.2033 s, Epoch: 8000/10000, Loss: 2.4915
# Sample:
# Wheard,wcttsainsttttut..autetrsyiatthaityor?ius
# eees
# eee?eee
# tarosssoosuacctttr,utattryerriradirtreatr

# ####
# Time: 4001.9944 s, Epoch: 8200/10000, Loss: 2.4465
# Sample:
# Whyosuboorourivporru?etu?uir;?adsdt??ut?utyssuirpos?ooowrartirrerrrt??surcucrrrrruor??puu
# tr??uustt?o?

# ####
# Time: 4118.3223 s, Epoch: 8400/10000, Loss: 2.4796
# Sample:
# Wh ay
# pet
#
# phooowonotourrnrrrratrrirdirirt?si??drrru?rutturtsuryer!irtdt???uyrum
# aper?iasowviasttt?sir

# ####
# Time: 4248.9856 s, Epoch: 8600/10000, Loss: 2.4863
# Sample:
# Wheeestt?uuy?
# uussoouspscoooeeeecfttett
# riatr,uoyoryrerr-?
# prr??u.
# msomamdeo;v??iss
# rdd
# rttusiuteitty?

# ####
# Time: 4353.1152 s, Epoch: 8800/10000, Loss: 2.4595
# Sample:
# Wheesett,,
# ttttahttt,
# eet?tutt?ucryrriryirrsiadtrirr??.ucuruirdrdccrri?ius
# lummmmmmmowrrasdr,tttttuttt

# ####
# Time: 4439.8329 s, Epoch: 9000/10000, Loss: 2.4557
# Sample:
# Wheedsatwitiarr
# c
# e
# ??XIrdrrrratrat???rirrrod?yrirrir?orurrrrrrtrrrrurrrr?utyirr??utu? uthietouttu?tru

# ####
# Time: 4538.1685 s, Epoch: 9200/10000, Loss: 2.4540
# Sample:
# Whast,itt
#
#
#
#
#
# M
# t!'
# Oladw fr ayy.,?ynsusiei?usymyooooooo??oittau?t
# uy!
# t,Eoory.??AssooAuat att
# ??tust

# ####
# Time: 4646.3338 s, Epoch: 9400/10000, Loss: 2.4801
# Sample:
# Whouayeeeicretvetttirosisspiureeeidtcaut4Eorrirodowrt?iircnttr?urs;!edecrt.????ru.i.'beedideddt?I?erii

# ####
# Time: 4738.8091 s, Epoch: 9600/10000, Loss: 2.4492
# Sample:
# Whastiiubeestt?
# uus
#
#
#
#
#
#
#
# bu by wooooooooooow,tputttuirutss? iitirurrrdd
# c?rrrrrrardenutrrererrr?us?!

# ####
# Time: 4829.7292 s, Epoch: 9800/10000, Loss: 2.4655
# Sample:
# Wheaeafisw,sst?
# t??tuu?yuyoeessee
# cipe,ut
# qy
# rrderrrerriiewreed
# crtrrtettut?tsrartrinicrruttaaarouttyt
