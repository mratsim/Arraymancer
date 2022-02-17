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
  os, random, times, strformat, algorithm, sequtils, tables,
  ../src/arraymancer

# ################################################################
#
#                     Environment constants
#
# ################################################################

# Printable chars - from Python: import string; string.printable
const IxToChar = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c"
const UnkCharIx = IxToChar.len # Unknown characters will be replaced by this
type PrintableIdx = uint8

func genCharToIx(): Table[char, PrintableIdx] =
  result = initTable[char, PrintableIdx]()
  for idx, ch in IxToChar:
    result[ch] = PrintableIdx idx

const
  CharToIx = genCharToIx()
  VocabSize = IxToChar.len + 1 # Cardinality of the set of PrintableChars. There is an extra 1 for the "UnknownChar"
  BatchSize = 100
  Epochs = 2000                # This take a long long time, I'm not even sure it converges
  Layers = 2
  HiddenSize = 100
  LearningRate = 0.01'f32
  EmbedSize = 100
  SeqLen = 200                 # Characters sequences will be split in chunks of 200
  StatusReport = 200           # Report training status every x batches

# ################################################################
#
#                           Helpers
#
# ################################################################

proc strToTensor(str: string|TaintedString): Tensor[PrintableIdx] =
  result = newTensor[PrintableIdx](str.len)

  # For each x in result, map the corresponding char index
  for i, val in result.menumerate:
    if str[i] in CharToIx:
      val = CharToIx[str[i]]
    else:
      # otherwise skip - this will be a padding index
      val = UnkCharIx


# Weighted random sampling / multinomial sampling
#   Note: during text generation we only work with
#         a batch size of 1 so for simplicity we use
#         seq and openarrays instead of tensors

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

  # Get a sample from an uniform distribution
  let u = T(rng.rand(1.0))

  # Get the Cumulative Distribution Function of our probabilities
  let cdf = cumsum(probs, axis = 0)

  # We pass our 1D Tensor as an openarray to `searchsorted` avoid copies
  let cdfA = cast[ptr UncheckedArray[T]](cdf.unsafe_raw_offset)
  result = cdfA.toOpenArray(0, cdf.size - 1).searchsorted(u, leftSide = false)

# ################################################################
#
#                        Neural network model
#
# ################################################################

# We use a classic Encoder-Decoder architecture, with text encoded into an internal representation
# and then decoded back into text.
# So we need to train the encoder, the internal representation and the decoder.

network ShakespeareModel:
  layers:
    encoder:  Embedding(VocabSize, EmbedSize, padding_idx = UnkCharIx)
    gru:      GRULayer(encoder.out_shape[0], HiddenSize, Layers)
    decoder:  Linear(HiddenSize, VocabSize)
  forward input, hidden0:
    let (output, hiddenN) = input.encoder.gru(hidden0)
    # result.output is of shape [Sequence, BatchSize, HiddenSize]
    # In our case the sequence is 1 so we can simply flatten
    let flattened = output.reshape(output.value.shape[1], HiddenSize)

    (output: flattened.decoder, hidden: hiddenN)
  
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

proc train[T](
        ctx: Context[AnyTensor[T]],
        model: ShakespeareModel[T],
        optimiser: var Optimizer[AnyTensor[T]],
        input, target: Tensor[PrintableIdx]): float32 =
  ## Train a model with an input and the corresponding characters to predict.
  ## Return the loss after the training session

  let seq_len = input.shape[0]
  var hidden = ctx.variable zeros[float32](Layers, BatchSize, HiddenSize)

  # We will cumulate the loss on the whole seq before backpropping at once.
  var seq_loss = ctx.variable(zeros[float32](1), requires_grad = true)

  for char_pos in 0 ..< seq_len:
    var output: Variable[Tensor[T]]
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

proc gen_text[T](
        ctx: Context[AnyTensor[T]],
        model: ShakespeareModel[T],
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
      var (_, hiddenX) = model.forward(primer[char_pos, _], hidden)
      hidden = hiddenX


    result = seed_chars

    # And start from the last char!
    var input = primer[^1, _]
    var output: Variable[Tensor[T]]

    for _ in 0 ..< seq_len:
      (output, hidden) = model.forward(input, hidden)
      # output is of shape [BatchSize, VocabSize] with BatchSize = 1

      # Go back in the tensor domain
      var preds = output.value

      # We scale by the temperature first.
      preds /.= temperature
      # Get a probability distribution.
      let probs = preds.softmax().squeeze(0)

      # Sample and append to the generated chars
      let ch_ix = probs.sample().PrintableIdx
      if ch_ix != UnkCharIx:
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
  let model = ctx.init(ShakespeareModel)#ctx.newShakespeareNet()

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

# ###########################################################################################
#
#                          Text generation - Shakespeare
#
# ###########################################################################################

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

# #################
# Starting training


# ####
# Time: 0.9528 s, Epoch: 0/2000, Loss: 4.6075
# Sample:
# Whxpq[<],@
# \;d@JmO.JQg-DN!e7tUXO{D(PftMkX7
#               Hhz;
# dN<w
# {9Z<&}Sw/Y\\6vE/>ISOUF
#                       URfPy=>&>z`

# ####
# Time: 162.7714 s, Epoch: 200/2000, Loss: 1.5452
# Sample:
# Whe coman:
# You will occse is not
# this bright in the deny,
# Doth
# What, he do not on the malding wouth th

# ####
# Time: 318.5498 s, Epoch: 400/2000, Loss: 1.4075
# Sample:
# Whis fortunable face,
# And specians for the thought to beeF the air are to your jat,
# To must be in that

# ####
# Time: 483.7722 s, Epoch: 600/2000, Loss: 1.4117
# Sample:
# Whfor as to the soons
# Of wish'd us o' this banes.
#
# KING HENRY VI:
# So places you do not sweet heart, bi

# ####
# Time: 644.9014 s, Epoch: 800/2000, Loss: 1.3998
# Sample:
# Whee, hence is bastard repose where and
# grie for likelical o'er his stating.
#
# ANGELO:
# O comfort world:

# ####
# Time: 812.3859 s, Epoch: 1000/2000, Loss: 1.3498
# Sample:
# Whall foectiove of Lord:
# Why Hastand Boisely.

# First Citizen:
# Good-shappets and all the secares Homedi

# ####
# Time: 969.4953 s, Epoch: 1200/2000, Loss: 1.3605
# Sample:
# Whou judgty injurity sorrow's quarrel conmioner?
#
# BAPTISTA:
# No fetcom up, I say with one more of time

# ####
# Time: 1138.4985 s, Epoch: 1400/2000, Loss: 1.3145
# Sample:
# Whe sweet Citying
# A bloody though yourson to the Duke of Hereford are:
# My life in Dost to be so on? He

# ####
# Time: 1303.5912 s, Epoch: 1600/2000, Loss: 1.3774
# Sample:
# Whemselves,
# And hates in a accides whilst my state,
# She dival wrough not unto this, see to your lander

# ####
# Time: 1470.3374 s, Epoch: 1800/2000, Loss: 1.3561
# Sample:
# Wh your banished, after but the only ignorland.
# O, must it close out lies
# To courtious are quiet upon,

# ##########
# Training end. Generating 4000 characters Shakespeare masterpiece in 3. 2. 1...


# Whter!
# Take's servant seal'd, making uponweed but rascally guess-boot,
# Bare them be that been all ingal to me;
# Your play to the see's wife the wrong-pars
# With child of queer wretchless dreadful cold
# Cursters will how your part? I prince!
# This is time not in a without a tands:
# You are but foul to this.
# I talk and fellows break my revenges, so, and of the hisod
# As you lords them or trues salt of the poort.
#
# ROMEO:
# Thou hast facted to keep thee, and am speak
# Of them; she's murder'd of your galla?
#
# ANTES:
# Nay, I hear i' the day, bie in half exorcheqous again.
# Cockin Tinved: I is wont? Who be youth friends
# In our beauty of one raised me in all me;
# This will recour castle appelied is:
# I thank you, lords.
# Who, I have not offer, the shipp'd, shalt it is Isabels
# We will be with my keepons of your witfers.
# I was as you have perfited to give car.
#
# SICINE:
# In a sisterexber his record to my turn
# Made you dishonour's, if they have so yean
# Reportistiful viel offs, which we will prayed
# By merry the nightly to find them:
# The fiery to: and she double last speak it,
# For I will resian, he, mark for the air:
# O did thy mustable lodge! Nen't, my mosts!
# I greet before,--hath age-tinent or breath?
#  I would your firms it be new-was 'scape. Is he shall choice,
# Were our husband, in what here twenties and forly,
# Althess to bries are time and senses, and dead-hear themselves
# Having, and this brother is they had'd is; I have a captive:
# My grains! a scarl doing of true forth, some trutis
# As Paduition, by this till us, as you teever
# Whething those baintious plague honour of gentleman,
# Through God lies,
# conunsel, to dishanging can for that men will well were my rasped me
# As well'd as the way off than her wairs with Lancaster show.
# Ah, will you forgot, and good lies of woman
# With a
# feshie:
# Good my Lord.
#
# AUTOLYCUS:
# Whit!
# Grave ta'en my lord, I'ld their names. The are mored of sorrow hath those
# soon weep'st his eyes. My horrcowns, bone, I kindness:
# How idle were which mean nothing cannot weep
# To rescockingly that hasting the sorrow,
# A good to grow'd of our hate how--
# Hear thee your tempest provided: I never confirm,
# Let's a brackful wife calms; they are instyef,
# Shall make thee, but my love.
#
# LADY ANNE:
# Methinks to him:
# But O, have it become ingly stand; think,
# And told the sringer'd againny, Pito:
# Ay, sir; answer'd awe! methink-'Ge is good hour!
# I pray you casquen not hear my form.
# Your unmanding them friends and barth halber,
# More words should not; and to a daughter'd and poor strop'd
# By one as we prove a cursed would not now:
# For thus in a flate death the heaven'd:
# And lies before I hapk or were.
#
# Nurse:
# Fearlwellare, confiarly Marciusbson,
# Were I how stop poiring to no more,
# To worser body to me and die clots, and out
# Their correction defimbry's truth.
#
# BRUTUS:
# Prother to be deadly of gold to be yet,
# Witholesfair than your complished, thus
# wearing triumph that live thyse toes a noble queen:
# I will yet, let him friends to given: take all
# Clease them a slain: our hours and saw Richmes,
# 'Foren thou straight whet it for your treis.
# First is, for you to cousosa thus I'll make weed.
#
# QUEEN:
# I thrive, and how all thy comes?
#
# PRINCE EDWARD:
# Why, the day of all spoil'd nor unsure?
# Come, but never my love is mine,
# To she he himself prevone one it eag.
# Holdis true, bid got I am will not to titteat?
#
# SICINIUS:
# Consign nows this,
# My turns and dead before they-that was me to thy deat?
#
# CORIOLANUS:
# Even earth,
# Your churchister of Romeo, and grace is honest
# and mine envyou.
#
# DUCHESS OF YORK:
# Stand doth ceasians of Edward is time
# Of those would hence I have stopp'd;
# That is this parlest for all time and that eyes
# -adey is remain twine, that can yield
# Have I cursed and were they shouldst fire; I
# privile to thy fair Richard quietlious.
#
# LADY CAPULEL:
# No, but some bebarduched fight the so?
# If I may shake one will't not find him be souls
# They have you inkfender in death to give:
# Soft! hast here and sister of yourmer shuts
# Yet be it strike deabe; thy sures the while.
#
# WARWICK:


# ###########################################################################################
#
#                 Text generation - Pride and Prejudice, Jane Austen
#
# ###########################################################################################

# $  ./build/ex06 build/pride_and_prejudice.txt
# Checking the first hundred characters of your file
# PRIDE AND PREJUDICE
#
# By Jane Austen
#
#
#
# Chapter 1
#
#
# It is a truth universally acknowledged, that a sin

# ####
# Starting training


# ####
# Time: 0.8692 s, Epoch: 0/2000, Loss: 4.6137
# Sample:
# A8T+ ^Cyvd&Ep<"e8tVXO{C(PftLlY7
#                                ^=d[KmP.IQg,DN
# {9!=&~Sw/Y]]6uD/?HSOUEVSgPy=>&?y{            Hiy;

# ####
# Time: 153.5895 s, Epoch: 200/2000, Loss: 1.3105
# Sample:
# Whereince you trood object. She Elizabeth; she gratise her Lyday's like no;
# in the manies.
#
#
#
# Ve

# ####
# Time: 301.1851 s, Epoch: 400/2000, Loss: 1.2050
# Sample:
# Whe had evident heard an
# apporing? Do her long luditure, he alsoment on it?
#
# STreeapen
# had get

# ####
# Time: 458.2439 s, Epoch: 600/2000, Loss: 1.1570
# Sample:
# Wher was to recommend on where his own; she are explain her for; and that he
# town to her
# feative her l

# ####
# Time: 609.7807 s, Epoch: 800/2000, Loss: 1.1460
# Sample:
# Wheir false of advice with which which was complimented. But
# as he had once towards the belief of
# spir

# ####
# Time: 768.7109 s, Epoch: 1000/2000, Loss: 1.1374
# Sample:
# Whall
# following
# oof
# those
# is
# very, it was disgrace, was soon engaged as his affected the common women

# ####
# Time: 915.6025 s, Epoch: 1200/2000, Loss: 1.1079
# Sample:
# Whis
# indiFforth, she added Darcy too, it may longed I suppose, over the
# you, there is
# in do?
#
#

# ####
# Time: 1063.1721 s, Epoch: 1400/2000, Loss: 1.1095
# Sample:
# Wher with which him at least the visitors to see Mr. Collins. We me all in friend who mistaken to temp

# ####
# Time: 1209.6270 s, Epoch: 1600/2000, Loss: 1.1141
# Sample:
# Whement reason! and at him all be view to him to give mine
# and
# Lydia, he was not understoom would hard

# ####
# Time: 1354.8098 s, Epoch: 1800/2000, Loss: 1.1236
# Sample:
# Wh them again, ageable.
# Even she knowsifience, entering to find the sense
# of course of assures how lon

# ##########
# Training end. Generating 4000 characters Shakespeare masterpiece in 3. 2. 1...


# Which you all temper, that I had Mr.
# Longbourn, Lizzy! that Miss de Jane! she cannot be anxious fair propose
# that it
# was she thought he was true, not you are often county
# tto disagreeing. They are you are willing to the best, the idea of her own neither eam since never and Mr. Bennet, he has love a cprovember for Hunsford, except of imaginable new more wanted there said before them, observation to have still every kingded, and received against as possition for the evensing--hortness toward too, and saying, which
# Mlay
# else has daugement to her sister, one again
# he
# ever deson its restance of her tone of protessed with his always congratulation.
#
# Ortrutes without entering; and her sister did nothing of a little beneach of attention. But so much twenting you perverseness were stairs.
#
# Lydy!
# Uflect as for good immediately to clother to you, as she thoughth indole young ladyship will be more feelings, that she did not
# and sister's good to do ill! Where were time before he has night; but
# without considered him to says, indeed, were a great heavance of Jane, she feeling at Roshe thing out colour Mr. Darcy is. Her favourously certain proved in acquaintance of listened.
#
# But who live forward more equacity, suitated, for Jane had satisfied there was early outs chance yourself, more buined
# their reason which she beg there imaginary gone with animation was seeing their being here; and air ran the weated with a still encomined to tell me to indifferent, however in Maria much compliments to see him. The
# supportation; belonged looking in this
# which gate of the housematy.
#
#
#
# Chance now. Elizabeth took denoring was not upon the room. Not at preserving eqiadly leaving that her family, and everything in love all the morning. They may joy iming any fact remil in ought to set you could
# not be--His won conviction to time was, then indeed, where considables for Jane; and if he evielver, and hoped soon face than she had alacrity from another, who had a very speech: 'Oh being in tear of both directly or bour into three more. Why must have both.
#
# I must turn was not out of declared, has been begged a comfort openly. A good repaired a Sir William. He
# was assured the eviven to her sisters' without warmity and of her, satisfied with a great child. The autuit and paid, and Mr. Darcy too happiness at his little chance? Oh, Lointhing entrowed
# heart, however, he has disanyonded, he is now.
#
# In the way, acconduch surprised himself, but them all, and once sound, with such an her sister daughter between them to be look except him freedom there worth home.
#
# There is attentive anybody.
#
# Mr. Bingley's discomposure of Maria, was no loss which her sorry to know
# to Colonel Fitzwilliam; especially, and supportable of the two hour, and to discover the sooner shared. After schemnession because had affected to
# kind disturbed; but
# she continued; nor informed, an inconcludled to take to assure the
# office he with some agoyced are
# so asks to have here; though, was toler more. Have heard with their humbore she would stay; but it is, she rivery
# twoo was such a daughter. Of her situation would always
# feeling to
# mention to be more than to come to say, whom Elizabeth added himself. Tto
# feelings with the whole into a Mr. Collins, and was truth, he amges
# Pemberley, it would
# table to Mr. Collins; but I assured out I did, and be delivery and gave a plan, said Jine, strange, which had no
# success of during determined him engaged.
#
# All aning out of the time of unknown to the whole, and
# her minulsficent, she
# keeps as without interrupted to encourable in such a good friendropy; yet. The
# design
# in the still out
# of her agreeable as
# saw more
# could. In his sister were placed a young man with they house. He is now
# trees. But his account attentions--and attended out? Upon my up. I shall insteems, replied Meryton of Kitty settling might pleasantly with her eyesoughter, said Elizabeth, have me
# this!
#
# Pride of danger was till you have no
# longer we wr
