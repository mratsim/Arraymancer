import times, ../../src/arraymancer, math, sequtils

# Reference python
# import numpy as np
# from scipy.misc import logsumexp
#
# foo = np.arange(1,10,dtype=np.float32)
# logsumexp(foo) # 9.4585514
# np.log(np.sum(np.exp(foo))) # 9.4585514


# Comparing 2 implementations of stable logsumexp
# - Classic: log ∑i exp(xi) = α + log ∑i exp(xi−α)
# with α = max(xi) for xi in x

# - Streaming: from http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
# which is similar to Welford algorithm for streaming mean and variance in statistics

proc logsumexp[T: SomeFloat](t: Tensor[T]): T =
  # Advantage:
  #  - OpenMP parallel
  #  - No branching in a tight loop
  # Disadvantage:
  #  - Two loops over the data, might be costly if tensor is big
  # Note: it should be rare to have over than 1000 classes (ImageNet) for softmax

  let alpha = t.max

  result = t.fold_inline() do:
    # Init first element
    x = exp(y - alpha)
  do:
    # Process next elements
    x += exp(y - alpha)
  do:
    # Merge the partial folds
    x += y

  result = alpha + ln(result)


proc streaming_max_sumexp*[T](t: Tensor[T]): tuple[max:T, sumexp: T] {.noSideEffect, inline.}=
  result.max = -Inf.T   # will store the streaming max of the tensor
  result.sumexp = 0.T   # will store the streaming sum of exp of the tensor

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp *= exp(result.max - x)
      result.sumexp += 1
      result.max = x

proc logsumexp_stream*[T: SomeFloat](t: Tensor[T]): T =
  # Advantage:
  #  - Only one loop over the data
  #  - Can be done "on-the-fly"
  # Disadvantage:
  #  - no parallelization
  #  - branching in tight loop
  #
  # Note: most image problems have less than 1000 classes (or even 100)
  # However NLP problems may have 50000+ words in dictionary
  # It would be great to parallelize the one-pass version
  # (there are parallel running version algorithm we can draw inspiration from)

  # Also as problem size grow, the 1-pass version should scale much better
  # However so does parallel code. ==> Benchmark needed with low, medium and huge scale problem.

  let (max, sumexp) = t.streaming_max_sumexp
  result = max + ln(sumexp)

when isMainModule:
  const nb_iter = 1_000

  echo "Warmup and sanity check, result should be 9.4585514"
  let foo = toSeq(1..<10).toTensor.astype(float32)
  echo logsumexp foo
  echo logsumexp_stream foo

  echo "## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)"
  # Note: "In real life" softmax is done by rows

  let t1000 = randomTensor(256, 1000, 1'f32)

  var dummy = 0'f32
  var start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp t1000
  echo " 1000 elements - logsumexp: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization

  dummy = 0'f32
  start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp_stream t1000
  echo " 1000 elements - logsumexp_stream: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization

  ##########################################################################
  echo "\n\n## 3 logsumexp with batch 256 x 3 labels tensors"
  let t3 = randomTensor(256, 3, 1'f32)

  dummy = 0'f32
  start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp t3
  echo " 3 elements - logsumexp: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization

  dummy = 0'f32
  start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp_stream t3
  echo " 3 elements - logsumexp_stream: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization


  ##########################################################################
  echo "\n\n## 3 logsumexp with batch 256 x 50000 labels tensors (NLP - language dictionary)"
  let t50000 = randomTensor(256, 50000, 1'f32)

  dummy = 0'f32
  start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp t50000
  echo " 50000 elements - logsumexp: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization

  dummy = 0'f32
  start = epochTime()
  for i in 0..< nb_iter:
    dummy += logsumexp_stream t50000
  echo " 50000 elements - logsumexp_stream: ", epochTime() - start
  echo " Dummy value: ", dummy # This is to avoid compiler optimization


## Results on i5-5257U (macOS High Sierra, dual-core mobile 2.7GHz, turbo 3.1)
## Note: OpenMP is significantly SLOWER on macOS than single-threaded
## while on Linux it scales with number of cores.
## See https://github.com/mratsim/Arraymancer/issues/134#issuecomment-340425671
## Potential reasons are:
## - memory fragmentation due to OSX mmap implementation
## - "False sharing"

# Compilation flags
# nim c -d:native -d:release --out:build/logsumexp --nimcache:./nimcache benchmarks/implementation/logsumexp.nim

# MEASURE with epochTime

# ## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)
#  1000 elements - logsumexp: 1.854911
#  Dummy value: 12994.2001953125
#  1000 elements - logsumexp_stream: 1.480516
#  Dummy value: 12994.2001953125


# ## 3 logsumexp with batch 256 x 3 labels tensors
#  3 elements - logsumexp: 0.00579799999999997
#  Dummy value: 7180.666015625
#  3 elements - logsumexp_stream: 0.004421999999999926
#  Dummy value: 7180.666015625


######################################################
## Measure with epochTime
## This is imprecise but needed for OpenMP (otherwise we get nb Cpu used x epochTime)

# Compilation flags NO OPENMP
# nim c -d:native -d:release --out:build/logsumexp --nimcache:./nimcache benchmarks/implementation/logsumexp.nim

# ## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)
#  1000 elements - logsumexp: 1.869992971420288
#  Dummy value: 12994.2001953125
#  1000 elements - logsumexp_stream: 1.463253974914551
#  Dummy value: 12994.2001953125


# ## 3 logsumexp with batch 256 x 3 labels tensors
#  3 elements - logsumexp: 0.005753040313720703
#  Dummy value: 7180.666015625
#  3 elements - logsumexp_stream: 0.004420995712280273
#  Dummy value: 7180.666015625

# Compilation flags WITH OPENMP
# nim c d:openmp -d:native -d:release --out:build/logsumexp --nimcache:./nimcache benchmarks/implementation/logsumexp.nim

# ## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)
#  1000 elements - logsumexp: 1.074860095977783
#  Dummy value: 12994.19921875
#  1000 elements - logsumexp_stream: 1.465465068817139
#  Dummy value: 12994.2001953125


# ## 3 logsumexp with batch 256 x 3 labels tensors
#  3 elements - logsumexp: 0.005573034286499023
#  Dummy value: 7180.666015625
#  3 elements - logsumexp_stream: 0.004441022872924805



##########################################################
# After branch stack-array-of-slice, completely different performance profile

# NO OPENMP
# ## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)
#  1000 elements - logsumexp: 2.175843000411987
#  Dummy value: 12994.2001953125
#  1000 elements - logsumexp_stream: 5.314570188522339
#  Dummy value: 12994.2001953125


# ## 3 logsumexp with batch 256 x 3 labels tensors
#  3 elements - logsumexp: 0.006352901458740234
#  Dummy value: 7180.666015625
#  3 elements - logsumexp_stream: 0.01533412933349609
#  Dummy value: 7180.666015625


# ## 3 logsumexp with batch 256 x 50000 labels tensors (NLP - language dictionary)
#  50000 elements - logsumexp: 103.6776480674744
#  Dummy value: 16920.689453125
#  50000 elements - logsumexp_stream: 263.6669411659241
#  Dummy value: 16920.689453125

# WITH OPENMP
# ## 1000 logsumexp with batch 256 x 1000 labels tensors (~ImageNet)
#  1000 elements - logsumexp: 1.03753399848938
#  Dummy value: 12994.19921875
#  1000 elements - logsumexp_stream: 1.65704607963562
#  Dummy value: 12994.2001953125


# ## 3 logsumexp with batch 256 x 3 labels tensors
#  3 elements - logsumexp: 0.005845785140991211
#  Dummy value: 7180.666015625
#  3 elements - logsumexp_stream: 0.004522085189819336
#  Dummy value: 7180.666015625


# ## 3 logsumexp with batch 256 x 50000 labels tensors (NLP - language dictionary)
#  50000 elements - logsumexp: 47.58395099639893
#  Dummy value: 16905.359375
#  50000 elements - logsumexp_stream: 91.42335796356201
#  Dummy value: 16920.689453125
