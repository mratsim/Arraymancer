# Copyright 2020 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ../tensor/tensor,
       ./distributions

import std / [math, strutils]
export nimIdentNormalize # for parseEnum

type
  KernelKind* = enum
    knCustom = "custom"
    knBox = "box"
    knTriangular = "triangular"
    knTrig = "trigonometric"
    knEpanechnikov = "epanechnikov"
    knGauss = "gauss"

  KernelFunc* = proc(x, x_i, bw: float): float {.inline.}

template makeKernel*(fn: untyped): untyped =
  proc `fn Kernel`*(x, x_i, bw: float): float {.inline.} =
    let val = (x - x_i) / bw
    result = fn(val)

makeKernel(box)
makeKernel(triangular)
makeKernel(trigonometric)
makeKernel(epanechnikov)

# manually build the gauss kerne, since the underlying distribution relies
# on more than 1 value.
proc gaussKernel*(x, x_i, bw: float): float {.inline.} =
  gauss(x = x, mean = x_i, sigma = bw, norm = true)

proc getCutoff(bw: float, kind: KernelKind): float =
  ## calculates a reasonable cutoff for a given `KernelKind` for a bandwidth `bw`
  case kind
  of knBox: result = 0.5 * bw
  of knTriangular: result = bw
  of knTrig: result = 0.5 * bw
  of knEpanechnikov: result = bw
  of knGauss: result = 3.0 * bw # 3 sigma amounts to 99.7% of gaussian kernel contribution
  of knCustom: doAssert false, "`getCutoff` must not be called with custom kernel!"

proc getKernelFunc(kind: KernelKind): KernelFunc =
  case kind
  of knBox: result = boxKernel
  of knTriangular: result = triangularKernel
  of knTrig: result = trigonometricKernel
  of knEpanechnikov: result = epanechnikovKernel
  of knGauss: result = gaussKernel
  of knCustom: doAssert false, "`getKernelFunc` must not be called with custom kernel!"

proc findWindow[T](dist: T, s: T, t: Tensor[T], oldStart = 0, oldStop = 0): (int, int) =
  ## returns the index (start or stop) given a distance `dist` that
  ## is allowed between s - x[j]
  # `s` and `x` must be sorted
  var
    startFound = false
    stopFound = false
  var j = oldStart
  while j < t.size:
    if not startFound and abs(s - t[j]) < dist:
      startFound = true
      result[0] = j
      j = if oldStop < j: j else: oldStop
      continue
    elif startFound and not stopFound and abs(s - t[j]) > dist:
      stopFound = true
      result[1] = j
      break
    inc j
 # set to max, if we left the while loop naturally
  if result[1] == 0: result[1] = t.size
  assert result[1] > result[0]

proc kde*[T: SomeNumber; U: int | Tensor[SomeNumber] | openArray[SomeNumber]](
    t: Tensor[T],
    kernel: KernelFunc,
    kernelKind = knCustom,
    adjust: float = 1.0,
    samples: U = 1000,
    bw: float = NaN,
    normalize = false,
    cutoff: float = NaN): Tensor[float] =
  ## Returns the kernel density estimation for the 1D tensor `t`. The returned
  ## `Tensor[float]` contains `samples` elements.
  ##
  ## The bandwidth is estimated using Silverman's rule of thumb.
  ##
  ## `adjust` can be used to scale the automatic bandwidth calculation.
  ## Note that this assumes the data is roughly normal distributed. To
  ## override the automatic bandwidth calculation, hand the `bw` manually.
  ## If `normalize` is true the result will be normalized such that the
  ## integral over it is equal to 1.
  ##
  ## By default the evaluation points will be `samples` linearly spaced points
  ## between `[min(t), max(t)]`. If desired the evaluation points can be given
  ## explicitly by handing a `Tensor[float] | openArray[float]` as `samples`.
  ##
  ## The `kernel` is the kernel function that will be used. Unless you want to
  ## use a custom kernel function, call the convenience wrapper below, which
  ## only takes a `KernelKind` (either as string or directly as an enum value)
  ## below, which defaults to a gaussian kernel.
  ##
  ## Custom kernel functions are supported by handing a function of signature
  ##
  ## `KernelFunc = proc(x, x_i, bw: float): float`
  ##
  ## to this procedure and setting the `kernelKind` to `knCustom`. This ``requires``
  ## to also hand a `cutoff`, which is the window of `s[j] - t[i] <= cutoff`, where
  ## `s[j]` is the `j`-th sample and `t[i]` the `i`-th input value. Only this window is
  ## considered for the kernel summation for efficiency. Set it such that the
  ## contribution of the custom kernel is very small (or 0) outside that range.
  let N = t.size
  # sort input
  let t = t.asType(float).sorted
  let (minT, maxT) = (min(t), max(t))
  when U is int:
    let x = linspace(minT, maxT, samples)
    let nsamples = samples
  elif U is seq | array:
    let x = toTensor(@samples).asType(float)
    let nsamples = x.size
  else:
    let x = samples.asType(float)
    let nsamples = x.size
  let A = min(std(t),
              iqr(t) / 1.34)
  let bwAct = if classify(bw) != fcNaN: bw
              else: 0.9 * A * pow(N.float, -1.0/5.0)
  result = newTensor[float](nsamples)
  let norm = 1.0 / (N.float * bwAct)
  var
    start = 0
    stop = 0
  doAssert classify(cutoff) != fcNan or kernelKind != knCustom, "If a custom " &
    "is used you have to provide a cutoff distance!"
  let cutoff = if classify(cutoff) != fcNan: cutoff
               else: getCutoff(bwAct, kernelKind)
  for i in 0 ..< t.size:
    (start, stop) = findWindow(cutoff, t[i], x, start, stop)
    # TODO: rewrite using kernel(t: Tensor) and fancy indexing?
    for j in start ..< stop:
      result[j] += norm * kernel(x[j], t[i], bwAct)

  if normalize:
    let normFactor = 1.0 / (result.sum * (maxT - minT) / nsamples.float)
    result.apply_inline(normFactor * x)

proc kde*[T: SomeNumber; U: KernelKind | string; V: int | Tensor[SomeNumber] | openArray[SomeNumber]](
    t: Tensor[T],
    kernel: U = "gauss",
    adjust: float = 1.0,
    samples: V = 1000,
    bw: float = NaN,
    normalize = false): Tensor[float] =
  ## This is a convenience wrapper around the above defined `kde` proc, which takes
  ## a kernel as a `string` corresponding to the string value of the `KernelKind`
  ## enum or a `KernelKind` value directly, which does not require to manually hand
  ## a kernel procedure.
  ##
  ## By default a gaussian kernel is used.
  when U is string:
    let kKind = parseEnum[KernelKind](kernel)
  else:
    let kKind = kernel
  let kernelFn = getKernelFunc(kKind)
  result = kde(t,
               kernelFn,
               kernelKind = kKind,
               adjust = adjust,
               samples = samples,
               bw = bw,
               normalize = normalize)
