import ggplotnim, sequtils, stats, algorithm, strutils, math
import arraymancer except readCsv

proc gauss*[T](x, mean, sigma: T, norm = false): float =
  ## Returns a value of the gaussian distribution descriped by `mean`, `sigma`
  ## at position `x`.
  ## based on the ROOT implementation of TMath::Gaus:
  ## https://root.cern.ch/root/html524/src/TMath.cxx.html#dKZ4iB
  ## inputs are converted to float
  if sigma == 0:
    result = 1.0e30
  let
    arg = (x - mean).float / sigma.float
    res = exp(-0.5 * arg * arg)
  if norm == false:
    result = res
  else:
    result = res / (2.50662827463100024 * sigma) # sqrt(2*Pi)=2.5066282746310002

proc gauss*[T](x: Tensor[T], mean, sigma: T, norm = false): Tensor[float] =
  ## version of gauss working on a full tensor
  result = x.map_inline(gauss(x, mean, sigma, norm = norm))

type
  KernelKind* = enum
    knCustom = "custom"
    knBox = "box"
    knTriangular = "triangular"
    knTrig = "trigonometric"
    knEpanechnikov = "epanechnikov"
    knGauss = "gauss"

  KernelFunc* = proc(x, x_i, bw: float): float

proc boxKernel*(x, x_i, bw: float): float =
  ## provides a box kernel
  result = if abs((x - x_i) / bw) < 0.5: 1.0 else: 0.0

proc triangularKernel*(x, x_i, bw: float): float =
  ## provides a triangular kernel
  let val = abs(x - x_i) / bw
  result = if val < 1.0: 1.0 - val else: 0.0

proc trigonometricKernel*(x, x_i, bw: float): float =
  ## provides a trigonometric kernel
  let val = abs(x - x_i) / bw
  result = if val < 0.5: 1.0 + cos(2 * PI * val) else: 0.0

proc epanechnikovKernel*(x, x_i, bw: float): float =
  ## provides an Epanechnikov kernel
  let val = abs(x - x_i) / bw
  result = if val < 1.0: 3.0 / 4.0 * (1 - val * val) else: 0.0

proc gaussKernel*(x, x_i, bw: float): float =
  ## provides a gaussian kernel
  result = gauss(x = x, mean = x_i, sigma = bw)

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
      j = if oldStop == 0: j else: oldStop
      continue
    elif startFound and not stopFound and abs(s - t[j]) > dist:
      stopFound = true
      result[1] = j
      break
    inc j

proc kde[T: SomeNumber](t: Tensor[T],
                        kernel: KernelFunc,
                        kernelKind = knCustom,
                        adjust: float = 1.0,
                        samples: int = 1000,
                        bw: float = NaN,
                        normalize = true,
                        cutoff: float = NaN): Tensor[float] =
  ## Returns the kernel density estimation for the 1D tensor `t`. The returned
  ## `Tensor[float]` contains `samples` elements.
  ## The bandwidth is estimated using Silverman's rule of thumb.
  ## `adjust` can be used to scale the automatic bandwidth calculation.
  ## Note that this assumes the data is roughly normal distributed. To
  ## override the automatic bandwidth calculation, hand the `bw` manually.
  ## If `normalize` is true the result will be normalized such that the
  ## integral over it is equal to 1.
  ##
  ## UPDATE / FINISH
  let N = t.size
  # sort input
  let t = t.sorted
  let (minT, maxT) = (min(t), max(t))
  let x = linspace(minT, maxT, samples)
  let A = min(std(t),
              iqr(t) / 1.34)
  let bwAct = if classify(bw) != fcNaN: bw
              else: 0.9 * A * pow(N.float, -1.0/5.0)
  result = newTensorUninit[float](samples)
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
    for j in start ..< stop:
      result[j] += norm * kernel(x[j], t[i], bwAct)

  if normalize:
    let normFactor = result.sum * (maxT - minT) / samples.float
    result.apply_inline(normFactor * x)

proc kde[T: SomeNumber; U: KernelKind | string](
    t: Tensor[T],
    kernel: U = "gauss",
    adjust: float = 1.0,
    samples: int = 1000,
    bw: float = NaN,
    normalize = true): Tensor[float] =
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

when isMainModule:
  let df = toDf(readCsv("/home/basti/CastData/ExternCode/ggplotnim/data/diamonds.csv"))
  let carat = df["carat"].toTensor(float)
  let x = linspace(min(carat), max(carat), 1000)
  let estimate = kde(carat)
  let dfEst = seqsToDf(x, estimate)
  ggplot(dfEst, aes("x", "estimate")) +
    geom_line(fillColor = some(parseHex("9B4EFF")),
              alpha = some(0.3)) +
    ggsave("density_test.pdf")

  main()
