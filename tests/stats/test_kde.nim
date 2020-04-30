import ../../src/arraymancer except readCsv
import unittest

suite "Kernel Density Estimation - KDE":
  test "Simple test with single entries per kernel":
    let measure = @[0, 3, 7, 10].toTensor()
    # choose samples and bw such that each the result will be
    # peaks of normalized height
    let res = kde(measure, bw = 0.1, samples = 11)
    # expected result is each measurement having a value of 1
    check res / max(res) == @[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 , 0.0, 1.0].toTensor()

  test "Custom sampling points":
    let measure = @[0, 3, 7, 10].toTensor()
    # choose samples and bw such that each the result will be
    # peaks of normalized height
    let res = kde(measure, bw = 0.1, samples = @[3, 7].toTensor)
    check res / max(res) == @[1.0, 1.0].toTensor()

  test "More complex example":
      let a = [1, 4, 2, 4, 6, 3, 2, 5, 2].toTensor()
      block:
        # these values were obtained from sklearn by:
        # In [1]: import numpy as np
        # In [2]: from sklearn.neighbors import KernelDensty
        # In [3]: a = np.asarray([1, 4, 2, 4, 6, 3, 2, 5, 2])
        # In [5]: samples = np.linspace(a.min(), a.max(), 20)
        # In [6]: kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(a.reshape(-1, 1))
        # In [7]: res = kde.score_samples(samples.reshape(-1, 1))
        # convert to log to normal
        # In [8]: np.exp(kde.score_samples(samples.reshape(-1, 1)))
        # Out[9]:
        # array([0.13198271, 0.15612982, 0.17669393, 0.19179353, 0.20047348,
        #        0.20296451, 0.20056443, 0.19515697, 0.18855785, 0.18196528,
        #        0.17573522, 0.16953105, 0.16271276, 0.15473731, 0.14538226,
        #        0.13473773, 0.12304798, 0.11054674, 0.09739618, 0.08374775])
        let exp = [0.13198271, 0.15612982, 0.17669393, 0.19179353, 0.20047348,
                   0.20296451, 0.20056443, 0.19515697, 0.18855785, 0.18196528,
                   0.17573522, 0.16953105, 0.16271276, 0.15473731, 0.14538226,
                   0.13473773, 0.12304798, 0.11054674, 0.09739618, 0.08374775].toTensor()
        # also force bw to 1 and compare. Roughly matches.
        let res = kde(a, samples = 20, bw = 1.0)
        for i in 0 ..< res.size:
          check abs(res[i] - exp[i]) < 1e-2
      block:
        let samples = linspace(0, 10, 20)
        let exp = [4.54049584e-02, 8.68118959e-02, 1.36989540e-01, 1.80197530e-01,
                   2.01438504e-01, 1.99660279e-01, 1.87215215e-01, 1.74511246e-01,
                   1.61222576e-01, 1.43349697e-01, 1.20606503e-01, 9.47003401e-02,
                   6.70860108e-02, 4.08096512e-02, 2.03705478e-02, 8.08071428e-03,
                   2.49765388e-03, 5.94614969e-04, 1.08309353e-04, 1.50365674e-05].toTensor()
        # exp obtained as above with `np.linspace(0, 10, 20)` as argument
        let res = kde(a, samples = samples, bw = 1.0)
        for i in 0 ..< res.size:
          check abs(res[i] - exp[i]) < 1e-2
