# Copyright (c) 2020 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest

suite "Distributions":
  test "Gaussian distribution":
    block:
      # individual elements
      block:
        let mean = 0.0
        let sigma = 1.0
        check gauss(0.0, mean, sigma) == 1.0
        check round(gauss(sigma, mean, sigma), places = 2) == 0.61
      block:
        let mean = 5.0
        let sigma = 1.0
        check gauss(5.0, mean, sigma) == 1.0
        check round(gauss(mean + sigma, mean, sigma), places = 2) == 0.61
    block:
      # full tensor
      let mean = 0.0
      let sigma = 1.0

      let t = linspace(-3.0, 3.0, 7)
      let exp = @[0.011, 0.135, 0.607, 1.0, 0.607, 0.135, 0.011]
      let tgauss = t.gauss(mean, sigma)
      for i in 0 ..< t.size:
        check round(tgauss[i], places = 3) == exp[i]
