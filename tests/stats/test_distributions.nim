# Copyright (c) 2020 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, math

suite "Distributions":
  test "Gaussian distribution":
    block:
      # individual 1
      let mean = 0.0
      let sigma = 1.0
      check gauss(0.0, mean, sigma) == 1.0
      check round(gauss(sigma, mean, sigma), places = 2) == 0.61
    block:
      # individual 2
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

  test "Box distribution":
    block:
      # individual 1
      check box(0.0) == 1.0
      check box(-0.5) == 1.0
      check box(0.5) == 1.0
      check box(-1.0) == 0.0
      check box(1.0) == 0.0
    block:
      # full tensor
      let t = linspace(-1.0, 1.0, 5)
      let exp = @[0.0, 1.0, 1.0, 1.0, 0.0].toTensor()
      let tbox = t.box()
      for i in 0 ..< t.size:
        check tbox[i] == exp[i]

  test "Triangular distribution":
    block:
      # individual 1
      check triangular(0.0) == 1.0
      check triangular(-0.5) == 0.5
      check triangular(0.5) == 0.5
      check triangular(-1.0) == 0.0
      check triangular(1.0) == 0.0
    block:
      # full tensor
      let t = linspace(-1.0, 1.0, 5)
      let exp = @[0.0, 0.5, 1.0, 0.5, 0.0].toTensor()
      let tbox = t.triangular()
      for i in 0 ..< t.size:
        check tbox[i] == exp[i]

  test "Trigonometric distribution":
    block:
      # individual 1
      check trigonometric(0.0) == 2.0
      check trigonometric(-0.5) == 1.0 + cos(2 * PI * 0.5)
      check trigonometric(0.5) == 1.0 + cos(2 * PI * 0.5)
      check trigonometric(-1.0) == 0.0
      check trigonometric(1.0) == 0.0
    block:
      # full tensor
      let t = linspace(-1.0, 1.0, 5)
      let half = 1.0 + cos(2 * PI * 0.5)
      let exp = @[0.0, half, 2.0, half, 0.0].toTensor()
      let tbox = t.trigonometric()
      for i in 0 ..< t.size:
        check tbox[i] == exp[i]

  test "Epanechnikov distribution":
    block:
      # individual 1
      let half = 9.0 / 16.0
      check epanechnikov(0.0) == 3.0 / 4.0
      check epanechnikov(-0.5) == half
      check epanechnikov(0.5) == half
      check epanechnikov(-1.0) == 0.0
      check epanechnikov(1.0) == 0.0
    block:
      # full tensor
      let t = linspace(-1.0, 1.0, 5)
      let half = 9.0 / 16.0
      let exp = @[0.0, half, 3.0 / 4.0, half, 0.0].toTensor()
      let tbox = t.epanechnikov()
      for i in 0 ..< t.size:
        check tbox[i] == exp[i]
