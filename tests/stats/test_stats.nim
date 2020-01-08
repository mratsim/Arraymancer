# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest

suite "Statistics":
  test "Covariance matrix":

    block: # https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
      let x =  [[4.0, 2.0, 0.6],
                [4.2, 2.1, 0.58],
                [3.9, 2.0, 0.58],
                [4.3, 2.1, 0.62],
                [4.1, 2.2, 0.63]].toTensor

      let expected = [[0.025, 0.0075, 0.00175],
                      [0.0075, 0.0070, 0.00135],
                      [0.00175, 0.00135, 0.00043]].toTensor

      let computed = covariance_matrix(x,x)

      check: computed.mean_absolute_error(expected) <= 1e-04 # The website seems to have precision issue

    block: # p13 of http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
      let data = [[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7],
                  [2.0, 1.6],
                  [1.0, 1.1],
                  [1.5, 1.6],
                  [1.1, 0.9]].toTensor

      let expected = [[0.616555556, 0.615444444],
                      [0.615444444, 0.716555556]].toTensor

      let computed = covariance_matrix(data, data)

      check: computed.mean_absolute_error(expected) <= 1e-09

    block: # Numpy doc https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cov.html

      let x =  [[0.0, 2.0],
                [1.0, 1.0],
                [2.0, 0.0]].toTensor

      let expected = [[ 1.0, -1.0],
                      [-1.0,  1.0]].toTensor

      let computed = covariance_matrix(x, x)

      check: computed.mean_absolute_error(expected) <= 1e-14

    block: # Numpy example 2
      let x =  [[-2.1, 3.0],
                [-1.0, 1.1],
                [ 4.3, 0.12]].toTensor

      let expected = [[11.71, -4.286],
                      [-4.286, 2.14413333]].toTensor

      let computed = covariance_matrix(x, x)

      check: computed.mean_absolute_error(expected) <= 1e-09
