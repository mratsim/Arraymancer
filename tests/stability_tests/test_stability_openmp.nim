# Copyright 2017 the Arraymancer contributors
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


import ../../src/arraymancer
import unittest, math, random, sequtils

suite "Load test - OpenMP":
  test "Integer - toTensor, slicing, matmul, inplace addition, display, rince & repeat":
    # The following bugs shows that OpenMP is very sensitive
    # https://github.com/mratsim/Arraymancer/issues/107
    # https://github.com/mratsim/Arraymancer/issues/78
    # https://github.com/mratsim/Arraymancer/issues/99
    # OpenMP operations is deactivated on Tensors with less than <1000 elements except for matmul
    # We test random seeded inputs in a hot loop to make sure the library is robust


    randomize(1337) # seed for reproducibility

    for _ in 0..<100:

      # Shape of the matrices
      let M = rand(2..100)
      let N = rand(2..100)
      let K = rand(2..100)

      # We create the matrices from seq to test the GC/OpenMP interaction
      let a = newSeqWith(M*N, random(-100_000_000..100_000_000)).toTensor.reshape(M, N)
      let b = newSeqWith(N*K, random(-100_000_000..100_000_000)).toTensor.reshape(N, K)

      # Create a c and test some computation
      var c = a*b - randomTensor(M,K, 100)

      # Take a slice and assign to it
      c[0..<(M div 2), 0..<(K div 2)] = randomTensor(M div 2,K div 2, 100)

      # Redo computation
      c += c

      # display the result
      discard $c
