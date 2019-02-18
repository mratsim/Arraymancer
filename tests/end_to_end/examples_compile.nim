# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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
import unittest

# TODO - currently compiles run the code :/

suite "Full examples - compilation check only":
  test "Example 1: XOR Perceptron":
    check: compiles: import ../../examples/ex01_xor_perceptron_from_scratch
  test "Example 2: MNIST via Convolutional Neural Net":
    check: compiles: import ../../examples/ex02_handwritten_digits_recognition

