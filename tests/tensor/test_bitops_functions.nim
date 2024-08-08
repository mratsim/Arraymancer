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
import std / unittest

proc main() =
  suite "Bitops functions":
    test "bitnot":
      let t = [0, 1, 57, 1022, -100].toTensor
      let expected = [-1, -2, -58, -1023, 99].toTensor
      check: t.bitnot == expected

    test "shr":
      let t1 = [0, 1, 57, 1022, -100].toTensor
      let t2 = [0, 1, 2, 3, 4].toTensor
      check: t1 shr 3 == [0, 0, 7, 127, -13].toTensor
      check: 1024 shr t2 == [1024, 512, 256, 128, 64].toTensor
      check: t1 shr t2 == [0, 0, 14, 127, -7].toTensor

    test "shl":
      let t1 = [0, 1, 57, 1022, -100].toTensor
      let t2 = [0, 1, 2, 3, 4].toTensor
      check: t1 shl 3 == [0, 8, 456, 8176, -800].toTensor
      check: 3 shl t2 == [3, 6, 12, 24, 48].toTensor
      check: t1 shl t2 == [0, 2, 228, 8176, -1600].toTensor

    test "bitand":
      let t1 = [0, 1, 57, 1022, -100].toTensor
      let t2 = [0, 2, 7, 15, 11].toTensor
      check: bitand(t1, 0b010_110_101) == [0, 1, 49, 180, 148].toTensor
      check: bitand(t1, 0b010_110_101) == bitand(0b010_110_101, t1)
      check: bitand(t1, t2) == [0, 0, 1, 14, 8].toTensor
      check: bitand(t1, t2) == bitand(t1, t2)

    test "bitor":
      let t1 = [0, 1, 57, 1022, -100].toTensor
      let t2 = [0, 2, 7, 15, 11].toTensor
      check: bitor(t1, 0b010_110_101) == [181, 181, 189, 1023, -67].toTensor
      check: bitor(t1, 0b010_110_101) == bitor(0b010_110_101, t1)
      check: bitor(t1, t2) == [0, 3, 63, 1023, -97].toTensor
      check: bitor(t1, t2) == bitor(t1, t2)

    test "bitxor":
      let t1 = [0, 1, 57, 1022, -100].toTensor
      let t2 = [0, 2, 7, 15, 11].toTensor
      check: bitxor(t1, 0b010_110_101) == [181, 180, 140, 843, -215].toTensor
      check: bitxor(t1, 0b010_110_101) == bitxor(0b010_110_101, t1)
      check: bitxor(t1, t2) == [0, 3, 62, 1009, -105].toTensor
      check: bitxor(t1, t2) == bitxor(t1, t2)

    test "reverse_bits":
      let t = [0, 1, 57, 1022].toTensor(uint16)
      let expected = [0, 32768, 39936, 32704].toTensor(uint16)
      check: t.reverse_bits == expected

main()
GC_fullCollect()
