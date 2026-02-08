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

# Please compile with -d:metal switch
import ../../src/arraymancer
import std / unittest

suite "Metal Shapeshifting":
  test "Transpose":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    # Transpose should work by manipulating metadata
    let transposed = metal_tensor.transpose()
    let back_to_cpu = transposed.toCpu()

    check: back_to_cpu == cpu_tensor.transpose()

  test "Reshape":
    let cpu_tensor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    let reshaped = metal_tensor.reshape([2, 3])
    let back_to_cpu = reshaped.toCpu()

    check: back_to_cpu == cpu_tensor.reshape([2, 3])

  test "Slice":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    let sliced = metal_tensor[1..2, 0..1]
    let back_to_cpu = sliced.toCpu()

    check: back_to_cpu == cpu_tensor[1..2, 0..1]

  test "Contiguous check":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    check: metal_tensor.isContiguous == cpu_tensor.isContiguous

  test "As contiguous":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()
    let transposed = metal_tensor.transpose()

    let contiguous = transposed.asContiguous(rowMajor)
    let back_to_cpu = contiguous.toCpu()

    check: back_to_cpu == cpu_tensor.transpose().asContiguous(rowMajor)
