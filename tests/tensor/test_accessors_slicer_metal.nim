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

suite "Metal Accessors and Slicing":
  test "Basic indexing":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    # Shape should be preserved
    check: metal_tensor.shape == [3, 3]

  test "Slice with range":
    let cpu_tensor = [[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    let sliced = metal_tensor[0..1, 1..2]
    let back_to_cpu = sliced.toCpu()

    check: back_to_cpu == cpu_tensor[0..1, 1..2]

  test "Slice with step":
    let cpu_tensor = [[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    let sliced = metal_tensor[0..3|2, 0..3|2]
    let back_to_cpu = sliced.toCpu()

    check: back_to_cpu == cpu_tensor[0..3|2, 0..3|2]

  test "Slice with underscore":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    let row = metal_tensor[1, _]
    let back_to_cpu = row.toCpu()

    check: back_to_cpu == cpu_tensor[1, _]

  test "Metadata preservation":
    let cpu_tensor = [[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]].toTensor
    let metal_tensor = cpu_tensor.toMetal()

    check: metal_tensor.rank == cpu_tensor.rank
    check: metal_tensor.size == cpu_tensor.size
