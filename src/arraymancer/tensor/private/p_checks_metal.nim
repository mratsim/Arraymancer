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

when defined(metal):
  import ../data_structure

  proc check_elementwise*[T,U](a: MetalTensor[T], b: MetalTensor[U]) {.inline.} =
    ## Check if element-wise operations can be applied to 2 MetalTensors
    if unlikely(a.shape != b.shape):
      raise newException(ValueError, "Both Tensors should have the same shape.\n Left-hand side has shape " &
                                     $a.shape & " while right-hand side has shape " & $b.shape)

  proc check_dot_prod*[T,U](a: MetalTensor[T], b: MetalTensor[U]) {.inline.} =
    ## Check if dot product can be applied to 2 MetalTensors
    if unlikely(a.rank != 1 or b.rank != 1):
      raise newException(ValueError, "Dot product is only supported for vectors (1D tensors)")
    if unlikely(a.shape[0] != b.shape[0]):
      raise newException(ValueError, "Both vectors should have the same length")

  proc check_matmat*[T,U](a: MetalTensor[T], b: MetalTensor[U]) {.inline.} =
    ## Check if matrix-matrix multiplication can be applied
    if unlikely(a.rank != 2 or b.rank != 2):
      raise newException(ValueError, "Matrix multiplication requires 2D tensors")
    if unlikely(a.shape[1] != b.shape[0]):
      raise newException(ValueError, "Incompatible shapes for matrix multiplication: " &
                                     $a.shape & " and " & $b.shape)

  proc check_matvec*[T,U](a: MetalTensor[T], b: MetalTensor[U]) {.inline.} =
    ## Check if matrix-vector multiplication can be applied
    if unlikely(a.rank != 2 or b.rank != 1):
      raise newException(ValueError, "Matrix-vector multiplication requires a 2D matrix and 1D vector")
    if unlikely(a.shape[1] != b.shape[0]):
      raise newException(ValueError, "Incompatible shapes for matrix-vector multiplication: " &
                                     $a.shape & " and " & $b.shape)
