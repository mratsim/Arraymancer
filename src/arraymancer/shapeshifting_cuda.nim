# Copyright 2017 Mamy André-Ratsimbazafy
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

proc transpose*(t: CudaTensor): CudaTensor =
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Warning ⚠ CudaTensor temporary default:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data

proc cuda_asContiguous = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_asContiguous, "CopyOp")

proc asContiguous*[T: SomeReal](t: CudaTensor[T], layout: OrderType = colMajor, force: bool = false):
  CudaTensor[T] =
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default CudaTensor will be colMajor (contrary to a cpu tensor).
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout
  ##
  ## Warning ⚠ CudaTensor temporary default:
  ##   If the CudaTensor is contiguous, this is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  if t.isContiguous and not force:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t

  result = newCudaTensor[T](t.shape, layout)

  cuda_assign_call(cuda_asContiguous, result, t)