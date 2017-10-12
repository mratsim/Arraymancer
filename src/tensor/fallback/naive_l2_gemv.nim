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

import  ../data_structure,
        ../operators_blas_l1,
        ../private/p_accessors_cpp, # Workaround for C++ mitems codegen bug
        nimblas

# Notes on optimizing performance:
# Google: https://github.com/google/gemmlowp/blob/master/todo/fast-gemv.txt
# UlmBLAS: https://github.com/michael-lehn/ulmBLAS/blob/master/ulmblas/level2/gemv.tcc


proc naive_gemv_fallback*[T: SomeInteger](
          alpha: T,
          A: Tensor[T],
          x: Tensor[T],
          beta: T,
          y: var Tensor[T]) =
  ## y <- alpha * A * x + beta * y


  if alpha == 0.T and beta == 1.T: return

  # BLAS: scal (multiplication by a scalar)
  # WARNING: This will multiply all values, regardless of stepping.
  when not defined(cpp):
    for val in y.mitems:
      val *= beta
  else: ## C++ workaround
    var data = y.dataArray
    for offset, _ in y.offsetValues:
      data[offset] *= beta


  if alpha == 0.T:
    return

  # TODO: instead of a naive implementation use BLIS/ulmBLAS implementation with
  # - if A is colMajor, use fused axpy BLAS op
  # - if A is rowMajor, use fused dotu BLAS op
  # - packing

  # Naive implementation: split the matrices along vertical axis

  let cont_A = A.unsafeContiguous(rowMajor, force=true)
  # if A is C_contiguous (row-major) slices along the row are also contiguous
  # so we can use unsafeReshape and avoid allocation inside the for loop
  let colA = cont_A.shape[1]

  var i: int = 0
  for ai in cont_A.axis(0):
    y[i] += alpha * dot(ai.unsafeReshape(colA), x)
    i += 1