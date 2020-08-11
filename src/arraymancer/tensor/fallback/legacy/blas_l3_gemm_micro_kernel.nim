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

import  macros,
        ../../backend/memory_optimization_hints

macro unroll_ukernel[MRNR, T](AB: array[MRNR, T],
                              a: ptr UncheckedArray[T], offA: int,
                              b: ptr UncheckedArray[T], offB: int,
                              MR, NR: static[int]): untyped =
  ## Unroll the following at compile time
  # for j in 0 ..< NR:
  #   for i in 0 ..< MR:
  #     AB[i + j*MR] += A[i + voffA] * B[j + voffB]
  result = newNimNode(nnkStmtList)
  for j in 0..NR-1:
    for i in 0..MR-1:
      result.add quote do:
        `AB`[`i`+`j`*`MR`] += `a`[`i`+`offA`] * `b`[`j`+`offB`]

template gemm_micro_kernelT[T](
            kc: int,
            alpha: T,
            A: typed, offA: int,
            B: typed, offB: int,
            beta: T,
            C: typed,
            offC: int,
            incRowC, incColC: int): untyped =

  ## Template and use FORCE_ALIGN
  withMemoryOptimHints()
  var AB{.align64.}: array[MRNR, T]
  var voffA = offA
  var voffB = offB

  var a {.restrict.}= assume_aligned(A.data, FORCE_ALIGN)
  var b {.restrict.}= assume_aligned(B.data, FORCE_ALIGN)

  ## Compute A*B
  for _ in 0 ..< kc:
    # for j in 0 ..< NR:
    #   for i in 0 ..< MR:
    #     AB[i + j*MR] += A[i + voffA] * B[j + voffB]
    unroll_ukernel(AB, a, voffA, b, voffB, MR, NR)
    voffA += MR
    voffB += NR

  when C is BlasBufferArray:
    var c {.restrict.}= assume_aligned(C.data, FORCE_ALIGN)
  elif C is seq:
    var c {.restrict.}= cast[ptr UncheckedArray[type C[0]]](addr C[0])

  ## C <- beta * C
  if beta == 0.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        c[i*incRowC + j*incColC + offC] = 0.T
  elif beta != 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        c[i*incRowC + j*incColC + offC] *= beta

  ## C <- C + alpha*AB, alpha !=0
  if alpha == 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        c[i*incRowC + j*incColC + offC] += AB[i + j*MR]
  else:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        c[i*incRowC + j*incColC + offC] += alpha*AB[i + j*MR]

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: BlasBufferArray[T], offA: int,
                          B: BlasBufferArray[T], offB: int,
                          beta: T,
                          C: var BlasBufferArray[T],
                          offC: int,
                          incRowC, incColC: int)
                          {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offC, incRowC, incColc)

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: BlasBufferArray[T], offA: int,
                          B: BlasBufferArray[T], offB: int,
                          beta: T,
                          C: var seq[T],
                          offC: int,
                          incRowC, incColC: int)
                          {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offC, incRowC, incColc)
