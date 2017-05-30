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

template gemm_micro_kernelT[T](
            kc: int,
            alpha: T,
            A: typed, offA: int,
            B: typed, offB: int,
            beta: T,
            C: typed,
            offC: int,
            incRowC, incColC: int): untyped =

  {.pragma: align16, codegenDecl: "$# $# __attribute__((aligned(16)))".}
  var AB{.align16.}: array[MR*NR, T]
  var voffA = offA
  var voffB = offB

  ## Compute A*B
  for _ in 0 ..< kc:
    for j in 0 ..< NR:
      for i in 0 .. < MR:
        AB[i + j*MR] += A[i + voffA] * B[j + voffB]
    voffA += MR
    voffB += NR

  ## C <- beta * C
  if beta == 0.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC + offC] = 0.T
  elif beta != 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC + offC] *= beta

  ## C <- C + alpha*AB, alpha !=0
  if alpha == 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC + offC] += AB[i + j*MR]
  else:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC + offC] += alpha*AB[i + j*MR]

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: ref array[MCKC, T], offA: int,
                          B: ref array[KCNC, T], offB: int,
                          beta: T,
                          C: var ref array[MRNR, T],
                          offC: int,
                          incRowC, incColC: int)
                          {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offC, incRowC, incColc)

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: ref array[MCKC, T], offA: int,
                          B: ref array[KCNC, T], offB: int,
                          beta: T,
                          C: var seq[T],
                          offC: int,
                          incRowC, incColC: int)
                          {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offC, incRowC, incColc)