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

## Pack A panels without padding
proc pack_MRxk[T](k: int,
                        A: seq[T], offA: var int,
                        incRowA, incColA: int,
                        buffer: var ref array[MCKC, T],
                        offBuf: var int) =

  var voffA = offA
  for j in 0 ..< k:
    for i in 0 ..< MR:
      buffer[i + offBuf] = A[i*incRowA + voffA]
    offBuf += MR
    voffA   += incColA

## Pack A with padding if needed
proc pack_A[T](mc, kc: int,
                     A: seq[T], offA: int,
                     incRowA, incColA: int,
                     buffer: var ref array[MCKC, T]) =

  let mp = mc div MR
  let mr = mc mod MR

  var offBuf = 0
  var voffA = offA

  for i in 0..<mp:
    pack_MRxk(kc, A, voffA, incRowA, incColA, buffer, offBuf)
    voffA  += MR*incRowA

  if mr > 0:
    for j in 0 ..< kc:
      for i in 0 ..< mr:
        buffer[i + offBuf] = A[i * incRowA + voffA]
      for i in mr ..< MR:
        buffer[i + offBuf] = 0.T
      offBuf += MR
      voffA   += incColA

## Pack B panels without padding
proc pack_kxNR[T](k: int,
                        B: seq[T], offB: int,
                        incRowB, incColB: int,
                        buffer: var ref array[KCNC, T],
                        offBuf: var int) =
  var voffB = offB
  for i in 0 ..< k:
    for j in 0 ..< NR:
      buffer[j + offBuf] = B[j*incColB + voffB]
    offBuf += NR
    voffB   += incRowB

## Pack B panels with padding if needed
proc pack_B[T](kc, nc: int,
                     B: seq[T], offB: int,
                     incRowB, incColB: int,
                     buffer: var ref array[KCNC, T]) =

  let np = nc div NR
  let nr = nc mod NR

  var offBuf = 0
  var voffB = offB

  for j in 0 ..< np:
    pack_kxNR(kc, B, voffB, incRowB, incColB, buffer, offBuf)
    voffB  += NR*incColB

  if nr > 0:
    for i in 0 ..< kc:
      for j in 0 ..< nr:
        buffer[j + offBuf] = B[j*incColB + voffB]
      for j in nr ..< NR:
        buffer[j + offBuf] = 0.T
      offBuf += NR
      voffB  += incRowB