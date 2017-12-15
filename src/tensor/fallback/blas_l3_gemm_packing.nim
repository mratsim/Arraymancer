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

proc pack_panel[T](k: int,
                    M: seq[T], offset: int, # Tensor data + offset
                    lsm, ssm: int, # Leading and secondary (dimension) stride of M, Leading: incColA/incRowB.
                    LR: static[int], # Leading block dimension, MR for A (MxK), NR for B (KxN)
                    buffer: var BlasBufferArray[T], # N = MCKC for A, KCNC for B
                    offBuf: var int) {.noSideEffect.} =
  ## Pack blocks of size LR of the matrices in the corresponding buffer
  var offM = offset
  for s in 0 ..< k: # Loop along the leaing dimension
    for lead in 0 ..< LR:
      buffer[lead + offBuf] = M[lead*lsm + offM]
    offBuf += LR
    offM += ssm

proc pack_dim[T](lc, kc: int, # lc = mc for A (MxK matrix) and lc = nc for B (KxN matrix)
                  M: seq[T], offset: int, # Tensor data + offset
                  lsm, ssm: int, # Leading and secondary (dimension) stride of M, Leading: incColA/incRowB.
                  LR: static[int], # Leading block dimension, MR for A (MxK), NR for B (KxN)
                  buffer: var BlasBufferArray[T]) # N = MCKC for A, KCNC for B
                  {.noSideEffect.} =

  let lp = lc div LR # Number of whole blocks along leading dim
  let lr = lc mod LR # Reminder of leading dim

  var offBuf = 0
  var offM = offset

  for lead in 0..<lp:
    pack_panel(kc, M, offM, lsm, ssm, LR, buffer, offBuf)
    offM  += LR*lsm

  if lr > 0:
    for s in 0 ..< kc:
      for lead in 0 ..< lr:
        buffer[lead + offBuf] = M[lead * lsm + offM]
      for lead in lr ..< LR:
        buffer[lead + offBuf] = 0.T
      offBuf += LR
      offM   += ssm