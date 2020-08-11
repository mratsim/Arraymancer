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

import ./data_structure
import sequtils

proc toRawSeq*[T](t:Tensor[T]): seq[T] {.noSideEffect.} =
  ## Convert a tensor to the raw sequence of data.

  # Due to forward declaration this proc must be declared
  # after "cpu" proc are declared in init_cuda
  when t is Tensor:
    return t.data
  elif t is CudaTensor:
    return t.cpu.data

proc export_tensor*[T](t: Tensor[T]):
  tuple[shape: seq[int], strides: seq[int], data: seq[T]] {.noSideEffect.}=
  ## Export the tensor as a tuple containing
  ## - shape
  ## - strides
  ## - data
  ## If the tensor was not contiguous (a slice for example), it is reshaped.
  ## Data is exported in C order (last index changes the fastest, column in 2D case)

  let contig_t = t.asContiguous

  result.shape = contig_t.shape
  result.strides = contig_t.strides
  result.data = contig_t.data

proc toSeq2D*[T](t: Tensor[T]): seq[seq[T]] =
  ## Exports a rank-2 tensor to a 2D sequence.
  if t.rank != 2:
    raise newException(ValueError, "Tensor must be of rank 2")
  result = newSeqWith(
    t.shape[0], newSeq[T](t.shape[1])
  )
  for i in 0 ..< t.shape[0]:
    for j in 0 ..< t.shape[1]:
      result[i][j] = t[i, j]

proc toSeq3D*[T](t: Tensor[T]): seq[seq[seq[T]]] =
  ## Exports a rank-3 tensor to a 3D sequence.
  if t.rank != 3:
    raise newException(ValueError, "Tensor must be of rank 3")
  result = newSeqWith(
    t.shape[0], newSeqWith(
      t.shape[1], newSeq[T](t.shape[2])
    )
  )
  for i in 0 ..< t.shape[0]:
    for j in 0 ..< t.shape[1]:
      for k in 0 ..< t.shape[2]:
        result[i][j][k] = t[i, j, k]

proc toSeq4D*[T](t: Tensor[T]): seq[seq[seq[seq[T]]]] =
  ## Exports a rank-4 tensor to a 4D sequence.
  if t.rank != 4:
    raise newException(ValueError, "Tensor must be of rank 4")
  result = newSeqWith(
    t.shape[0], newSeqWith(
      t.shape[1], newSeqWith(
        t.shape[2], newSeq[T](t.shape[3])
      )
    )
  )
  for i in 0 ..< t.shape[0]:
    for j in 0 ..< t.shape[1]:
      for k in 0 ..< t.shape[2]:
        for l in 0 ..< t.shape[3]:
          result[i][j][k][l] = t[i, j, k, l]

proc toSeq5D*[T](t: Tensor[T]): seq[seq[seq[seq[seq[T]]]]] =
  ## Exports a rank-5 tensor to a 5D sequence.
  if t.rank != 5:
    raise newException(ValueError, "Tensor must be of rank 5")
  result = newSeqWith(
    t.shape[0], newSeqWith(
      t.shape[1], newSeqWith(
        t.shape[2], newSeqWith(
          t.shape[3], newSeq[T](t.shape[4])
        )
      )
    )
  )
  for i in 0 ..< t.shape[0]:
    for j in 0 ..< t.shape[1]:
      for k in 0 ..< t.shape[2]:
        for l in 0 ..< t.shape[3]:
          for m in 0 ..< t.shape[4]:
            result[i][j][k][l][m] = t[i, j, k, l, m]
