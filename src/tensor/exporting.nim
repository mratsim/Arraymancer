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

  let contig_t = t.unsafeContiguous

  result.shape = contig_t.shape
  result.strides = contig_t.strides
  result.data = contig_t.data