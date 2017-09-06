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


proc cudaMalloc[T](size: int): ptr T =
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc freeCudaTensor[T](p: ref[ptr T]) =
  if not p[].isNil:
    check cudaFree(p[])

template tensorCuda*[T](out_shape: openarray[int], t: CudaTensor[T]) =
  new(t.data_ref, finalizer = freeCudaTensor)
  t.shape = @out_shape
  t.data_ptr = cudaMalloc[T](t.shape.product)
  t.data_ref[] = t.data_ptr
  t.strides = shape_to_strides(t.shape)
  t.offset = 0

proc newCudaTensor*[SR: SomeReal](shape: openarray[int], T: typedesc[SR]): CudaTensor[T] {.inline.}=
  tensorCuda[T](shape, result)