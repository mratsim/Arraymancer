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


import  ../backend/metal/metal_backend,
        ../data_structure

proc newMetalStorage*[T: SomeFloat](length: int): MetalStorage[T] =
  result.Flen = length
  result.Fbuffer = createMetalBuffer(length * sizeof(T))
  new result.Fref_tracking
  result.Fref_tracking.value = result.Fbuffer

template tensorMetal*[T: SomeFloat](
  shape: typed,
  layout: OrderType = colMajor,
  result: var MetalTensor[T]) =

  result.shape.copyFrom(shape)
  shape_to_strides(result.shape, layout, result.strides)
  result.offset = 0
  result.storage = newMetalStorage[T](result.size)
  result.cpuData = @[]

proc newMetalTensor*[T: SomeFloat](
  shape: varargs[int],
  layout: OrderType = colMajor): MetalTensor[T] {.noinit.} =
  ## Internal proc
  ## Allocate a MetalTensor
  ## WARNING: The Metal memory is not initialized to 0

  tensorMetal(shape, layout, result)

proc newMetalTensor*[T: SomeFloat](
  shape: Metadata,
  layout: OrderType = colMajor): MetalTensor[T] {.noinit.} =

  tensorMetal(shape, layout, result)

proc get_data_ptr*[T: SomeFloat](t: MetalTensor[T]): ptr T {.inline.} =
  ## Get access to the data pointer of a MetalTensor
  ## This is unsafe and should only be used internally
  cast[ptr T](t.storage.Fbuffer.devicePtr)

proc get_data_ptr*[T: SomeFloat](t: var MetalTensor[T]): ptr T {.inline.} =
  ## Get access to the data pointer of a MetalTensor
  ## This is unsafe and should only be used internally
  cast[ptr T](t.storage.Fbuffer.devicePtr)
