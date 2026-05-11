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

import  ../data_structure,
        ./opencl_global_state,
        ./global_config,
        nimcl, opencl, clblast, macros

export nimcl, opencl, opencl_global_state, clblast


# Data structures to ease interfacing with OpenCL and kernels

proc toClpointer*[T](p: ClStorage[T]): PMem {.noSideEffect, inline.}=
  cast[PMem](p.Fdata)

proc toClpointer*[T](p: ClTensor[T]): PMem {.noSideEffect, inline.}=
  cast[PMem](p.storage.Fdata)

proc clMalloc*[T](size: Natural): ptr UncheckedArray[T] {.inline.}=
  ## Internal proc.
  ## Wrap OpenCL createBuffer
  cast[type result](
    buffer[T](clContext0, size)
  )



# ##############################################################
# # Base ClStorage type

proc newClStorage*[T: SomeFloat](length: int): ClStorage[T] =
  result.Flen = length
  new result.Fref_tracking
  result.Fdata = clMalloc[T](result.Flen)
  result.Fref_tracking.value = result.Fdata

# #########################################################
# # Sending tensor layout to OpenCL Kernel

type
  ClLayoutArrayObj* = object
    value*: ptr UncheckedArray[cint]
  ClLayoutArray* = ref ClLayoutArrayObj
    ## Reference to an array on the device
    # TODO: finalizer
    # or replace by a distinct type with a destructor

  ClTensorLayout [T: SomeFloat] = object
    ## Mimicks CudaTensor
    ## Metadata stored on GPU or Accelerators

    rank*: cint               # Number of dimension of the tensor
    shape*: ClLayoutArray
    strides*: ClLayoutArray
    offset*: cint
    data*: ptr T              # Data on OpenCL device
    len*: cint                # Number of elements allocated in memory

when NimMajor == 1:
  proc `=destroy`*(p: var ClLayoutArrayObj) {.noSideEffect.}=
    if not p.value.isNil:
      discard releaseMemObject p.value.toClpointer
else:
  proc `=destroy`*(p: ClLayoutArrayObj) {.noSideEffect.}=
    if not p.value.isNil:
      discard releaseMemObject p.value.toClpointer

proc layoutOnDevice*[T:SomeFloat](t: ClTensor[T]): ClTensorLayout[T] =
  ## Store a ClTensor shape, strides, etc information on the GPU
  #
  # TODO: instead of storing pointers to shape/stride/etc that are passed to each kernel
  # pass the layout object directly and call it with layout->shape, layout->rank

  result.rank = t.rank.cint

  result.offset = t.offset.cint
  result.data = t.get_data_ptr
  result.len = t.size.cint

  new result.shape
  new result.strides

  result.shape.value = clMalloc[cint](MAXRANK)
  result.strides.value = clMalloc[cint](MAXRANK)

  var
    tmp_shape: array[MAXRANK, cint] # ClLayoutArray
    tmp_strides: array[MAXRANK, cint] # ClLayoutArray

  for i in 0..<t.rank:
    tmp_shape[i] = t.shape[i].cint
    tmp_strides[i] = t.strides[i].cint


  # TODO: use streams and async
  let size = t.rank * sizeof(cint)
  check enqueueWriteBuffer(
    clQueue0,
    result.shape.value.toClpointer,
    CL_true, # Blocking copy, we don't want tmp_shape to disappear whil copy is pending
    0,
    size,
    addr tmp_shape[0],
    0, nil, nil
  )

  check enqueueWriteBuffer(
    clQueue0,
    result.strides.value.toClpointer,
    CL_true, # Blocking copy, we don't want tmp_strides to disappear whil copy is pending
    0,
    size,
    addr tmp_strides[0],
    0, nil, nil
  )
