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

import  ../private/sequninit,
        ../std_version_types,
        ./private/p_init_opencl,
        ./backend/opencl_backend,
        ./data_structure,
        ./init_cpu

proc opencl*[T:SomeFloat](t: Tensor[T]): ClTensor[T] {.noinit.}=
  ## Convert a tensor on Cpu to a tensor on an OpenCL device.

  result = newClTensor[T](t.shape)

  let contig_t = t.asContiguous(rowMajor, force = true)
  let size = result.size * sizeof(T)

  # TODO error checking in Nim opencl is broken
  # See https://github.com/nim-lang/opencl/pull/3

  check enqueueWriteBuffer(
    clQueue0,
    result.get_data_ptr.toClpointer,
    CL_true, # Blocking copy, we don't want contig_t to disappear while copy is pending
    0,
    size,
    contig_t.get_data_ptr.toClpointer,
    0, nil, nil
  )

proc cpu*[T:SomeFloat](t: ClTensor[T]): Tensor[T] {.noinit.}=
  ## Convert a tensor on an OpenCL device to a tensor on Cpu.
  # We use blocking copy in this case to make sure
  # all data is available for future computation

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.storage.allocCpuStorage(t.storage.Flen)

  let size = t.storage.Flen * sizeof(T)

  # TODO error checking in Nim opencl is broken
  # See https://github.com/nim-lang/opencl/pull/3

  check enqueueReadBuffer(
    clQueue0,
    t.get_data_ptr.toClpointer,
    CL_true, # Blocking copy, we don't want computation to continue while copy is still pending
    0,
    size,
    result.get_data_ptr.toClpointer,
    0, nil, nil
  )

proc zeros_like*[T: SomeFloat](t: ClTensor[T]): ClTensor[T] {.noinit, inline.} =
  ## Creates a new ClTensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the CudaTensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed ClTensor of the same shape

  # TODO use clEnqueueFillBuffer (OpenCL 1.2 only)
  result = zeros[T](t.shape).opencl

proc ones_like*[T: SomeFloat](t: ClTensor[T]): ClTensor[T] {.noinit, inline.} =
  ## Creates a new ClTensor filled with 1 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - A CudaTensor
  ## Result:
  ##      - A one-ed ClTensor of the same shape
  result = ones[T](t.shape).opencl
