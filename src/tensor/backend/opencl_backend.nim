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
        nimcl, opencl

export nimcl, opencl, opencl_global_state


# Data structures to ease interfacing with OpenCL and kernels

proc toClpointer*[T](p: ptr T|ptr UncheckedArray[T]): PMem {.noSideEffect.}=
  cast[PMem](p)

proc clMalloc*[T](size: Natural): ptr UncheckedArray[T] {.inline.}=
  ## Internal proc.
  ## Wrap OpenCL createBuffer
  cast[type result](
    buffer[T](clContext0, size)
  )

proc deallocCl*[T](p: ref[ptr UncheckedArray[T]]) {.noSideEffect.}=
  if not p[].isNil:
    check releaseMemObject p[].toClpointer

# ##############################################################
# # Base ClStorage type

proc newClStorage*[T: SomeReal](length: int): ClStorage[T] =
  result.Flen = length
  new(result.Fref_tracking, deallocCl)
  result.Fdata = clMalloc[T](result.Flen)
  result.Fref_tracking[] = result.Fdata