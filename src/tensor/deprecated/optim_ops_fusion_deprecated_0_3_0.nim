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

import  ../private/p_checks,
        ../private/p_init_cpu,
        ../data_structure


proc unsafeToTensorReshape*[T](data: seq[T], shape: varargs[int]): Tensor[T] {.noSideEffect, deprecated.} =
  ## Deprecated
  ##
  ## Fuse unsafeToTensor and unsafeReshape in one operation
  ##
  ## With move semantics + reference semantics this is not needed.
  # Note: once this is removed, CpuStorage can be changed to not expose Fdata.

  when compileOption("boundChecks"):
    check_nested_elements(shape.toMetadataArray, data.len)

  tensorCpu(shape, result)
  shallowCopy(result.storage.Fdata, data)

template rewriteUnsafeToTensorReshape*{unsafeReshape(unsafeToTensor(s), shape)}(
  s: seq,
  shape: varargs[int]): auto =
  ## Fuse ``sequence.unsafeToTensor().unsafeReshape(new_shape)`` into a single operation.
  ##
  ## Operation fusion leverage the Nim compiler and should not be called explicitly.
  unsafeToTensorReshape(s, shape, dummy_bugfix)
