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

type
  CpuStorage*[T] = ref object
    ## This implements a opaque refcounted storage for copy-on-write.
    ## Data is shared between multiple tensors as long as none modifies the data.
    ## If a mutation is needed, it is done in-place if the Tensor is the only one referring to this storage.
    ## Otherwise the mutator copies the data and refers to his own copy.
    Frefcount: int
    Fdata*: seq[T] # This should only be visible internally

  CudaSeq* [T: SomeReal] = object
    ## Seq-like structure on the Cuda backend.
    ##
    ## Nim garbage collector will automatically ask cuda to clear GPU memory if ``data`` becomes unused.
    ##
    ## Warning ⚠: This will be revamped, renamed and kept private before 0.3.0 for copy-on-write semantics on Cuda.
    len*: int
    data*: ref[ptr UncheckedArray[T]]

  # Note, non-CPU storage are also forward-declared here so that AnyTensor
  # is always Tensor + CudaTensor + ...
  # The implementation requires cudaMalloc / cudaFree and cannot be done in this file
  # as it is also imported on for non-Cuda targets.


proc incRef*(store: CpuStorage){.inline.}=
  if not store.isNil: # If a tensor is in a wrapper like the autograd Variable it may not be initialized
    inc store.Frefcount

proc decRef*(store: CpuStorage){.inline.}=
  if not store.isNil: # We may swap unitialized storage with initialized storage and destroy it after
    dec store.Frefcount

proc initRef*(store: CpuStorage){.inline.}=
  store.Frefcount = 1

proc isUniqueRef*(store: CpuStorage){.inline.}=
  store.Frefcount == 1
