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



# Data structure to ease interfacing with Cuda and kernels
# This needs -d:cuda compilation flag to work


# MAXDIMS is defined in Arraymancer's global_config.nim
# Unfortunately const cannot be exportc by Nim so we use a template to emit the code with the const

## So that layout->strides can be used in Cuda kernel, it's easier if everything is declared from cpp
## pending https://github.com/nim-lang/Nim/issues/6415
#
# template create_CudaTensorLayout(N: static[int]) =
#   ## This Layout in C++ will be overriden by a CudaMemCpy from the Nim data structure
#   {. emit:[ """
# 
#     template <typename T>
#     struct CudaTensorLayout {
#       int rank;
#       int shape[""", N,"""];
#       int strides[""", N,"""];
#       int offset;
#       T * __restrict__ data;
#       };
# 
# 
#   """].}
# 
# create_CudaTensorLayout(MAXDIMS)

type
  CudaTensorLayout [T: SomeReal] = object
    ## Mimicks CudaTensor
    ## This will be stored on GPU in the end
    ## Goald is to avoids clumbering proc with cudaMemcpyshape, strides, offset, data, rank, len
    ## Also using arrays instead of seq avoids having to indicate __restrict__ everywhere to indicate no-aliasing
    ## Check https://github.com/mratsim/Arraymancer/issues/26 (Optimizing Host <-> Cuda transfer)
    ## on why I don't (yet?) use Unified Memory and choose to manage it manually.

    rank: cint               # Number of dimension of the tensor
    shape: array[MAXDIMS, cint]
    strides: array[MAXDIMS, cint]
    offset: cint
    data: ptr T              # Data on Cuda device
    len: cint                # Number of elements allocated in memory