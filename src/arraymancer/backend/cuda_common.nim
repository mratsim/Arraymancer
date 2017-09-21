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


# This file is included for all compilation target (Cpu, CUDA, ...) so that typeclass AnyTensor works properly
# Data structure to ease interfacing with Cuda
type
  CudaSeq*[T: SomeReal] = object
    ## Cuda-Seq like structure
    ## End goal is for it to have value semantics like Nim seq
    ## and optimize to not copy if referenced only once
    len: int
    data: ptr array[0,T]