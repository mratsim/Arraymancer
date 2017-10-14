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

import ./p_kernels_generate

include incl_accessors_cuda,
        incl_kernels_cuda,
        incl_higher_order_cuda

# Design: due to limitations and difficulties in C++
# (no mutable iterators, stack hidden from GC, exception issue, casting workaround for builtin_assume_aligned)
# I choose to sacrifice readability here to avoid a plethora of "when not defined(cpp) in the codebase.
# This file will serve as a bridge between Cuda C++ and Nim C.

# Explanation:
#
# - To avoid creating a proc AST from scratch: we create a dummy proc.
# - pass it to an overloading template (cuda_genkernel_assign for in-place/assignments)
# - overloading template creates the cuda code, the import and a Nim wrapper
# - Unfortunately the Nim wrapper cannot be called as-is in other file as it uses C++ semantics
# - So we compile this file first to C++, rename it to .cu and tell nvcc to use the new .cu file

# Important: this file will be compiled to C++, only primitive types like pointers and float should be used,
# other here be dragons during linking with C code.


cuda_genkernel_assign("cuda_mAdd", "mAddOp", cuda_mAdd_f32, cuda_mAdd_f64)
cuda_genkernel_assign("cuda_mSub", "mSubOp", cuda_mSub_f32, cuda_mSub_f64)
cuda_genkernel_assign("cuda_unsafeContiguous", "CopyOp", cuda_unsfeContiguous_f32, cuda_unsafeContiguous_f64)
cuda_genkernel_binary("cuda_Add", "AddOp", cuda_Add_f32, cuda_Add_f64)
cuda_genkernel_binary("cuda_Sub", "SubOp", cuda_Sub_f32, cuda_Sub_f64)