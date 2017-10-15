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


# Collection of cuda basic element-wise operations
# to be use by higher-order functions.
# The end-goal is to have a macro/template that can auto-generate these from:
#
# elementwise:
#   C = (A + B*sin(D))/exp(-X)
#
# __ldg is a cuda intrinsics to load read-only data
# from a special cache

# Assignment op
# Does element-wise A[i] `op=` B[i]
template cuda_assign_op(op_name, op_symbol: string)=
  {.emit: ["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T *  __restrict__ src){
      *dst """,op_symbol,""" __ldg(src);
      }
  };
  """].}

# Assignment with scalars
template cuda_assignscal_op(op_name, op_symbol: string)=
  {.emit: ["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T *  __restrict__ scal){
      *dst """,op_symbol,""" scal;
      }
  };
  """].}

# Binary op
# Does C[i] = A[i] `op` B[i]
template cuda_binary_op(op_name, op_symbol: string)=
  {.emit:["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T *  __restrict__ A,
      const T *  __restrict__ B){
      *dst = __ldg(A)""", op_symbol, """ __ldg(B);
      }
  };
  """].}

# Binary op with scalar on the left
# Does C[i] = a `op` B[i]
template cuda_lscal_op(op_name, op_symbol: string)=
  {.emit:["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T alpha,
      const T *  __restrict__ B){
      *dst = alpha""", op_symbol, """ __ldg(B);
      }
  };
  """].}

# Binary op with scalar on the right
# Does C[i] = A[i] `op` beta
template cuda_rscal_op(op_name, op_symbol: string)=
  {.emit:["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T *  __restrict__ A,
      const T beta){
      *dst = __ldg(A)""", op_symbol, """ beta;
      }
  };
  """].}

# Unary op
# Does C[i] = op(A[i])
template cuda_unary_op(op_name, op_symbol: string)=
  {.emit:["""
  template<typename T>
  struct """,op_name,"""{
  __device__ __forceinline__ void operator()(
      T *  __restrict__ dst,
      const T *  __restrict__ src){
      *dst = """, op_symbol, """(__ldg(src));
      }
  };
  """].}

cuda_assign_op("CopyOp", "=")
cuda_assign_op("mAddOp", "+=")
cuda_assign_op("mSubOp", "-=")
cuda_assign_op("mMulOp", "*=")
cuda_assign_op("mDivOp", "/=")

cuda_assignscal_op("CopyScalOp", "=")
cuda_assignscal_op("mscalAddOp", "+=")
cuda_assignscal_op("mscalSubOp", "-=")
cuda_assignscal_op("mscalMulOp", "*=")
cuda_assignscal_op("mscalDivOp", "/=")

cuda_binary_op("AddOp", "+")
cuda_binary_op("SubOp", "-")
cuda_binary_op("MulOp", "*")
cuda_binary_op("DivOp", "/")

cuda_lscal_op("LscalMul","*")
cuda_lscal_op("LscalDiv","/")
cuda_lscal_op("LscalSub","-")

cuda_rscal_op("RscalDiv","/")
cuda_rscal_op("RscalSub","-")
cuda_rscal_op("RscalAdd","+")

cuda_unary_op("NegOp","-")
cuda_unary_op("ExpOp","exp")
cuda_unary_op("SinOp","sin")
cuda_unary_op("CosOp","cos")
cuda_unary_op("TanOp","tan")
cuda_unary_op("TanhOp","tanh")
