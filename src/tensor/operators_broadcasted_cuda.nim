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

import  ./private/p_init_cuda,
        ./private/p_kernels_interface_cuda,
        ./private/p_checks,
        ./data_structure,
        ./higher_order,
        ./shapeshifting_cuda,
        ./operators_blas_l1_cuda

include ./private/incl_accessors_cuda,
        ./private/incl_higher_order_cuda,
        ./private/incl_kernels_cuda

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division
cuda_binary_glue("cuda_Mul", "MulOp", cuda_Mul)
cuda_binary_glue("cuda_Div", "DivOp", cuda_Div)

proc `.+`*[T: SomeReal](a, b: CudaTensor[T]): CudaTensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a + tmp_b

proc `.-`*[T: SomeReal](a, b: CudaTensor[T]): CudaTensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a - tmp_b


proc `.*`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.

  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)

  result = newCudaTensor[T](tmp_a.shape)
  cuda_binary_call(cuda_Mul, result, tmp_a, tmp_b)

proc `./`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## CudaTensor substraction

  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)

  result = newCudaTensor[T](tmp_a.shape)
  cuda_binary_call(cuda_Div, result, tmp_a, tmp_b)

# ##############################################
# # Broadcasting in-place Tensor-Tensor

cuda_assign_glue("cuda_mMulOp", "mMulOp", cuda_mMulOp)
cuda_assign_glue("cuda_mDivOp", "mDivOp", cuda_mDivOp)

proc `.+=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  a += tmp_b

proc `.-=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor broadcasted in-place substraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  a -= tmp_b

proc `.*=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  cuda_assign_call(cuda_mMulOp, a, tmp_b)

proc `./=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor broadcasted in-place float division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  cuda_assign_call(cuda_mDivOp, a, tmp_b)

# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

cuda_rscal_glue("cuda_rscalSub","RscalSub", cuda_rscalSub)
cuda_rscal_glue("cuda_rscalAdd","RscalAdd", cuda_rscalAdd)

proc `.+`*[T: SomeReal](t: CudaTensor[T], val: T): CudaTensor[T] {.noInit.} =
  ## Broadcasted addition for scalar + tensor.
  result = newCudaTensor[T](t.shape)
  cuda_rscal_call(cuda_rscalAdd, result, t, val)

proc `.-`*[T: SomeReal](t: CudaTensor[T], val: T): CudaTensor[T] {.noInit.} =
  ## Broadcasted substraction for scalar - tensor.
  result = newCudaTensor[T](t.shape)
  cuda_rscal_call(cuda_rscalSub, result, t, val)


cuda_lscal_glue("cuda_lscalSub","LscalSub", cuda_lscalSub)
cuda_lscal_glue("cuda_lscalAdd","LscalAdd", cuda_lscalAdd)
cuda_lscal_glue("cuda_lscalDiv","LscalDiv", cuda_lscalDiv)

proc `.+`*[T: SomeReal](val: T, t: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## Broadcasted addition for tensor + scalar.
  result = newCudaTensor[T](t.shape)
  cuda_lscal_call(cuda_lscalAdd, result, val, t)

proc `.-`*[T: SomeReal](val: T, t: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## Broadcasted substraction for tensor - scalar.
  result = newCudaTensor[T](t.shape)
  cuda_lscal_call(cuda_lscalSub, result, val, t)

proc `./`*[T: SomeReal](val: T, t: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## Broadcasted division of a float by a tensor of floats.
  result = newCudaTensor[T](t.shape)
  cuda_lscal_call(cuda_lscalDiv, result, val, t)

# ##############################################
# # Broadcasting in-place Tensor-Scalar
cuda_assignscal_glue("cuda_mscalSub","mscalSubOp", cuda_mscalSub)
cuda_assignscal_glue("cuda_mscalAdd","mscalAddOp", cuda_mscalAdd)

proc `.+=`*[T: SomeReal](t: var CudaTensor[T], val: T) =
  ## Broadcasted addition for scalar + tensor.
  cuda_assignscal_call(cuda_mscalAdd, t, val)

proc `.-=`*[T: SomeReal](t: var CudaTensor[T], val: T) =
  ## Broadcasted substraction for scalar - tensor.
  cuda_assignscal_call(cuda_mscalSub, t, val)