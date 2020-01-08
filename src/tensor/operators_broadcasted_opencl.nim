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

import  ./backend/opencl_backend,
        ./private/p_kernels_interface_opencl,
        ./private/p_init_opencl,
        ./private/p_checks,
        ./data_structure,
        ./shapeshifting_opencl,
        ./operators_blas_l1_opencl

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division
proc `+.`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a + tmp_b

proc `.+`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline, deprecated:"Use `+.` instead".} =
  a +. b

proc `-.`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a - tmp_b

proc `.-`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline, deprecated:"Use `-.` instead".} =
  a -. b

genClInfixOp(float32, "float", elwise_mul, "clMul", "*", exported = false)
genClInfixOp(float64, "double", elwise_mul, "clAdd", "*", exported = false)
genClInfixOp(float32, "float", elwise_div, "clSub", "/", exported = false)
genClInfixOp(float64, "double", elwise_div, "clSub", "/", exported = false)

proc `*.`*[T: SomeFloat](a,b: ClTensor[T]): ClTensor[T] {.noInit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = elwise_mul(tmp_a, tmp_b)

proc `.*`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline, deprecated:"Use `*.` instead".} =
  a *. b

proc `/.`*[T: SomeFloat](a,b: ClTensor[T]): ClTensor[T] {.noInit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = elwise_div(tmp_a, tmp_b)

proc `./`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] {.noInit,inline, deprecated:"Use `/.` instead".} =
  a /. b
