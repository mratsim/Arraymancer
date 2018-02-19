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


import  ./backend/opencl_backend except check
import  clblast
# TODO error checking in Nim opencl is broken
# See https://github.com/nim-lang/opencl/pull/3

import  ./backend/metadataArray,
        ./private/p_kernels_interface_opencl,
        ./private/p_init_opencl,
        ./private/p_checks,
        ./data_structure

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

template dotImpl(T: typedesc, clblast_proc: untyped): untyped =
  proc dot*(a, b: ClTensor[T]): T =
    ## Vector to Vector dot (scalar) product
    when compileOption("boundChecks"):
      check_dot_prod(a,b)

    var clResult = newClStorage[T](1)

    check clblast_proc(a.size, clResult.toClpointer, 0,
          a.toClpointer, a.offset, a.strides[0],
          b.toClpointer, b.offset, b.strides[0],
          unsafeAddr clQueue0, nil)

    # TODO error checking in Nim opencl is broken
    # See https://github.com/nim-lang/opencl/pull/3
    let err2 = enqueueReadBuffer(
      clQueue0,
      clResult.toClpointer,
      CL_true, # Blocking copy, we don't want computation to continue while copy is still pending
      0,
      sizeof(result),
      result.addr.toClpointer,
      0, nil, nil
    )

    assert err2 == TClResult.SUCCESS

dotImpl(float32, clblastSdot)
dotImpl(float64, clblastDdot)


genClInfixOp(float32, "float", `+`, "clAdd", "+")
genClInfixOp(float64, "double", `+`, "clAdd", "+")
genClInfixOp(float32, "float", `-`, "clSub", "-")
genClInfixOp(float64, "double", `-`, "clSub", "-")

genClInPlaceOp(float32, "float", `+=`, "clAdd", "+=")
genClInPlaceOp(float64, "double", `+=`, "clAdd", "+=")
genClInPlaceOp(float32, "float", `-=`, "clSub", "-=")
genClInPlaceOp(float64, "double", `-=`, "clSub", "-=")