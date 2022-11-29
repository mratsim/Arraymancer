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

import  ./data_structure,
        ./higher_order_applymap,
        ./private/p_empty_tensors,
        sugar, math, complex

## NOTE: This should be `{.noinit.}`, but this is blocked by:
## https://github.com/nim-lang/Nim/issues/16253
proc astype*[T; U: not Complex](t: Tensor[T], typ: typedesc[U]): Tensor[U] =
  ## Apply type conversion on the whole tensor. This is a no-op if `T` is the same
  ## as `U`.
  when T is U:
    result = t
  else:
    returnEmptyIfEmpty(t)
    result = t.map(x => x.U)

## NOTE: This should be `{.noinit.}`, see above.
proc astype*[T: SomeNumber, U: Complex](t: Tensor[T], typ: typedesc[U]): Tensor[U] =
  ## Apply type conversion on the whole tensor
  returnEmptyIfEmpty(t)
  when T is SomeNumber and U is Complex[float32]:
    result = t.map(x => complex32(x.float32))
  elif T is SomeNumber and U is Complex[float64]:
    result = t.map(x => complex64(x.float64))
  else:
    {.error: "Unreachable".}

# #############################################################
# Autogen universal functions

# Note, the makeUniversal/Local documentation gets duplicated in docs at each template call
# And shouldn't use ##
template makeUniversal*(func_name: untyped) =
  # Lift an unary function into an exported universal function.
  #
  # Universal functions apply element-wise.
  #
  # ``makeUniversal`` does not work when the internal type of the Tensor changes,
  # for example, a function "isEven: int -> bool".
  # Use ``map`` in this case instead instead
  proc func_name*[T](t: Tensor[T]): Tensor[T] {.noinit.} =
    ## Auto-generated universal version of the function.
    ##
    ## The function can be used directly on tensors and will work element-wise.
    returnEmptyIfEmpty(t)
    t.map_inline(func_name(x))
  export func_name

template makeUniversalLocal*(func_name: untyped) =
  # Lift an unary function into a non-exported universal function.
  #
  # Universal functions apply element-wise.
  #
  # ``makeUniversalLocal`` does not work when the internal type of the Tensor changes,
  # for example, a function "isEven: int -> bool".
  # Use ``map`` in this case instead instead
  proc func_name[T](t: Tensor[T]): Tensor[T] {.noinit.} =
    returnEmptyIfEmpty(t)
    t.map_inline(func_name(x))

# Unary functions from Nim math library

makeUniversal(fac)
#makeUniversal(classify)
#makeUniversal(isPowerOfTwo)
#makeUniversal(nextPowerOfTwo)
#makeUniversal(countBits32)
#makeUniversal(sum)
makeUniversal(sqrt)
makeUniversal(cbrt)
makeUniversal(ln)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sinh)
makeUniversal(sin)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf)
makeUniversal(erfc)
makeUniversal(lgamma)
makeUniversal(gamma)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(trunc)
makeUniversal(round)
#makeUniversal(splitDecimal)
makeUniversal(degToRad)
makeUniversal(radToDeg)
