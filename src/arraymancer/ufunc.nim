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

proc astype*[T, U](t: Tensor[T], typ: typedesc[U]): Tensor[U] {.noSideEffect.}=
  ## Apply type conversion on the whole tensor
  result = t.map(x => x.U)


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
  proc func_name*(t: Tensor): Tensor {.noSideEffect, inline.}=
    ## Auto-generated universal version of the function.
    ##
    ## The function can be used directly on tensors and will work element-wise.
    t.map(func_name)
  export func_name

template makeUniversalLocal*(func_name: untyped) =
  # Lift an unary function into a non-exported universal function.
  #
  # Universal functions apply element-wise.
  #
  # ``makeUniversalLocal`` does not work when the internal type of the Tensor changes,
  # for example, a function "isEven: int -> bool".
  # Use ``map`` in this case instead instead
  proc func_name(t: Tensor): Tensor {.noSideEffect, inline.}=
    t.map(func_name)

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
makeUniversal(tgamma)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(trunc)
makeUniversal(round)
#makeUniversal(splitDecimal)
makeUniversal(degToRad)
makeUniversal(radToDeg)

# Built-in nim functions that doesn't work with makeUniversal

proc abs*[T](t: Tensor[T]): Tensor[T] {.noSideEffect, inline.}=
  proc abs_proxy(x: T):T = abs(x)
  t.map(abs_proxy)
