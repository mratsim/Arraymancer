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
proc asType*[T; U: not Complex](t: Tensor[T], typ: typedesc[U]): Tensor[U] =
  ## Apply type conversion on the whole tensor. This is a no-op if `T` is the same
  ## as `U`.
  when T is U:
    result = t
  else:
    returnEmptyIfEmpty(t)
    result = t.map(x => x.U)

## NOTE: This should be `{.noinit.}`, see above.
proc asType*[T: SomeNumber, U: Complex](t: Tensor[T], typ: typedesc[U]): Tensor[U] =
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

import std / [strutils, macros]
macro genDocstring(name, suffix: untyped): untyped =
  var text = """
Applies `$#` to every element of the input tensor `t` and returns a copy.
""" % name.strVal
  if suffix.strVal.len > 0:
    text.add "\n" & (suffix.strVal)
  result = newNimNode(nnkCommentStmt)
  result.strVal = text

template makeUniversal*(func_name: untyped, docSuffix = "") =
  ## Lift an unary function into an exported universal function.
  ##
  ## Universal functions apply element-wise.
  ##
  ## ``makeUniversal`` does not work when the internal type of the Tensor changes,
  ## for example, a function "isEven: int -> bool".
  ## Use ``map`` in this case instead instead
  proc func_name*[T](t: Tensor[T]): Tensor[T] {.noinit.} =
    genDocstring(func_name, docSuffix) # generate the doc string
    returnEmptyIfEmpty(t)
    t.map_inline(func_name(x))
  export func_name

template makeUniversalLocal*(func_name: untyped) =
  ## Lift an unary function into a non-exported universal function.
  ##
  ## Universal functions apply element-wise.
  ##
  ## ``makeUniversalLocal`` does not work when the internal type of the Tensor changes,
  ## for example, a function "isEven: int -> bool".
  ## Use ``map`` in this case instead instead
  proc func_name[T](t: Tensor[T]): Tensor[T] {.noinit.} =
    returnEmptyIfEmpty(t)
    t.map_inline(func_name(x))

# Unary functions from Nim math library

makeUniversal(fac,
  docSuffix="Note that the scalar [fac](https://nim-lang.org/docs/math.html#fac%2Cint) function computes the factorial of its input, which must be a non-negative integer.")
makeUniversal(isNaN)
#makeUniversal(isPowerOfTwo)
#makeUniversal(nextPowerOfTwo)
#makeUniversal(countBits32)
#makeUniversal(sum)
makeUniversal(sqrt,
  docSuffix="Note that the scalar [sqrt](https://nim-lang.org/docs/math.html#sqrt%2Cfloat64) function computes the square root of its input.")
makeUniversal(cbrt,
  docSuffix="Note that the scalar [cbrt](https://nim-lang.org/docs/math.html#cbrt%2Cfloat64) function computes the cube root of its input.")
makeUniversal(ln)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(arccosh)
makeUniversal(arcsinh)
makeUniversal(arctanh)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sinh)
makeUniversal(sin)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf,
  docSuffix="Note that the scalar [erf](https://nim-lang.org/docs/math.html#erf%2Cfloat64) function computes the Gauss [error function](https://en.wikipedia.org/wiki/Error_function) of its input.")
makeUniversal(erfc,
  docSuffix="Note that the scalar [erfc](https://nim-lang.org/docs/math.html#erfc%2Cfloat64) function computes the [complementary error function](https://en.wikipedia.org/wiki/Error_function#Complementary_error_function) of its input.")
makeUniversal(gamma,
  docSuffix="Note that the scalar [gamma](https://nim-lang.org/docs/math.html#lgamma%2Cfloat64) function computes the [gamma function](https://en.wikipedia.org/wiki/Gamma_function) of its input.")
makeUniversal(lgamma,
  docSuffix="Note that the scalar [lgamma](https://nim-lang.org/docs/math.html#lgamma%2Cfloat64) function computes the natural logarithm of the [gamma function](https://en.wikipedia.org/wiki/Gamma_function) of its input.")
makeUniversal(floor,
  docSuffix="Note that the scalar [floor](https://nim-lang.org/docs/math.html#floor%2Cfloat64) function returns the largest integer not greater than its input to the decimal point.")
makeUniversal(ceil,
  docSuffix="Note that the scalar [ceil](https://nim-lang.org/docs/math.html#ceil%2Cfloat64) function returns the smallest integer not smaller than its input to the decimal point.")
makeUniversal(trunc,
  docSuffix="Note that the scalar [trunc](https://nim-lang.org/docs/math.html#trunc%2Cfloat64) function truncates its input to the decimal point.")
makeUniversal(round,
  docSuffix="Note that the scalar [round](https://nim-lang.org/docs/math.html#round%2Cfloat64) function rounds its input to the closest integer. \n" &
  "Unlike nim's standard library version, this function does not take a `places` argument. If you need to round to a specific number of decimal places, use `map_inline` instead (e.g. `t.map_inline(round(x, places = 3))`).")
#makeUniversal(splitDecimal)
makeUniversal(degToRad)
makeUniversal(radToDeg)
