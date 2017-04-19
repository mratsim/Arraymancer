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


proc fmap*[B: static[Backend],T, U](t: Tensor[B,T], g: T -> U): Tensor[B,U] {.noSideEffect.}=
    ## Map a unary function T -> U on Tensor[T]

    # First convert the offset pointer back to index
    result.shape = t.shape
    result.strides = t.strides
    result.offset = t.offset
    result.data = t.data.map(g)

template makeUniversal*(func_name: untyped) =
    ## Lift an unary function into an exported universal function.
    ## Universal functions apply element-wise
    # For now, makeUniversal does not work when internal type is changing
    # use fmap instead
    proc func_name*(t: Tensor): Tensor = t.fmap(func_name)
    export func_name

template makeUniversalLocal*(func_name: untyped) =
    ## Lift an unary function into a non-exported universal function
    ## Universal functions apply element-wise
    # For now, makeUniversalLocal does not work when internal type is changing
    # use fmap instead
    proc func_name(t: Tensor): Tensor = t.fmap(func_name)

## Unary functions from Nim math library

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