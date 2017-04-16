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
    ptrMath:
        # TODO: Thoroughly test this, especially with negative offsets
        let d0: ptr T = unsafeAddr(t.data[0])
        let offset_idx: int = t.offset - d0

    result.dimensions = t.dimensions
    result.strides = t.strides
    # TODO: Bounds checking - result[0, 0, 0 ...] is included, result[dim1.len, dim2.len, ...] is not
    result.data = t.data.map(g)

    ptrMath:
        result.offset = addr(result.data[0]) + offset_idx

template makeUniversal*(func_name: untyped) =
    ## Lift an unary function into an exported universal function.
    ## Universal functions apply element-wise
    # For now, makeUniversal does not work when internal type is changing
    proc func_name*(t: Tensor): Tensor = t.fmap(func_name)
    export func_name

template makeUniversalLocal*(func_name: untyped) =
    ## Lift an unary function into a non-exported universal function
    ## Universal functions apply element-wise
    # For now, makeUniversalLocal does not work when internal type is changing
    proc func_name(t: Tensor): Tensor = t.fmap(func_name)

## Unary functions from Nim math library
## For now, for functions with input type != result type would need fmap use

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