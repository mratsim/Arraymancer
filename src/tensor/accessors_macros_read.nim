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

# #########################################################################
# Slicing macros - read access

import  ../private/ast_utils,
        ./private/[p_checks, p_accessors, p_accessors_macros_desugar],
        ./init_cpu,
        ./accessors_macros_syntax

template slicerT[T](result: AnyTensor[T], slices: varargs[SteppedSlice]): untyped=
  ## Slicing routine

  for i, slice in slices:
    # Check if we start from the end
    let a = if slice.a_from_end: result.shape[i] - slice.a
            else: slice.a

    let b = if slice.b_from_end: result.shape[i] - slice.b
            else: slice.b

    # Bounds checking
    when compileOption("boundChecks"): check_steps(a,b, slice.step)
    ## TODO bounds-check the offset or leave the default?
    ## The default only checks when we retrieve the value

    # Compute offset:
    result.offset += a * result.strides[i]
    # Now change shape and strides
    result.strides[i] *= slice.step
    result.shape[i] = abs((b-a) div slice.step) + 1

proc slicer[T](t: AnyTensor[T], slices: varargs[SteppedSlice]): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  slicerT(result, slices)

proc slicer[T](t: AnyTensor[T],
                slices: varargs[SteppedSlice],
                ellipsis: Ellipsis): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, SteppedSlices and Ellipsis
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = @slices & newSeqWith(t.rank - slices.len, span)
  slicerT(result, full_slices)

proc slicer[T](t: AnyTensor[T],
                ellipsis: Ellipsis,
                slices: varargs[SteppedSlice]
                ): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = newSeqWith(t.rank - slices.len, span) & @slices
  slicerT(result, full_slices)

proc slicer[T](t: AnyTensor[T],
                slices1: varargs[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: varargs[SteppedSlice]
                ): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = concat(@slices1,
                            newSeqWith(t.rank - slices1.len - slices2.len, span),
                            @slices2)
  slicerT(result, full_slices)

proc unsafeSlicer[T](t: Tensor[T], slices: varargs[SteppedSlice]): Tensor[T] {.noSideEffect.}=
  ## Take a Tensor and SteppedSlices
  ## Returns:
  ##    A view of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.
  ##    Warning: mutating the result will mutate the original
  ##    As such a `var Tensor` is required
  ## WARNING: passing a non-var Tensor is unsafe

  result = unsafeView(t)
  slicerT(result, slices)


proc unsafeSlicer[T](t: AnyTensor[T],
                      slices: varargs[SteppedSlice],
                      ellipsis: Ellipsis): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, SteppedSlices and Ellipsis
  ## Returns:
  ##    A view of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.
  ##    Warning: mutating the result will mutate the original
  ##    As such a `var Tensor` is required
  ## WARNING: passing a non-var Tensor is unsafe

  result = unsafeView(t)
  let full_slices = @slices & newSeqWith(t.rank - slices.len, span)
  slicerT(result, full_slices)

proc unsafeSlicer[T](t: AnyTensor[T],
                      ellipsis: Ellipsis,
                      slices: varargs[SteppedSlice]
                      ): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A view of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.
  ##    Warning: mutating the result will mutate the original
  ##    As such a `var Tensor` is required
  ## WARNING: passing a non-var Tensor is unsafe

  result = unsafeView(t)
  let full_slices = newSeqWith(t.rank - slices.len, span) & @slices
  slicerT(result, full_slices)

proc unsafeSlicer[T](t: AnyTensor[T],
                      slices1: varargs[SteppedSlice],
                      ellipsis: Ellipsis,
                      slices2: varargs[SteppedSlice]
                      ): AnyTensor[T] {.noSideEffect.}=
  ## Take a Tensor, SteppedSlices, Ellipsis and SteppedSlices
  ## Returns:
  ##    A view of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.
  ##    Warning: mutating the result will mutate the original
  ##    As such a `var Tensor` is required
  ## WARNING: passing a non-var Tensor is unsafe

  result = unsafeView(t)
  let full_slices = concat(@slices1,
                            newSeqWith(t.rank - slices1.len - slices2.len, span),
                            @slices2)
  slicerT(result, full_slices)

# #########################################################################
# Dispatching logic

macro inner_typed_dispatch(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though
  if isAllInt(args):
    result = newCall(bindSym("atIndex"), t)
    for slice in args:
      result.add(slice)
  else:
    result = newCall(bindSym("slicer"), t)
    for slice in args:
      if isInt(slice):
        ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)

macro `[]`*[T](t: AnyTensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor or a CudaTensor
  ## Input:
  ##   - a Tensor or a CudaTensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a tensor corresponding to the slice
  ## Warning ⚠ CudaTensor temporary default:
  ##   For CudaTensor only, this is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ## Usage:
  ##    - Basic indexing - foo[2, 3]
  ##    - Basic indexing - foo[1+1, 2*2*1]
  ##    - Basic slicing - foo[1..2, 3]
  ##    - Basic slicing - foo[1+1..4, 3-2..2]
  ##    - Span slices - foo[_, 3]
  ##    - Span slices - foo[1.._, 3]
  ##    - Span slices - foo[_..3, 3]
  ##    - Span slices - foo[_.._, 3]
  ##    - Stepping - foo[1..3\|2, 3]
  ##    - Span stepping - foo[_.._\|2, 3]
  ##    - Span stepping - foo[_.._\|+2, 3]
  ##    - Span stepping - foo[1.._\|1, 2..3]
  ##    - Span stepping - foo[_..<4\|2, 3]
  ##    - Slicing until at n from the end - foo[0..^4, 3]
  ##    - Span Slicing until at n from the end - foo[_..^2, 3]
  ##    - Stepped Slicing until at n from the end - foo[1..^1\|2, 3]
  ##    - Slice from the end - foo[^1..0\|-1, 3]
  ##    - Slice from the end - expect non-negative step error - foo[^1..0, 3]
  ##    - Slice from the end - foo[^(2*2)..2*2, 3]
  ##    - Slice from the end - foo[^3..^2, 3]
  let new_args = getAST(desugar(args))

  result = quote do:
    inner_typed_dispatch(`t`, `new_args`)

macro unsafe_inner_typed_dispatch(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though
  if isAllInt(args):
    result = newCall(bindSym("atIndex"), t)
    for slice in args:
      result.add(slice)
  else:
    result = newCall(bindSym("unsafeSlicer"), t)
    for slice in args:
      if isInt(slice):
        ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)

macro unsafeSlice*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a view of the Tensor corresponding to the slice
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ## Usage:
  ##   See the ``[]`` macro
  let new_args = getAST(desugar(args))

  result = quote do:
    unsafe_inner_typed_dispatch(`t`, `new_args`)