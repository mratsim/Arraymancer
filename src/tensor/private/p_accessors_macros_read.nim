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

import  ../../private/ast_utils,
        ../data_structure, ../init_cpu, ../accessors_macros_syntax,
        ../backend/metadataArray,
        ./p_checks, ./p_accessors, ./p_accessors_macros_desugar,
        sequtils, macros, strformat

template slicerImpl*[T](result: AnyTensor[T]|var AnyTensor[T], slices: ArrayOfSlices): untyped =
  ## Slicing routine

  when compileOption("boundChecks"):
    if unlikely(slices.len > result.rank):
      raise newException(
        IndexError,
        &"Rank of slice expression ({slices.len}) is larger then the tensor rank ({result.rank})."
      )

  for i, slice in slices:
    # Check if we start from the end
    let a = if slice.a_from_end: result.shape[i] - slice.a
            else: slice.a

    let b = if slice.b_from_end: result.shape[i] - slice.b
            else: slice.b

    # Bounds checking
    when compileOption("boundChecks"): check_steps(a, b, slice.step)
    ## TODO bounds-check the offset or leave the default?
    ## The default only checks when we retrieve the value

    # Compute offset:
    result.offset += a * result.strides[i]
    # Now change shape and strides
    result.strides[i] *= slice.step
    result.shape[i] = abs((b-a) div slice.step) + 1

proc slicer*[T](t: AnyTensor[T], slices: varargs[SteppedSlice]): AnyTensor[T] {.noInit,noSideEffect.}=
  ## Take a Tensor and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  slicerImpl(result, slices.toArrayOfSlices)

proc slicer*[T](t: AnyTensor[T],
                slices: varargs[SteppedSlice],
                ellipsis: Ellipsis): AnyTensor[T] {.noInit,noSideEffect.}=
  ## Take a Tensor, SteppedSlices and Ellipsis
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = slices.toArrayOfSlices & initSpanSlices(t.rank - slices.len)
  slicerImpl(result, full_slices)

proc slicer*[T](t: AnyTensor[T],
                ellipsis: Ellipsis,
                slices: varargs[SteppedSlice]
                ): AnyTensor[T] {.noInit,noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = initSpanSlices(t.rank - slices.len) & slices.toArrayOfSlices
  slicerImpl(result, full_slices)

proc slicer*[T](t: AnyTensor[T],
                slices1: varargs[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: varargs[SteppedSlice]
                ): AnyTensor[T] {.noInit,noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = concat(slices1.toArrayOfSlices,
                            initSpanSlices(t.rank - slices1.len - slices2.len),
                            slices2.toArrayOfSlices)
  slicerImpl(result, full_slices)

proc slicer*[T](t: Tensor[T], slices: ArrayOfSlices): Tensor[T] {.noInit,noSideEffect.}=
  ## Take a Tensor and SteppedSlices
  ## Returns:
  ##    A view of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.
  ##    Warning: mutating the result will mutate the original
  ##    As such a `var Tensor` is required

  result = t
  slicerImpl(result, slices)

# #########################################################################
# Dispatching logic

macro slice_typed_dispatch*(t: typed, args: varargs[typed]): untyped =
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