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
        ../data_structure, ../accessors_macros_syntax,
        ./p_checks, ./p_accessors,
        ../../std_version_types,
        sequtils, macros

from ../init_cpu import toTensor

template slicerImpl*[T](result: AnyTensor[T]|var AnyTensor[T], slices: ArrayOfSlices): untyped =
  ## Slicing routine

  when compileOption("boundChecks"):
    if unlikely(slices.len > result.rank):
      raise newException(
        IndexDefect,
        "Rank of slice expression (" & $slices.len &
          ") is larger than the tensor rank. Tensor shape" & $result.shape & "."
      )

  for i, slice in pairs(slices): # explicitly calling `pairs` is required for ident resolution
    # Check if we start from the end
    let a = if slice.a_from_end: result.shape[i] - slice.a
            else: slice.a

    let b = if slice.b_from_end: result.shape[i] - slice.b
            else: slice.b

    # Bounds checking
    when compileOption("boundChecks"):
      check_start_end(a, b, result.shape[i])
      check_steps(a, b, slice.step)
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

type FancySelectorKind* = enum
  FancyNone
  FancyIndex
  FancyMaskFull
  FancyMaskAxis
  # Workaround needed for https://github.com/nim-lang/Nim/issues/14021
  FancyUnknownFull
  FancyUnknownAxis

proc getFancySelector*(ast: NimNode, axis: var int, selector: var NimNode): FancySelectorKind =
  ## Detect indexing in the form
  ##   - "tensor[_, _, [0, 1, 4], _, _]
  ##   - "tensor[_, _, [0, 1, 4], `...`]
  ##  or with the index selector being a tensor
  result = FancyNone
  var foundNonSpanOrEllipsis = false
  var ellipsisAtStart = false

  template checkNonSpan(): untyped {.dirty.} =
    doAssert not foundNonSpanOrEllipsis,
        "Fancy indexing is only compatible with full spans `_` on non-indexed dimensions" &
        " and/or ellipsis `...`"

  var i = 0
  while i < ast.len:
    let cur = ast[i]

    # Important: sameType doesn't work for generic type like Array, Seq or Tensors ...
    #            https://github.com/nim-lang/Nim/issues/14021
    if cur.sameType(bindSym"SteppedSlice") or cur.isInt():
      if cur.eqIdent"Span":
        discard
      else:
        doAssert result == FancyNone
        foundNonSpanOrEllipsis = true
    elif cur.sameType(bindSym"Ellipsis"):
      if i == ast.len - 1: # t[t.sum(axis = 1) >. 0.5, `...`]
        doAssert not ellipsisAtStart, "Cannot deduce the indexed/sliced dimensions due to ellipsis at the start and end of indexing."
        ellipsisAtStart = false
      elif i == 0: # t[`...`, t.sum(axis = 0) >. 0.5]
        ellipsisAtStart = true
      else:
        # t[0 ..< 10, `...`, t.sum(axis = 0) >. 0.5] is unsupported
        # so we tag as "foundNonSpanOrEllipsis"
        foundNonSpanOrEllipsis = true
    elif cur.kind == nnkBracket:
      checkNonSpan()
      axis = i
      if cur[0].kind == nnkIntLit:
        result = FancyIndex
        selector = cur
      elif cur[0].isBool():
        let full = i == 0 and ast.len == 1
        result = if full: FancyMaskFull else: FancyMaskAxis
        selector = cur
      else:
        # byte, char, enums are all represented by integers in the VM
        error "Fancy indexing is only possible with integers or booleans"
    else:
      checkNonSpan()
      axis = i
      let full = i == 0 and ast.len == 1
      result = if full: FancyUnknownFull else: FancyUnknownAxis
      selector = cur
    inc i

  # Handle ellipsis at the start
  if result != FancyNone and ellipsisAtStart:
    axis = ast.len - axis

  # replace all possible `nnkSym` by `idents` because we otherwise might get
  # type mismatches
  selector = replaceSymsByIdents(selector)

macro slice_typed_dispatch*(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    result = newCall(bindSym"atIndex", t)
    for slice in args:
      result.add(slice)
    return

  # Fancy indexing
  # -----------------------------------------------------------------
  # Cannot depend/bindSym the "selectors.nim" proc
  # Due to recursive module dependencies
  var selector: NimNode
  var axis: int
  let fancy = args.getFancySelector(axis, selector)
  if fancy == FancyIndex:
    return newCall(
        ident"index_select",
        t, newLit axis, selector
      )
  if fancy == FancyMaskFull:
    return newCall(
        ident"masked_select",
        t, selector
      )
  elif fancy == FancyMaskAxis:
    return newCall(
        ident"masked_axis_select",
        t, selector, newLit axis
      )

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"slicer", t)
    for slice in args:
      if isInt(slice):
        ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_select = ident"masked_select"
  let lateBind_masked_axis_select = ident"masked_axis_select"
  let lateBind_index_select = ident"index_select"

  result = quote do:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    when FancyTensorType is Tensor[bool]:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_select`(`t`, `selector`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_select`(`t`, `selector`, `axis`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_select`(`t`, `axis`, `selector`)
