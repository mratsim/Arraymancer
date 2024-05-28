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

import  ../../laser/private/nested_containers,
        ../../private/ast_utils,
        ../data_structure, ../accessors_macros_syntax,
        ../higher_order_applymap,
        ./p_accessors_macros_read,
        ./p_checks,
        ./p_accessors,
        sequtils, macros

# #########################################################################
# Slicing macros - write access

## `RelaxedRankOne` is a CT variable exposed to the user to recover the old behavior
## of how rank 1 tensors are treated in mutating slices.
## If set to `false` using `-d:RelaxedRankOne=false`, slice assignments using rank 1
## arrays / seqs / tensors have to match exactly. If it is `true`, only the number of
## input elements have to match for a more convenient interface.
const RelaxedRankOne* {.booldefine.} = true

# #########################################################################
# Setting a single value

template slicerMutImpl_val[T](t: var Tensor[T], slices: ArrayOfSlices, val: T): untyped =
  var sliced = t.slicer(slices)
  for old_val in sliced.mitems:
    old_val = val

proc slicerMut*[T](t: var Tensor[T], slices: openArray[SteppedSlice], val: T) {.noSideEffect.}=
  ## Assign the value to the whole slice
  slicerMutImpl_val(t, slices.toArrayOfSlices, val)

proc slicerMut*[T](t: var Tensor[T],
                slices: openArray[SteppedSlice],
                ellipsis: Ellipsis,
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = slices.toArrayOfSlices & initSpanSlices(t.rank - slices.len)
  slicerMutImpl_val(t, full_slices, val)

proc slicerMut*[T](t: var Tensor[T],
                ellipsis: Ellipsis,
                slices: openArray[SteppedSlice],
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = initSpanSlices(t.rank - slices.len) & slices.toArrayOfSlices
  slicerMutImpl_val(t, full_slices, val)

proc slicerMut*[T](t: var Tensor[T],
                slices1: openArray[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: openArray[SteppedSlice],
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, Ellipsis, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = concat(slices1,
                            initSpanSlices(t.rank - slices1.len - slices2.len),
                            slices2)
  slicerMutImpl_val(t, full_slices, val)

# ###########################################################################
# Assign value from an openArray of the same shape

template slicerMutImpl_oa[T](t: var Tensor[T], slices: openArray[SteppedSlice], oa: openArray) =
  ## Assign value from openarrays

  var sliced = t.slicer(slices)
  when compileOption("boundChecks"):
    check_shape(sliced, oa, relaxed_rank1_check = RelaxedRankOne)

  var data = toSeq(flatIter(oa))
  when compileOption("boundChecks"):
    check_nested_elements(oa.getShape(), data.len)

  # Unfortunately we need to loop twice over data/oa
  # Reason 1: we can't check the iterator length before consuming it
  # Reason 2: we can't capture an open array, i.e. do zip(sliced.real_indices, flatClosureIter(oa))
  for i, x in sliced.menumerate:
    x = data[i]

proc slicerMut*[T](t: var Tensor[T], slices: openArray[SteppedSlice], oa: openArray) {.noSideEffect.}=
  ## Assign value from openArrays
  ## The openArray must have the same shape as the slice
  slicerMutImpl_oa(t, slices, oa)

proc slicerMut*[T](t: var Tensor[T],
                  slices: openArray[SteppedSlice],
                  ellipsis: Ellipsis,
                  oa: openArray) {.noSideEffect.}=
  ## Assign value from openArrays
  ## The openArray must have the same shape as the slice
  # TODO: tests
  let full_slices = slices.toArrayOfSlices & initSpanSlices(t.rank - slices.len)
  slicerMutImpl_oa(t, slices, oa)

proc slicerMut*[T](t: var Tensor[T],
                  ellipsis: Ellipsis,
                  slices: openArray[SteppedSlice],
                  oa: openArray) {.noSideEffect.}=
  ## Assign value from openArrays
  ## The openArray must have the same shape as the slice
  # TODO: tests
  let full_slices = initSpanSlices(t.rank - slices.len) & slices.toArrayOfSlices
  slicerMutImpl_oa(t, slices, oa)


proc slicerMut*[T](t: var Tensor[T],
                slices1: openArray[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: openArray[SteppedSlice],
                oa: openArray) {.noSideEffect.}=
  ## Assign value from openArrays
  ## The openArray must have the same shape as the slice
  # TODO: tests
  let full_slices = concat(slices1,
                            initSpanSlices(t.rank - slices1.len - slices2.len),
                            slices2)
  slicerMutImpl_oa(t, full_slices, val)

# #########################################################################
# Setting from a Tensor

template slicerMutImpl_T[T](t: var Tensor[T], slices: openArray[SteppedSlice], t2: Tensor[T]) =
  ## Assign the value to the whole slice

  var sliced = t.slicer(slices)

  when compileOption("boundChecks"):
    check_shape(sliced, t2, relaxed_rank1_check = RelaxedRankOne)

  apply2_inline(sliced, t2):
    y

proc slicerMut*[T](t: var Tensor[T], slices: openArray[SteppedSlice], t2: Tensor[T])=
  ## Assign the value to the whole slice
  slicerMutImpl_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  slices: openArray[SteppedSlice],
                  ellipsis: Ellipsis,
                  t2: Tensor[T]) =
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = slices.toArrayOfSlices & initSpanSlices(t.rank - slices.len)
  slicerMutImpl_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  ellipsis: Ellipsis,
                  slices: openArray[SteppedSlice],
                  t2: Tensor[T]) =
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = initSpanSlices(t.rank - slices.len) & slices.toArrayOfSlices
  slicerMutImpl_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  slices1: openArray[SteppedSlice],
                  ellipsis: Ellipsis,
                  slices2: openArray[SteppedSlice],
                  t2: Tensor[T]) =
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = concat(slices1,
                            initSpanSlices(t.rank - slices1.len - slices2.len),
                            slices2)
  slicerMutImpl_T(t, slices, t2)

# #########################################################################
# Dispatching logic

macro slice_typed_dispatch_mut*(t: typed, args: varargs[typed], val: typed): untyped =
  ## Assign `val` to Tensor T at slice/position `args`

  # Type check the argument for a saner error message
  checkValidSliceType(args)

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    result = newCall(bindSym"atIndexMut", t)
    for slice in args:
      result.add(slice)
    result.add(val)
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
        ident"index_fill",
        t, newLit axis, selector,
        val
      )
  if fancy == FancyMaskFull:
    return newCall(
        ident"masked_fill",
        t, selector,
        val
      )
  elif fancy == FancyMaskAxis:
    return newCall(
        ident"masked_axis_fill",
        t, selector, newLit axis,
        val
      )

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"slicerMut", t)
    sliceDispatchImpl(result, args, false)

    result.add(val)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_fill = ident"masked_fill"
  let lateBind_masked_axis_fill = ident"masked_axis_fill"
  let lateBind_index_fill = ident"index_fill"

  result = quote do:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    when FancyTensorType is Tensor[bool]:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_fill`(`t`, `selector`, `val`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_fill`(`t`, `selector`, `axis`, `val`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_fill`(`t`, `axis`, `selector`, `val`)

# ############################################################################
# Slicing a var returns a var (for Result[_] += support)
# And apply2(result[_], foo) support
#
# Unused: Nim support for var return types is problematic

proc slicer_var[T](t: var AnyTensor[T], slices: openArray[SteppedSlice]): var AnyTensor[T] {.noinit,noSideEffect.}=
  ## Take a Tensor and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  slicerImpl(result, slices)

proc slicer_var[T](t: var AnyTensor[T],
                slices: openArray[SteppedSlice],
                ellipsis: Ellipsis): var AnyTensor[T] {.noinit,noSideEffect.}=
  ## Take a Tensor, SteppedSlices and Ellipsis
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = slices.toArrayOfSlices & initSpanSlices(t.rank - slices.len)
  slicerImpl(result, full_slices)

proc slicer_var[T](t: var AnyTensor[T],
                ellipsis: Ellipsis,
                slices: openArray[SteppedSlice]
                ): var AnyTensor[T] {.noinit,noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = initSpanSlices(t.rank - slices.len) & slices.toArrayOfSlices
  slicerImpl(result, full_slices)

proc slicer_var[T](t: var AnyTensor[T],
                slices1: openArray[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: openArray[SteppedSlice]
                ): var AnyTensor[T] {.noinit,noSideEffect.}=
  ## Take a Tensor, Ellipsis and SteppedSlices
  ## Returns:
  ##    A copy of the original Tensor
  ##    Offset and strides are changed to achieve the desired effect.

  result = t
  let full_slices = concat(slices1.toArrayOfSlices,
                            initSpanSlices(t.rank - slices1.len - slices2.len),
                            slices1.toArrayOfSlices)
  slicerImpl(result, full_slices)

macro slice_typed_dispatch_var*(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though

  # Type check the argument for a saner error message
  checkValidSliceType(args)

  if isAllInt(args):
    result = newCall(bindSym("atIndex"), t)
    for slice in args:
      result.add(slice)
  else:
    result = newCall(bindSym("slicer_var"), t)
    sliceDispatchImpl(result, args, true)
