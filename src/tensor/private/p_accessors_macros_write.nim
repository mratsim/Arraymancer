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

import  ../../private/[nested_containers, ast_utils],
        ../data_structure, ../accessors_macros_syntax,
        ./p_accessors_macros_desugar,
        ./p_accessors_macros_read,
        ./p_checks,
        ./p_accessors,
        sequtils, macros

# #########################################################################
# Slicing macros - write access

# #########################################################################
# Setting a single value

template slicerMutT_val[T](t: var Tensor[T], slices: varargs[SteppedSlice], val: T): untyped =
  var sliced = t.unsafeSlicer(slices)
  for old_val in sliced.mitems:
    old_val = val

proc slicerMut*[T](t: var Tensor[T], slices: varargs[SteppedSlice], val: T) {.noSideEffect.}=
  ## Assign the value to the whole slice
  slicerMutT_val(t, slices, val)

proc slicerMut*[T](t: var Tensor[T],
                slices: varargs[SteppedSlice],
                ellipsis: Ellipsis,
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = @slices & newSeqWith(t.rank - slices.len, span)
  slicerMutT_val(t, full_slices, val)

proc slicerMut*[T](t: var Tensor[T],
                ellipsis: Ellipsis,
                slices: varargs[SteppedSlice],
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = newSeqWith(t.rank - slices.len, span) & @slices
  slicerMutT_val(t, full_slices, val)

proc slicerMut*[T](t: var Tensor[T],
                slices1: varargs[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: varargs[SteppedSlice],
                val: T) {.noSideEffect.}=
  ## Take a var Tensor, Ellipsis, SteppedSlices, Ellipsis and a value
  ## Assign the value to the whole slice
  # TODO: tests

  let full_slices = concat(@slices1,
                            newSeqWith(t.rank - slices1.len - slices2.len, span),
                            @slices2)
  slicerMutT_val(t, full_slices, val)

# ###########################################################################
# Assign value from an openarray of the same shape

template slicerMutT_oa[T](t: var Tensor[T], slices: varargs[SteppedSlice], oa: openarray) =
  ## Assign value from openarrays
  var sliced = t.unsafeSlicer(slices)
  when compileOption("boundChecks"):
    check_shape(sliced, oa)

  let data = toSeq(flatIter(oa))
  when compileOption("boundChecks"):
    check_nested_elements(oa.shape, data.len)

  # Unfortunately we need to loop twice over data/oa
  # Reason 1: we can't check the iterator length before consuming it
  # Reason 2: we can't capture an open array, i.e. do zip(sliced.real_indices, flatClosureIter(oa))
  for i, x in sliced.menumerate:
    x = data[i]


proc slicerMut*[T](t: var Tensor[T], slices: varargs[SteppedSlice], oa: openarray) {.noSideEffect.}=
  ## Assign value from openarrays
  ## The openarray must have the same shape as the slice
  slicerMutT_oa(t, slices, oa)

proc slicerMut*[T](t: var Tensor[T],
                  slices: varargs[SteppedSlice],
                  ellipsis: Ellipsis,
                  oa: openarray) {.noSideEffect.}=
  ## Assign value from openarrays
  ## The openarray must have the same shape as the slice
  # TODO: tests
  let full_slices = @slices & newSeqWith(t.rank - slices.len, span)
  slicerMutT_oa(t, slices, oa)

proc slicerMut*[T](t: var Tensor[T],
                  ellipsis: Ellipsis,
                  slices: varargs[SteppedSlice],
                  oa: openarray) {.noSideEffect.}=
  ## Assign value from openarrays
  ## The openarray must have the same shape as the slice
  # TODO: tests
  let full_slices = newSeqWith(t.rank - slices.len, span) & @slices
  slicerMutT_oa(t, slices, oa)


proc slicerMut*[T](t: var Tensor[T],
                slices1: varargs[SteppedSlice],
                ellipsis: Ellipsis,
                slices2: varargs[SteppedSlice],
                oa: openarray) {.noSideEffect.}=
  ## Assign value from openarrays
  ## The openarray must have the same shape as the slice
  # TODO: tests
  let full_slices = concat(@slices1,
                            newSeqWith(t.rank - slices1.len - slices2.len, span),
                            @slices2)
  slicerMutT_oa(t, full_slices, val)

# #########################################################################
# Setting from a Tensor

template slicerMutT_T[T](t: var Tensor[T], slices: varargs[SteppedSlice], t2: Tensor[T]) =
  ## Assign the value to the whole slice
  var sliced = t.unsafeSlicer(slices)

  when compileOption("boundChecks"):
    check_shape(sliced, t2)

  for x, val in mzip(sliced, t2):
    x = val

proc slicerMut*[T](t: var Tensor[T], slices: varargs[SteppedSlice], t2: Tensor[T]) {.noSideEffect.}=
  ## Assign the value to the whole slice
  slicerMutT_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  slices: varargs[SteppedSlice],
                  ellipsis: Ellipsis,
                  t2: Tensor[T]) {.noSideEffect.}=
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = @slices & newSeqWith(t.rank - slices.len, span)
  slicerMutT_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  ellipsis: Ellipsis,
                  slices: varargs[SteppedSlice],
                  t2: Tensor[T]) {.noSideEffect.}=
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = newSeqWith(t.rank - slices.len, span) & @slices
  slicerMutT_T(t, slices, t2)

proc slicerMut*[T](t: var Tensor[T],
                  slices1: varargs[SteppedSlice],
                  ellipsis: Ellipsis,
                  slices2: varargs[SteppedSlice],
                  t2: Tensor[T]) {.noSideEffect.}=
  ## Assign the value to the whole slice
  # TODO: tests
  let full_slices = concat(@slices1,
                            newSeqWith(t.rank - slices1.len - slices2.len, span),
                            @slices2)
  slicerMutT_T(t, slices, t2)

# #########################################################################
# Dispatching logic

macro inner_typed_dispatch_mut*(t: typed, args: varargs[typed], val: typed): untyped =
  ## Assign `val` to Tensor T at slice/position `args`
  if isAllInt(args):
    result = newCall(bindSym("atIndexMut"), t)
    for slice in args:
      result.add(slice)
    result.add(val)
  else:
    result = newCall(bindSym("slicerMut"), t)
    for slice in args:
      if isInt(slice):
        ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)
    result.add(val)