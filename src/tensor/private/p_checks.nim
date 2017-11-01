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

import  ../../private/[functional, nested_containers],
        ../backend/metadataArray,
        ../data_structure

proc check_nested_elements*(shape: seq[int], len: int) {.noSideEffect, inline.}=
  ## Compare the detected shape from flatten with the real length of the data
  ## Input:
  ##   -- A shape (sequence of int)
  ##   -- A length (int)
  if unlikely(shape.product != len):
    raise newException(IndexError, "Each nested sequence at the same level must have the same number of elements")

proc check_index*(t: Tensor, idx: varargs[int]) {.noSideEffect, inline.}=
  if unlikely(idx.len != t.rank):
    raise newException(IndexError, "Number of arguments: " &
                    $(idx.len) &
                    ", is different from tensor rank: " &
                    $(t.rank))

proc check_contiguous_index*(t: Tensor, idx: int) {.noSideEffect, inline.}=
  if unlikely(idx < 0 or idx >= t.size):
    raise newException(IndexError, "Invalid contigous index: " &
                    $idx &
                    " while tensor size is" &
                    $(t.size))

proc check_elementwise*(a, b:AnyTensor)  {.noSideEffect, inline.}=
  ## Check if element-wise operations can be applied to 2 Tensors
  if unlikely(a.shape != b.shape):
    raise newException(ValueError, "Both Tensors should have the same shape.\n Left-hand side has shape " &
                                   $a.shape & " while right-hand side has shape " & $b.shape)

proc check_size*[T,U](a:Tensor[T], b:Tensor[U])  {.noSideEffect, inline.}=
  ## Check if the total number of elements match

  if unlikely(a.size != b.size):
    raise newException(ValueError, "Both Tensors should have the same total number of elements.\n" &
      "Left-hand side has " & $a.size & " (shape: " & $a.shape & ") while right-hand side has " &
      $b.size & " (shape: " & $b.shape & ")."
    )

proc check_steps*(a,b, step:int) {.noSideEffect, inline.}=
  ## Though it might be convenient to automatically step in the correct direction like in Python
  ## I choose not to do it as this might introduce the typical silent bugs typechecking/Nim is helping avoid.

  # if unlikely(a == 0 and b == -1 and step == 1):
  #   # Very specific scenario to allow initialization of concatenation with empty dimension
  #   # like shape of (3, 0)
  #   return
  if unlikely((b-a) * step < 0):
    raise newException(IndexError, "Your slice start: " &
                $a & ", and stop: " &
                $b & ", or your step: " &
                $step &
                """, are not correct. If your step is positive
                start must be inferior to stop and inversely if your step is negative
                start must be superior to stop.""")

proc check_shape*(a, b: Tensor|openarray) {.noSideEffect, inline.}=
  ## Compare shape

  when b is Tensor:
    let b_shape = b.shape
  else:
    let b_shape = b.shape.toMetadataArray

  if unlikely(a.shape != b_shape):
    raise newException(IndexError, "Your tensors or openarrays do not have the same shape: " &
                                   $a.shape &
                                   " and " & $b_shape)

proc check_reshape*(t: AnyTensor, new_shape:MetadataArray) {.noSideEffect, inline.}=
  if unlikely(t.size != new_shape.product):
    raise newException(ValueError, "The total number of elements in the old (" &
                                    $t.size &
                                    ") and the new (" &
                                    $new_shape.product &
                                    ") reshaped tensor must be the same")

proc check_nocopyReshape*(t: AnyTensor) {.noSideEffect, inline.}=
  if unlikely(not t.isContiguous):
    raise newException(ValueError, "The tensor must be contiguous for reshape without copy")

proc check_concat*(t1, t2: Tensor, axis: int) {.noSideEffect,inline.}=
  let check1 = t1.shape[0..<axis] == t2.shape[0..<axis]
  let check2 = t2.shape[axis+1..t1.shape.high] == t2.shape[axis+1..t2.shape.high]

  if unlikely(not check1 or not check2):
    raise newException(ValueError, "Concatenation Error: Except along the concatenation axis tensors must have the same shape")

proc check_squeezeAxis*(t: AnyTensor, axis: int) {.noSideEffect, inline.}=
  if unlikely(axis >= t.rank):
    raise newException(ValueError, "The axis is out of range, axis is " & $axis & " while the tensor rank is " & $t.rank )

proc check_unsqueezeAxis*(t: AnyTensor, axis: int) {.noSideEffect, inline.}=
  if unlikely(t.rank == 0 or axis > t.rank or axis < 0):
    raise newException(ValueError, "The new axis is out of range, axis is " & $axis & " while the tensor rank is " & $t.rank )

proc check_dot_prod*(a, b:AnyTensor) {.noSideEffect, inline.}=
  if unlikely(a.rank != 1 or b.rank != 1): raise newException(ValueError, "Dot product is only supported for vectors (tensors of rank 1)")
  if unlikely(a.shape != b.shape): raise newException(ValueError, "Vector should be the same length")

proc check_matmat*(a, b: AnyTensor) {.noSideEffect, inline.}=
  let colA = a.shape[1]
  let rowB = b.shape[0]

  if unlikely(colA != rowB):
    raise newException(IndexError, "Number of columns in the first matrix: " &
                    $(colA) &
                    ", must be the same as the number of rows in the second matrix: " &
                    $(rowB))

proc check_matvec*(a, b: AnyTensor) {.noSideEffect, inline.}=
  let colA = a.shape[1]
  let rowB = b.shape[0]

  if unlikely(colA != rowB):
    raise newException(IndexError, "Number of columns in the matrix: " &
                    $(colA) &
                    ", must be the same as the number of rows in the vector: " &
                    $(rowB))

proc check_axis_index*(t: AnyTensor, axis, index: int) {.noSideEffect, inline.}=
  if unlikely(not (axis < t.rank and index < t.shape[axis])):
    raise newException(IndexError, "The axis is out of range, axis requested is " &
                                    $axis &
                                    " and index requested " & $index &
                                    " while tensor shape is " & $(t.shape))
