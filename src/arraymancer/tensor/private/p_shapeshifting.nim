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

import  ../../laser/tensor/[allocator, initialization],
        ../../private/sequninit,
        ../data_structure, ../higher_order_applymap,
        ../init_cpu,
        ./p_checks,
        nimblas

proc contiguousImpl*[T](t: Tensor[T], layout: OrderType, result: var Tensor[T]) =
  if layout == rowMajor:
    result = t.map_inline(x)
  else: # colMajor
    var size: int
    initTensorMetadata(result, size, t.shape, colMajor)
    allocCpuStorage(result.storage, size)
    apply2_inline(result, t):
      y

proc reshape_with_copy*[T](t: Tensor[T], new_shape: varargs[int]|Metadata, result: var Tensor[T]) =
  result = newTensorUninit[T](new_shape)
  result.apply2_inline(t,y)

proc reshape_no_copy*(t: AnyTensor, new_shape: varargs[int]|Metadata, result: var AnyTensor, layout: OrderType) {.noSideEffect.}=
  result.shape.copyFrom(new_shape)
  shape_to_strides(result.shape, layout, result.strides)
  result.offset = t.offset

proc reshapeImpl*(t: AnyTensor, new_shape: varargs[int]|Metadata, result: var AnyTensor) =
  when compileOption("boundChecks"):
    check_reshape(t, new_shape)

  if t.is_C_contiguous:
    reshape_no_copy(t, new_shape, result, rowMajor)
    result.storage = t.storage
  elif t.is_F_contiguous:
    reshape_no_copy(t, new_shape, result, colMajor)
    result.storage = t.storage
  else:
    reshape_with_copy(t, new_shape, result)

proc broadcastImpl*(t: var AnyTensor, shape: varargs[int]|Metadata) {.noSideEffect.}=
  when compileOption("boundChecks"):
    assert t.rank == shape.len

  for i in 0..<t.rank:
    if t.shape[i] == 1:
      if shape[i] != 1:
        t.shape[i] = shape[i]
        t.strides[i] = 0
    elif t.shape[i] != shape[i]:
      # TODO: $varargs is not supported on stable
      var strShape = "["
      for i in shape:
        if i != 0: strShape.add ','
        strShape.add ' '
        strShape.add $i
      strShape.add ']'
      raise newException(ValueError, "The broadcasted size of the tensor " & strShape &
        ", must match existing size " & $t.shape & " for non-singleton dimension")

proc broadcast2Impl*[T](a, b: AnyTensor[T], result: var tuple[a, b: AnyTensor[T]]) {.noSideEffect.}=
  let rank = max(a.rank, b.rank)

  var shapeA, stridesA, shapeB, stridesB = Metadata(len: rank) # initialized with 0

  for i in 0..<rank:
    let shape_A_iter = if i < rank: a.shape[i] else: 1
    let shape_B_iter = if i < rank: b.shape[i] else: 1

    if shape_A_iter == shape_B_iter:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = a.strides[i]
      stridesB[i] = b.strides[i]

    elif shape_A_iter == 1:
      shapeA[i] = shape_B_iter
      shapeB[i] = shape_B_iter

      # stridesA[i] is already 0
      stridesB[i] = b.strides[i]
    elif shape_B_iter == 1:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = a.strides[i]
      # stridesB[i] is already 0
    else:
      raise newException(ValueError, "Broadcasting error: non-singleton dimensions must be the same in both tensors. Tensors' shapes were: " &
        $a.shape & " and " & $b.shape)

  result.a.shape = shapeA
  result.a.strides = stridesA
  result.a.offset = a.offset

  result.b.shape = shapeB
  result.b.strides = stridesB
  result.b.offset = b.offset


proc exch_dim*[T](t: Tensor[T], dim1, dim2: int): Tensor[T] {.noInit,noSideEffect.}=
  if dim1 == dim2:
    return

  result = t # copy or no-copy is managed in the caller of exch_dim or permuteImpl
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permuteImpl*[T](result: var Tensor[T], dims: varargs[int]) {.noSideEffect.} =
  var perm = dims.toMetadata
  for i, p in perm:
    if p != i and p != -1:
      var j = i
      while true:
        result = exch_dim(result, j, perm[j])
        (perm[j], j) = (-1, perm[j])
        if perm[j] == i:
          break
      perm[j] = -1

proc squeezeImpl*(t: var AnyTensor) {.noSideEffect.} =
  var idx_real_dim = 0

  for i in 0..<t.rank:
    if t.shape[i] != 1:
      if i != idx_real_dim:
        t.shape[idx_real_dim] = t.shape[i]
        t.strides[idx_real_dim] = t.strides[i]
      inc idx_real_dim

  t.shape = t.shape[0..<idx_real_dim]
  t.strides = t.strides[0..<idx_real_dim]

proc squeezeImpl*(t: var AnyTensor, axis: int) {.noSideEffect.} =
  when compileOption("boundChecks"):
    check_squeezeAxis(t, axis)

  if t.rank > 1 and t.shape[axis] == 1: # We don't support rank 0 Tensor
    t.shape.delete(axis)
    t.strides.delete(axis)

proc unsqueezeImpl*(t: var AnyTensor, axis: int) {.noSideEffect.} =
  when compileOption("boundChecks"):
    check_unsqueezeAxis(t, axis)

  # set the stride to be consistent with the rest of the lib
  var stride: int
  if axis >= t.rank:
    stride = 1
  else:
    stride = t.shape[axis]*t.strides[axis]

  t.shape.insert(1, axis)
  t.strides.insert(stride, axis)
