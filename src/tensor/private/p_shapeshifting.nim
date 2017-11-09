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

import  ../backend/metadataArray,
        ../data_structure, ../init_cpu, ../higher_order,
        ./p_checks,
        nimblas

template contiguousT*[T](result, t: Tensor[T], layout: OrderType): untyped=
  if layout == rowMajor:
    result = t.map_inline(x)
  else:
    let t_transposed = t.unsafeTranspose()
    result = t_transposed.map_inline(x)

proc reshape_with_copy*[T](t: Tensor[T], new_shape: varargs[int]|MetadataArray): Tensor[T] {.noInit,inline.}=
  # Can't call "tensorCpu" template here for some reason
  result = newTensorUninit[T](new_shape)
  result.apply2_inline(t,y)

template reshape_no_copy*(t: AnyTensor, new_shape: varargs[int]|MetadataArray): untyped =
  when compileOption("boundChecks"):
    check_nocopyReshape t
    when not (new_shape is MetadataArray):
      check_reshape(t, new_shape.toMetadataArray)
    else:
      check_reshape(t, new_shape)
  result.shape.copyFrom(new_shape)
  shape_to_strides(result.shape, rowMajor, result.strides)
  result.offset = t.offset

template broadcastT*(t: var AnyTensor, shape: varargs[int]|MetadataArray) =
  when compileOption("boundChecks"):
    assert t.rank == shape.len

  for i in 0..<t.rank:
    if t.shape[i] == 1:
      if shape[i] != 1:
        t.shape[i] = shape[i]
        t.strides[i] = 0
    elif t.shape[i] != shape[i]:
      raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")

template broadcast2T*[T](a, b: AnyTensor[T], result: var tuple[a, b: AnyTensor[T]]) =
  let rank = max(a.rank, b.rank)

  var shapeA, stridesA, shapeB, stridesB = initMetadataArray(rank) # initialized with 0

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
      raise newException(ValueError, "Broadcasting error: non-singleton dimensions must be the same in both tensors")

  result.a.shape = shapeA
  result.a.strides = stridesA
  result.a.offset = a.offset

  result.b.shape = shapeB
  result.b.strides = stridesB
  result.b.offset = b.offset


proc exch_dim*(t: Tensor, dim1, dim2: int): Tensor {.noInit,noSideEffect.}=
  if dim1 == dim2:
    return

  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

template permuteT*(result: var AnyTensor, dims: varargs[int]): untyped =
  var perm = dims.toMetadataArray
  for i, p in perm:
    if p != i and p != -1:
      var j = i
      while true:
        result = result.exch_dim(j, perm[j])
        (perm[j], j) = (-1, perm[j])
        if perm[j] == i:
          break
      perm[j] = -1


template squeezeT*(t: var AnyTensor): untyped =
  var idx_real_dim = 0

  for i in 0..<t.rank:
    if t.shape[i] != 1:
      if i != idx_real_dim:
        t.shape[idx_real_dim] = t.shape[i]
        t.strides[idx_real_dim] = t.strides[i]
      inc idx_real_dim

  t.shape = t.shape[0..<idx_real_dim]
  t.strides = t.strides[0..<idx_real_dim]

template squeezeT*(t: var AnyTensor, axis: int): untyped =
  when compileOption("boundChecks"):
    check_squeezeAxis(t, axis)

  if t.rank > 1 and t.shape[axis] == 1: # We don't support rank 0 Tensor
    t.shape.delete(axis)
    t.strides.delete(axis)

template unsqueezeT*(t: var AnyTensor, axis: int): untyped =
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
