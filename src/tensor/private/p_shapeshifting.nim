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
    result = t.mapT(x)
  else:
    let t_transposed = t.unsafeTranspose()
    result = t_transposed.mapT(x)

proc reshape_with_copy*[T](t: Tensor[T], new_shape: MetadataArray): Tensor[T] {.inline.}=
  # Can't call "tensorCpu" template here for some reason
  result = newTensorUninit[T](new_shape)
  result.apply2T(t,y)

template reshape_no_copy*(t: AnyTensor, new_shape: varargs[int]): untyped =
  let ns = new_shape.toMetadataArray
  when compileOption("boundChecks"):
    check_nocopyReshape t
    check_reshape(t, ns)

  var matched_dims = 0
  for shapes in zip(t.shape, ns): # This relies on zip stopping early
    if shapes[0] != shapes[1]:
      break
    inc matched_dims

  result.shape = ns

  # Strides extended for unmatched dimension
  let ext_strides = result.shape[matched_dims..result.shape.high].shape_to_strides
  result.strides = t.strides[0..<matched_dims] & ext_strides
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

proc exch_dim*(t: Tensor, dim1, dim2: int): Tensor {.noSideEffect.}=
  if dim1 == dim2:
    return

  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

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
