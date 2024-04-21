# Copyright 2017-2020 Mamy-André Ratsimbazafy
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

import  ./private/p_shapeshifting,
        ./private/p_checks,
        ./private/p_accessors_macros_write,
        ./private/p_empty_tensors,
        ./accessors,
        ./data_structure, ./init_cpu, ./higher_order_applymap
import std / [sequtils, math]

# NOTE: Procs that accepts shape are duplicated to accept both varargs and Metadata
# until either https://github.com/nim-lang/Nim/issues/6528 or https://github.com/nim-lang/Nim/issues/6529
# are solved/added.

proc transpose*(t: Tensor): Tensor {.noinit,noSideEffect,inline.} =
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is not copied or modified, only metadata is modified.
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage

proc asContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noinit.} =
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default tensor will be rowMajor.
  ##
  ## The layout is kept if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout.
  ##
  ## Result is always a fully packed tensor even if the input is a contiguous slice.

  let cCont = t.is_C_contiguous
  let fCont = t.is_F_contiguous

  if (cCont or fCont) and not force:
    return t
  elif cCont and layout == rowMajor:
    return t
  elif fCont and layout == colMajor:
    return t
  contiguousImpl(t, layout, result)

proc reshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noinit.} =
  ## Reshape a tensor. If possible no data copy is done and the returned tensor
  ## shares data with the input. If input is not contiguous, this is not possible
  ## and a copy will be made.
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same. Unlike numpy,
  ##     dimensions cannot be -1 to infer their value. If that is what you need
  ##     you must use the alternative `reshape_infer` proc.
  ## Returns:
  ##   - a tensor with the same data but reshaped.
  reshapeImpl(t, new_shape, result, infer = false)

proc reshape*(t: Tensor, new_shape: Metadata): Tensor {.noinit.} =
  ## Reshape a tensor. If possible no data copy is done and the returned tensor
  ## shares data with the input. If input is not contiguous, this is not possible
  ## and a copy will be made.
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.
  reshapeImpl(t, new_shape, result, infer = false)

proc reshape_infer*(t: Tensor, new_shape: varargs[int]):
    Tensor {.noinit.} =
  ## Reshape a tensor. If possible no data copy is done and the returned tensor
  ## shares data with the input. If input is not contiguous, this is not possible
  ## and a copy will be made.
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same. The new shape can
  ##     contain -1 to infer the size of one (and only one) dimension
  ## Returns:
  ##   - a tensor with the same data but reshaped.
  reshapeImpl(t, new_shape, result, infer = true)

proc flatten*(t: Tensor): Tensor {.noinit,inline.} =
  ## Flatten a tensor, returning a rank-1 tensor with the same data as the input.
  ##
  ## This is the same as `t.reshape([t.size.int])`. Therefore, if possible no
  ## data copy is done and the returned tensor shares data with the input.
  ## If input is not contiguous, this is not possible and a copy will be made.
  ##
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor rank-1 tensor with the same data as the input.
  t.reshape([t.size.int])

proc broadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noinit,noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastImpl(shape)

proc broadcast*[T](t: Tensor[T], shape: Metadata): Tensor[T] {.noinit,noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastImpl(shape)

proc broadcast*[T: SomeNumber](val: T, shape: varargs[int]): Tensor[T] {.noinit.} =
  ## Broadcast a number
  ##
  ## Input:
  ##   - a number to be broadcasted
  ##   - a tensor shape that will be broadcasted to
  ## Returns:
  ##   - a tensor with the broadcasted shape where all elements has the broadcasted value
  ##
  ## The broadcasting is made using tensor data of size 1 and 0 strides, i.e.
  ## the operation is memory efficient.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.
  ##   Modifying any value from this broadcasted tensor will change all its values.
  result.shape.copyFrom(shape)
  result.strides = default(Metadata)
  result.offset = 0
  result.storage.allocCpuStorage(1)
  result.unsafe_raw_buf[0] = val

proc broadcast*[T: SomeNumber](val: T, shape: Metadata): Tensor[T] {.noinit,noSideEffect.} =
  ## Broadcast a number
  ##
  ## Input:
  ##   - a number to be broadcasted
  ##   - a tensor shape that will be broadcasted to
  ## Returns:
  ##   - a tensor with the broadcasted shape where all elements has the broadcasted value
  ##
  ## The broadcasting is made using tensor data of size 1 and 0 strides, i.e.
  ## the operation is memory efficient.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.
  ##   Modifying any value from this broadcasted tensor will change all its values.
  result.shape.copyFrom(shape)
  result.strides = default(Metadata)
  result.offset = 0
  result.storage.allocCpuStorage(1)
  result.unsafe_raw_buf[0] = val

template bc*(t: (Tensor|SomeNumber), shape: varargs[int]): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

template bc*(t: (Tensor|SomeNumber), shape: Metadata): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

proc broadcast2*[T](a, b: Tensor[T]): tuple[a, b: Tensor[T]] {.noSideEffect, noinit.}=
  ## Broadcast 2 tensors so they have compatible shapes for element-wise computations.
  ##
  ## Tensors in the tuple can be accessed with output.a and output.b
  ##
  ## The returned broadcasted Tensors share the underlying data with the input.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##   A broadcasted tensor should not be modified and only used for computation.

  broadcast2Impl(a,b, result)

  result.a.storage = a.storage
  result.b.storage = b.storage


proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noinit,noSideEffect.} =
  ## Permute the dimensions of a tensor into a different order
  ## Input:
  ##   - a tensor
  ##   - the new dimension order
  ## Returns:
  ##   - a tensor with re-ordered dimensions but sharing the same data
  ## See also:
  ##   - moveaxis
  ## Usage:
  ##  .. code:: nim
  ##     # keep dim 0 at position 0 and swap dims 1 and 2
  ##     a.permute(0,2,1)
  ## Notes:
  ##   Call `.clone()` if you want to make a copy of the data, otherwise
  ##   changes to the data of returned tensor will affect the input tensor.

  # TODO: bounds check
  result = t
  permuteImpl(result, dims)

proc moveaxis*(t: Tensor, initial: Natural, target: Natural): Tensor {.noinit.} =
  ## Move one of the axes of a tensor into a new position
  ## Input:
  ##   - a tensor
  ##   - the initial position of the axes to move
  ##   - the target position of the axes to move
  ## Returns:
  ##   - a tensor with moved axes but sharing the same data
  ## See also:
  ##   - permute
  ## Usage:
  ##  .. code:: nim
  ##     # move dim 0 to position 2, which makes
  ##     # dim 1 become dim 0 and dim 2 become dim 1
  ##     a.moveaxis(0, 2)
  ## Notes:
  ##   Call `.clone()` if you want to make a copy of the data, otherwise
  ##   changes to the data of returned tensor will affect the input tensor.
  if initial == target:
    return t
  doAssert initial < t.rank,
    "moveaxis initial axis position (" & $initial & ") exceeds input tensor rank (" & $t.rank & ")"
  doAssert target < t.rank,
    "moveaxis target axis position (" & $target & ") exceeds input tensor rank (" & $t.rank & ")"
  var dims = toSeq(0 ..< t.rank)
  # Note that dims[initial] == initial!
  dims.delete(initial)
  dims.insert(initial, target)
  result = t
  permuteImpl(result, dims)

proc concat*[T](t_list: varargs[Tensor[T]], axis: int): Tensor[T] {.noinit.} =
  ## Concatenate tensors
  ## Input:
  ##   - Tensors
  ##   - An axis (dimension)
  ## Returns:
  ##   - a tensor
  mixin `|`

  var axis_dim = 0
  let t0 = t_list[0]

  for t in t_list:
    when compileOption("boundChecks"):
      check_concat(t0, t, axis)
    axis_dim += t.shape[axis]

  var concat_shape = t0.shape
  concat_shape[axis] = axis_dim

  ## Setup the Tensor
  result = newTensorUninit[T](concat_shape)

  ## Fill in the copy with the matching values
  ### First a sequence of SteppedSlices corresponding to each tensors to concatenate
  var slices = concat_shape.mapIt((0..<it)|1) # TODO avoid allocation
  var iaxis = 0

  ### Now, create "windows" in the result tensor and assign corresponding tensors
  for t in t_list:
    skipIfEmpty(t)
    slices[axis].a = iaxis
    slices[axis].b = iaxis + t.shape[axis] - 1
    result.slicerMut(slices, t)
    iaxis += t.shape[axis]

proc append*[T](t: Tensor[T], values: Tensor[T]): Tensor[T] {.noinit.} =
  ## Create a copy of an rank-1 input tensor with values appended to its end
  ##
  ## Inputs:
  ##   - Rank-1 tensor
  ##   - Rank-1 tensor of extra values to append
  ## Returns:
  ##   - A copy of the input tensor t with the extra values appended at the end.
  ## Notes:
  ##   Append does not occur in-place (a new tensor is allocated and filled).
  ##   To concatenate more than one tensor or tensors that you must use `concat`.
  ##   Compared to numpy's `append`, this proc requires that you explicitly
  ##   flatten the inputs if they are not rank-1 tensors. It also does not
  ##   support the `axis` parameter. If you want to append the values along a
  ##   specific axis, you should use `concat` instead.
  ## Examples:
  ## > echo append([1, 2, 3].toTensor, [4, 5, 6, 7].toTensor)
  ## > # Tensor[system.int] of shape "[9]" on backend "Cpu"
  ## > #    1     2     3     4     5     6     7
  ## >
  ## > echo append([1, 2, 3].toTensor, [[4, 5, 6], [7, 8, 9]].toTensor)
  ## > # Error: unhandled exception: `values.rank == 1` append only works
  ## > # on rank-1 tensors but extra values tensor has rank 2 [AssertionDefect]
  ## >
  ## > echo append([1, 2, 3].toTensor, [[4, 5, 6], [7, 8, 9]].toTensor.flatten)
  ## > #    1     2     3     4     5     6     7     8     9
  doAssert t.rank == 1,
    "`append` only works on rank-1 tensors but first input tensor has rank " &
    $t.rank & " (use `concat` for higher rank tensors)"
  doAssert values.rank == 1,
    "`append` only works on rank-1 tensors but extra values tensor has rank " &
    $values.rank & " (use `concat` for higher rank tensors)"
  let result_size = t.size + values.size
  result = newTensorUninit[T](result_size)
  result[0 ..< t.size] = t
  result[t.size ..< result.size] = values

proc append*[T](t: Tensor[T], values: varargs[T]): Tensor[T] {.noinit.} =
  ## Create a copy of an rank-1 input tensor with one or more values appended to its end
  ##
  ## Inputs:
  ##   - Rank-1 tensor of type T
  ##   - An open array or a list of values of type T
  ## Returns:
  ##   - A copy of the input tensor t with the extra values appended at the end.
  ## Notes:
  ##   Append does not occur in-place (a new tensor is allocated and filled).
  ##   Compared to numpy's `append`, this proc requires that you explicitly
  ##   flatten the input tensor if its rank is greater than 1. It also does not
  ##   support the `axis` parameter. If you want to append values along a
  ##   specific axis, you should use `concat` instead.
  ## Examples:
  ## - Append a single value
  ## > echo append([1, 2, 3].toTensor, 4)
  ## > # Tensor[system.int] of shape "[9]" on backend "Cpu"
  ## > #    1     2     3     4
  ## - Append a multiple values
  ## > echo append([1, 2, 3].toTensor, [4, 5, 6, 7])
  ## > # Tensor[system.int] of shape "[9]" on backend "Cpu"
  ## > #    1     2     3     4     5     6     7
  ## - Append an openArray of values
  ## > echo append([1, 2, 3].toTensor, [4, 5, 6, 7])
  ## > # Tensor[system.int] of shape "[9]" on backend "Cpu"
  ## > #    1     2     3     4     5     6     7
  ## - Only rank-1 tensors are supported
  ## > echo append([[1, 2, 3], [4, 5, 6]].toTensor, [7, 8, 9])
  ## > # Error: unhandled exception: `t.rank == 1` append only works
  ## > # on rank-1 tensors but first input tensor has rank 2 [AssertionDefect]
  ## - Flatten higher ranked tensors before appending
  ## > echo append([[1, 2, 3], [4, 5, 6]].toTensor.flatten, [7, 8, 9])
  ## > #    1     2     3     4     5     6     7     8     9
  doAssert t.rank == 1,
    "`append` only works on rank-1 tensors but input tensor has rank " &
    $t.rank & " (use `concat` for higher rank tensors)"
  let result_size = t.size + values.len
  result = newTensorUninit[T](result_size)
  if t.size > 0:
    result[0 ..< t.size] = t
  result[t.size ..< result.size] = values

func squeeze*(t: AnyTensor): AnyTensor {.noinit.}=
  ## Squeeze tensors. For example a Tensor of shape [4,1,3] will become [4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  result = t
  result.squeezeImpl

func squeeze*(t: Tensor, axis: Natural): Tensor {.noinit.}=
  ## Collapse the given axis, if the dimension is not 1, it does nothing.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that axis collapsed, if it was a singleton dimension
  result = t
  result.squeezeImpl(axis)

func unsqueeze*(t: Tensor, axis: Natural): Tensor {.noinit.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that new axis
  result = t
  result.unsqueezeImpl(axis)

proc stack*[T](tensors: varargs[Tensor[T]], axis: Natural = 0): Tensor[T] {.noinit.} =
  ## Join a sequence of tensors along a new axis into a new tensor.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a new stacked tensor along the new axis
  func stack_unsqueeze(t: Tensor[T]): Tensor[T] = t.unsqueeze(axis)
  tensors.map(stack_unsqueeze).concat(axis)

func split*[T](t: Tensor[T], chunk_size: Positive, axis: Natural): seq[Tensor[T]] {.noinit.} =
  ## Split the tensor into chunks of size ``chunk_size`` along the specified axis.
  ## Last chunk size will equal the remainder if the specified axis length is not divisible
  ## by ``chunk_size``

  doAssert t.shape[axis] >= chunk_size
  let nb_chunks = t.shape[axis] div chunk_size
  let rem_size  = t.shape[axis] mod chunk_size

  result = newSeq[Tensor[T]](nb_chunks + int(rem_size != 0))
  for i in 0 ..< nb_chunks:
    result[i] = t.atAxisIndex(axis, i * chunk_size, chunk_size)

  if rem_size != 0:
    result[^1] = t.atAxisIndex(axis, nb_chunks * chunk_size, rem_size)

func chunk*[T](t: Tensor[T], nb_chunks: Positive, axis: Natural): seq[Tensor[T]] {.noinit.} =
  ## Splits a Tensor into n chunks along the specified axis.
  ##
  ## In case a tensor cannot be split evenly,
  ## with la == length_axis, n = n_chunks
  ## it returns la mod n subtensors of size `(la div n) + 1`
  ##            the rest of size `la div n`.
  ##
  ## This is consistent with numpy array_split

  doAssert t.shape[axis] >= nb_chunks
  let chunk_size = t.shape[axis] div nb_chunks
  let remainder  = t.shape[axis] mod nb_chunks

  result = newSeq[Tensor[T]](nb_chunks)
  for i in 0 ..< nb_chunks:
    if i < remainder:
      result[i] = t.atAxisIndex(axis, i * chunk_size + i, chunk_size + 1)
    else:
      result[i] = t.atAxisIndex(axis, i * chunk_size + remainder, chunk_size)

proc rollRank1Tensor[T](t: Tensor[T], shift: int): Tensor[T] {.noinit,inline.} =
  let shift = floorMod(shift, t.size)
  if shift == 0:
    return t
  let shiftedLen = t.size - shift
  result = newTensorUninit[T](t.shape)
  result[shift ..< t.size] = t[0 ..< shiftedLen]
  result[0 ..< shift] = t[shiftedLen ..< t.size]

proc roll*[T](t: Tensor[T], shift: int): Tensor[T] {.noinit.} =
  ## Roll elements of tensor "globally" (i.e. across all axes).
  ##
  ## This takes a tensor, flattens it, rolls the elements `shift` positions
  ## (taking the last `shift` elements of the flattened tensor and putting
  ## them at the beginning of the flattened tensor), and then reshapes the
  ## rolled tensor back to the original shape.
  ##
  ## This is different from the version of this proc that accepts an axis,
  ## which rolls _slices_ of a tensor taken along the selected axis.
  ##
  ## Input:
  ##   - t: Input tensor.
  ##   - shift: Integer number of places by which elements are shifted.
  ## Return:
  ##   - Output tensor, with the same shape as `a`.
  ##
  ## Examples:
  ## > let x = arange(5)
  ## > echo x.roll(2)
  ## > Tensor[system.int] of shape "[5]" on backend "Cpu"
  ## >     3     4     0     1     2
  ## > echo x.roll(-2)
  ## > Tensor[system.int] of shape "[5]" on backend "Cpu"
  ## >     2     3     4     0     1
  ## > let x2 = arange(5).reshape(2, 5)
  ## > echo x2
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |0      1     2     3     4|
  ## > # |5      6     7     8     9|
  ## > echo roll(x2, 1)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |9      0     1     2     3|
  ## > # |4      5     6     7     8|
  ## > echo roll(x2, -1)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |1      2     3     4     5|
  ## > # |6      7     8     9     0|
  rollRank1Tensor(t.flatten, shift).reshape(t.shape)

proc roll*[T](t: Tensor[T], shift: int, axis: Natural): Tensor[T] {.noinit.} =
  ## Roll slices of a tensor along a given axis.
  ##
  ## Slices that roll beyond the last position are re-introduced at
  ## the first.
  ##
  ## Note that calling this proc with a rank-1 tensor, will simply check that
  ## `axis` == 0 and then call the (axis-less) version of this proc.
  ##
  ## Input:
  ##   - t : Input tensor.
  ##   - shift : Integer number of places by which elements are shifted.
  ##   - axis : an axis (dimension).
  ## Return:
  ##   - Output tensor, with the same shape as `t`.
  ##
  ## Notes:
  ##  - numpy's `roll` also supports passing a list of shifts and axis, while
  ##    this proc doesn't. However, you can achieve the same effect by calling
  ##    roll multiple times in a row (i.e. `np.roll(t, [1, 2], axis=[0, 1])` is
  ##    equivalent to `t.roll(1, axis=0).roll(2, axis=1)` which is arguably
  ##    more clear).
  ##
  ## Examples:
  ## > let x = arange(5)
  ## > echo x.roll(2, axis=0)
  ## > Tensor[system.int] of shape "[5]" on backend "Cpu"
  ## >     3     4     0     1     2
  ## > echo x.roll(-2, axis=0)
  ## > Tensor[system.int] of shape "[5]" on backend "Cpu"
  ## >     2     3     4     0     1
  ##
  ## > let x2 = arange(5).reshape(2, 5)
  ## > echo x2
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |0      1     2     3     4|
  ## > # |5      6     7     8     9|
  ## > echo roll(x2, 1, axis=0)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |5      6     7     8     9|
  ## > # |0      1     2     3     4|
  ## > echo roll(x2, -1, axis=0)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |5      6     7     8     9|
  ## > # |0      1     2     3     4|
  ## > echo roll(x2, 1, axis=1)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |4      0     1     2     3|
  ## > # |9      5     6     7     8|
  ## > echo roll(x2, -1, axis=1)
  ## > # Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > # |1      2     3     4     0|
  ## > # |6      7     8     9     5|
  ## > echo x2.roll(1, axis=0).roll(2, axis=1)
  ## > Tensor[system.int] of shape "[2, 5]" on backend "Cpu"
  ## > |8      9     5     6     7|
  ## > |3      4     0     1     2|
  if shift == 0 or t.rank == 0:
    result = t
  elif t.rank == 1:
    doAssert axis == 0, "roll axis must be 0 for rank-1 tensors"
    result = rollRank1Tensor(t, shift)
  else:
    doAssert axis < t.rank,
      "roll axis (" & $axis & ") exceeds input tensor rank (" & $t.rank & ")"
    var rolled_slices = newSeq[Tensor[T]](t.shape[axis])
    let shift = floorMod(shift, t.shape[axis])
    if shift == 0:
      return t
    for n, t_slice in enumerateAxis(t, axis = axis):
      let result_idx = floorMod(n + shift, t.shape[axis])
      rolled_slices[result_idx] = t_slice
    result = concat(rolled_slices, axis)

proc repeat_values*[T](t: Tensor[T], reps: int, axis = -1): Tensor[T] {.noinit.} =
  ## Create a new tensor with each value repeated (the same amount of)`reps` times
  ##
  ## Inputs:
  ##   - t: A tensor.
  ##   - reps: The integer number of times that each value must be repeated.
  ##   - axis: The axis over which values will be repeated. Defaults to the
  ##           last axis.
  ##
  ## Returns:
  ##   - A new tensor containing the values of the input tensor repeated `reps`
  ##     times over the selected axis.
  ##
  ## Notes:
  ##   - All values are repeated (the same amount of) `reps` times along the
  ##     selected axis. This makes the output shape the same as the input shape
  ##     except at the selected axis, which is `reps` times greater.
  ##   - There are an alternative versions of this function which take a list
  ##     of `reps` instead of a single `reps` value.
  ##   - The equivalent numpy function is called `repeat`, while the
  ##     equivalent Matlab function is called `repelem`. Different names
  ##     where chosen here to avoid confusion with nim's `repeat` function
  ##     which behaves like numpy's `tile`, not like this function.
  ##
  ## Examples:
  ## ```nim
  ## let t = arange(6).reshape(2, 3)
  ## echo t.repeat_values(2)
  ## # Tensor[system.int] of shape "[3, 8]" on backend "Cpu"
  ## # |0      0     1     1     2     2     3     3|
  ## # |4      4     5     5     6     6     7     7|
  ## # |8      8     9     9    10    10    11    11|
  ##
  ## echo t.repeat_values(2, axis = 0)
  ## # Tensor[system.int] of shape "[6, 4]" on backend "Cpu"
  ## # |0      1     2     3|
  ## # |0      1     2     3|
  ## # |4      5     6     7|
  ## # |4      5     6     7|
  ## # |8      9    10    11|
  ## # |8      9    10    11|
  ## ```
  let axis = if axis >= 0: axis else: t.shape.len + axis

  when compileOption("boundChecks"):
    doAssert axis < t.rank,
      "repeat_values called with an axis (" & $axis &
      ") that exceeds the input tensor rank (" & $t.rank & ")"

  var target_shape = t.shape
  target_shape[axis] *= reps

  result = newTensorUninit[T](t.size * reps)
  var step = 1
  for idx in countdown(t.shape.high, axis + 1):
    step *= t.shape[idx]
  for (idx, it) in t.enumerate():
    for n in countup(0, reps - 1):
      let base = (step * reps) * (idx div step) + idx mod step
      result[base + n * step] = it
  return result.reshape(target_shape)

proc repeat_values*[T](t: Tensor[T], reps: openArray[int]): Tensor[T] {.noinit.} =
  ## Create a new rank-1 tensor with each value `t[i]` repeated `reps[i]` times
  ##
  ## Compared to the version of `repeat_values` that takes a single integer
  ## `reps` value this version always returns a rank-1 tensor (regardless of  and does not take
  ## the input shape) and does not take an axis argument.
  ##
  ## Inputs:
  ##   - t: A tensor.
  ##   - reps: A sequence or array of integers indicating the number of times
  ##           that each value must be repeated. It must have as many values
  ##           as the input tensor.
  ##
  ## Returns:
  ##   - A new rank-1 tensor containing the values of the input tensor repeated
  ##     `reps` times.
  ##
  ## Notes:
  ##   - If a rep value is 0, the corresponding item in the input tensor will
  ##     be skipped from the output.
  ##   - The equivalent numpy function is called `repeat`, while the
  ##     equivalent Matlab function is called `repelem`. Different names
  ##     where chosen here to avoid confusion with nim's `repeat` function
  ##     which behaves like numpy's `tile`, not like this function.
  ##
  ## Example:
  ## ```nim
  ## let t = [3, 5, 2, 4].toTensor
  ## echo t.repeat_values([1, 0, 3, 2])
  ## # Tensor[system.int] of shape "[6]" on backend "Cpu"
  ## #     3     2     2     2     4     4
  ## ```
  when compileOption("boundChecks"):
    doAssert reps.len == t.len,
      "repeat_values called with a reps list whose length (" & $reps.len &
      ") does not match the input tensor size (" & $t.len & ")"

  result = newTensorUninit[T](sum(reps))
  var base_pos = 0
  for (idx, it) in t.enumerate():
    for n in countup(0, reps[idx] - 1):
      result[base_pos + n] = it
    base_pos += reps[idx]
  return result

proc repeat_values*[T](t: Tensor[T], reps: Tensor[int]): Tensor[T] {.noinit, inline.} =
  ## Create a new rank-1 tensor with each value `t[i]` repeated `reps[i]` times
  ##
  ## Overload of this function which takes a `Tensor[int]` instead of an
  ## `openArray[int]`. Behavior is exactly the same as the `openArray[int]`
  ## version.
  ## ```
  t.repeat_values(reps.toSeq1D)

proc tile*[T](t: Tensor[T], reps: varargs[int]): Tensor[T] =
  ## Construct a new tensor by repeating the input tensor a number of times on one or more axes
  ##
  ## Inputs:
  ##   - t: The tensor to repeat
  ##   - reps: One or more integers indicating the number of times to repeat
  ##           the tensor on each axis (starting with axis 0)
  ##
  ## Result:
  ##   - A new tensor whose shape is `t.shape *. reps`
  ##
  ## Notes:
  ##   - If a rep value is 1, the tensor is not repeated on that particular axis
  ##   - If there are more rep values than the input tensor has axes, additional
  ##     dimensions are prepended to the input tensor as needed. Note that this
  ##     is similar to numpy's `tile` function behavior, but different to
  ##     Matlab's `repmat` behavior, which appends missing dimensions instead
  ##     of prepending them.
  ##   - This function behavior is similar to nims `sequtils.repeat`, in that
  ##     it repeats the full tensor multiple times. If what you want is to
  ##     repeat the _elements_ of the tensor multiple times, rather than the
  ##     full tensor, use the `repeat_values` procedure instead.
  ##
  ## Examples:
  ## ```nim
  ## let x = arange(4).reshape(2, 2)
  ##
  ## # When the number of reps and tensor dimensions match, the ouptut tensor
  ## # shape is the `reps *. t.shape`
  ## echo tile(x, 2, 3)
  ## > Tensor[system.int] of shape "[4, 6]" on backend "Cpu"
  ## > |0      1     0     1     0     1|
  ## > |2      3     2     3     2     3|
  ## > |0      1     0     1     0     1|
  ## > |2      3     2     3     2     3|
  ##
  ## # If there are fewer reps than tensor dimensions, start
  ## # repeating on the first axis (leaving alone axis with missing reps)
  ## echo tile(x, 2)
  ## > Tensor[system.int] of shape "[4, 2]" on backend "Cpu"
  ## > |0      1|
  ## > |2      3|
  ## > |0      1|
  ## > |2      3|
  ##
  ## # If there are more reps than tensor dimensions, prepend the missing
  ## # dimensions before repeating
  ## echo tile(x, 1, 2, 3)
  ## > Tensor[system.int] of shape "[1, 4, 6]" on backend "Cpu"
  ## >                 0
  ## > |0      1     0     1     0     1|
  ## > |2      3     2     3     2     3|
  ## > |0      1     0     1     0     1|
  ## > |2      3     2     3     2     3|
  ## ```
  result = t
  for ax in countdown(reps.high, 0):
    var concat_seq = repeat(result, reps[ax])
    if ax >= result.shape.len:
      # mutate the repeated tensors to have one more axis
      concat_seq.applyIt(unsqueeze(it, 0))
    result = concat(concat_seq, axis=ax)

