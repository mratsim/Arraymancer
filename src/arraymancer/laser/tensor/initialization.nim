# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import std / complex

import
  ../openmp,
  ../compiler_optim_hints,
  ../strided_iteration/foreach,
  ../dynamic_stack_arrays,
  ../private/nested_containers,
  ./datatypes
# Standard library
import std / [typetraits, sequtils, sets]

# The following export is needed to avoid a compilation error in
# algorithms.nim/intersection() when running the test_algorithms test:
# `Error: type mismatch - Expression: items(s1)`
# (Alternative: could use `bind sets.items` in `intersection` and `setDiff`)
export sets

# Third-party
import nimblas

when (NimMajor, NimMinor) < (1, 4):
  import ../../std_version_types

## Initialization and copy routines

func toMetadata*(s: varargs[int]): Metadata =
  result.len = s.len
  for i in 0 ..< s.len:
    result.data[i] = s[i]

template toMetadata*(m: Metadata): Metadata = m

template initTensorMetadataImpl(
    result: var Tensor,
    size: var int, shape: openArray[int]|Metadata,
    layout: static OrderType) =
  ## We don't use a proc directly due to https://github.com/nim-lang/Nim/issues/6529
  result.shape = shape.toMetadata
  mixin rank
  result.strides.len = result.rank

  size = 1
  when layout == rowMajor:
    for i in countdown(shape.len - 1, 0):
      result.strides[i] = size
      size *= shape[i]
  elif layout == colMajor:
    for i in 0 ..< shape.len:
      result.strides[i] = size
      size *= shape[i]
  else:
    {.error: "Unreachable, unknown layout".}

func initTensorMetadata*(
       result: var Tensor,
       size: var int, shape: openArray[int],
       layout: static OrderType = rowMajor) =
  ## result metadata and size will be initialized in-place
  initTensorMetadataImpl(result, size, shape, layout)

func initTensorMetadata*(
       result: var Tensor,
       size: var int, shape: Metadata,
       layout: static OrderType = rowMajor) =
  ## result metadata and size will be initialized in-place
  initTensorMetadataImpl(result, size, shape, layout)

proc deepCopy*[T](dst: var Tensor[T], src: Tensor[T]) =
  ## Performs a deep copy of y and copies it into x.
  ## Deepcopy is recursive including for ref types and custom types
  ## that implement deepCopy.
  ##
  ## Note that if x was already initialized with a ``storage``,
  ## the storage will be detached from x. This does not write
  ## into existing storage.
  var size: int
  initTensorMetadata(dst, size, src.shape)
  allocCpuStorage(dst.storage, size)

  when T is KnownSupportsCopyMem:
    # We use memcpy, due to SIMD optimizations in memcpy,
    # we require higher parallelization thresholds
    if src.is_C_contiguous:
      omp_parallel_chunks(
            size, chunk_offset, chunk_size,
            OMP_MEMORY_BOUND_GRAIN_SIZE * 4):
        copyMem(
          dst.unsafe_raw_offset[chunk_offset].addr,
          src.unsafe_raw_offset[chunk_offset].unsafeAddr,
          chunk_size * sizeof(T)
          )
    else:
      forEachStrided d in dst, s in src:
        d = s
  else:
    # If the type doesn't supports memcpy,
    # we assume we can't use OpenMP and we need
    # recursive deepCopy
    forEachSerial d in dst, s in src:
      deepCopy(d, s) # recursive deepcopy

proc copyFrom*[T](dst: var Tensor[T], src: Tensor[T]) =
  ## Copy the source tensor into the destination tensor.
  ## Both should have the same shape. If destination tensor is a view
  ## only the data exposed by the view is modified.
  ##
  ## This is useful to update subslices of an existing tensor.
  ##
  ## ⚠️ Warning:
  ##    The data exposed by the destination tensor will be overwritten.
  ##    If destination tensor is a view, all views of that data will be changed.
  ##    They however conserve their shape and strides.
  ##
  ## Note: The copy is not recursive.
  when T is KnownSupportsCopyMem:
    # We use memcpy, due to SIMD optimizations in memcpy,
    # we require higher parallelization thresholds
    if src.is_C_contiguous:
      assert dst.shape == src.shape
      omp_parallel_chunks(
            src.size, chunk_offset, chunk_size,
            OMP_MEMORY_BOUND_GRAIN_SIZE * 4):
        copyMem(
          dst.unsafe_raw_offset[chunk_offset].addr,
          src.unsafe_raw_offset[chunk_offset].unsafeAddr,
          chunk_size * sizeof(T)
        )
    else:
      forEachStrided d in dst, s in src:
        d = s
  else:
    # If the type doesn't supports memcpy,
    # we assume we can't use OpenMP
    forEachSerial d in dst, s in src:
      d = s # non-recursive copy

proc copyFromRaw*[T](dst: var Tensor[T], buffer: ptr T, len: Natural) =
  ## Copy data from the buffer into the destination tensor.
  ## Destination tensor size and buffer length should be the same
  when T is KnownSupportsCopyMem:
    withCompilerOptimHints()
    mixin size
      # either this or `from ../../tensor/data_structure import size` is needed
      # as can be seen with: `type tmp = typeof(Tensor[int].default.size)` which
      # would fail at top-level
    doAssert dst.size == len, "Tensor size and buffer length should be the same"
    let buf{.restrict.} = cast[ptr UncheckedArray[T]](buffer)
    omp_parallel_chunks(
            len, chunk_offset, chunk_size,
            OMP_MEMORY_BOUND_GRAIN_SIZE * 4):
      copyMem(
        dst.unsafe_raw_offset[chunk_offset].addr,
        buf[chunk_offset].unsafeAddr,
        chunk_size * sizeof(T)
      )
  else:
    {.fatal: "Only non-ref types and types with trivial destructors can be raw copied.".}

proc setZero*[T](t: var Tensor[T], check_contiguous: static bool = true) =
  ## Reset/initialize the tensor data to binary zero.
  ## The tensor metadata is not touched.
  ## Input tensor must be contiguous. For seq based Tensors the underlying
  ## sequence will be reset and set back to the tensors size.
  ##
  ## ⚠️ Warning:
  ##    The data of the input tensor will be overwritten.
  ##    If destination tensor is a view, all views of that data will be changed.
  ##    They however conserve their shape and strides.
  when check_contiguous:
    if unlikely(not t.is_C_contiguous):
      # TODO: error model - https://github.com/numforge/laser/issues/2
      # + If using exceptions, display the tensor ident with astToStr
      raise newException(ValueError, "Input tensor is not contiguous.")

  when not (T is KnownSupportsCopyMem):
    t.storage.raw_buffer.reset()
    t.storage.raw_buffer.setLen(t.size)
  else:
    mixin size
    omp_parallel_chunks(
          t.size, chunk_offset, chunk_size,
          OMP_MEMORY_BOUND_GRAIN_SIZE * 4):
      zeroMem(
        t.unsafe_raw_offset[chunk_offset].addr,
        chunk_size * sizeof(T)
      )

proc newTensor*[T](shape: varargs[int] = [0]): Tensor[T] =
  ## Create a new tensor of type T with the given shape.
  ##
  ## If no shape is provided, we create an empty rank-1 tensor.
  ## To create a rank-0 tensor, explicitly pass and empty shape `[]`.
  ##
  ## Note that in general it is not a good idea to use rank-0 tensors.
  ## However, they can be used as "sentinel" values for Tensor arguments.
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  when T is KnownSupportsCopyMem:
    # seq based tensors are zero'ed by default upon construction
    setZero(result, check_contiguous = false)

proc newTensor*[T](shape: Metadata): Tensor[T] =
  ## Create a new tensor of type T with the given shape.
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  when T is KnownSupportsCopyMem:
    # seq based tensors are zero'ed by default upon construction
    setZero(result, check_contiguous = false)

proc toTensor[T](a: openArray[T], shape: Metadata): Tensor[T] =
  ## Convert an openArray to a Tensor
  ##
  ## Input:
  ##      - An array or a seq, must be flattened. Called by `toTensor` below.
  ## Result:
  ##      - A Tensor of the same shape
  var data = @a
  if unlikely(shape.product != data.len):
    raise newException(
      IndexDefect,
      "Each nested sequence at the same level" &
        " must have the same number of elements"
      )
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  when T is KnownSupportsCopyMem:
    result.copyFromRaw(data[0].unsafeAddr, data.len)
  else:
    when defined(gcArc) or defined(gcOrc):
      result.storage.raw_buffer = move data
    else:
      shallowCopy(result.storage.raw_buffer, data)

proc toTensor*[T](a: openArray[T]): auto =
  ## Convert an openArray into a Tensor
  ##
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A Tensor of the same shape
  ##
  # Note: we removed the dummy static bugfix related to Nim issue
  # https://github.com/nim-lang/Nim/issues/6343
  # motivated by
  # https://github.com/nim-lang/Nim/issues/20993
  # due to the previous local type alias causing issues.
  let shape = getShape(a)
  let data = toSeq(flatIter(a))
  result = toTensor(data, shape)

proc toTensor*[T; U](a: openArray[T], typ: typedesc[U]): Tensor[U] {.inline.} =
  ## Convert an openArray into a Tensor of type `typ`
  ##
  ## This is a convenience function which given an input `a` is equivalent to
  ## calling `a.toTensor().asType(typ)`. If `typ` is the same type of the
  ## elements of `a` then it is the same as `a.toTensor()` (i.e. there is no
  ## overhead).
  ##
  ## Inputs:
  ##      - An array or a seq (can be nested)
  ##      - The target type of the result Tensor
  ## Result:
  ##      - A Tensor of the selected type and the same shape as the input
  when T is U:
    toTensor(a)
  else:
    toTensor(a).asType(typ)

proc toTensor*[T](a: HashSet[T] | OrderedSet[T]): Tensor[T] =
  ## Convert a HashSet or an OrderedSet into a Tensor
  ##
  ## Input:
  ##      - An HashSet or an OrderedSet
  ## Result:
  ##      - A Tensor of the same shape
  var shape = MetaData()
  let data = toSeq(a)
  shape.add(data.len)
  result = toTensor(data, shape)

proc toTensor*[T; U](a: HashSet[T] | OrderedSet[T], typ: typedesc[U]): Tensor[U] {.inline.} =
  ## Convert a HashSet or an OrderedSet into a Tensor of type `typ`
  ##
  ## This is a convenience function which given an input `a` is equivalent to
  ## calling `a.toTensor().asType(typ)`. If `typ` is the same type of the
  ## elements of `a` then it is the same as `a.toTensor()` (i.e. there is no
  ## overhead).
  ##
  ## Inputs:
  ##      - An HashSet or an OrderedSet
  ##      - The target type of the result Tensor
  ## Result:
  ##      - A Tensor of the selected type
  when T is U:
    toTensor(a)
  else:
    toTensor(a).asType(typ)

proc fromBuffer*[T](rawBuffer: ptr UncheckedArray[T], shape: varargs[int], layout: static OrderType): Tensor[T] =
  ## Creates a `Tensor[T]` from a raw buffer, cast as `ptr UncheckedArray[T]`. The
  ## size derived from the given shape must match the size of the buffer!
  ##
  ## If you type cast a raw `pointer` to `ptr UncheckedArray[T]` before handing it to this
  ## proc, make sure to cast to the correct type as we cannot check the validity of
  ## the type!
  ##
  ## Its counterpart ``toUnsafeView`` can be used to obtain ``ptr UncheckedArray`` from a Tensor.
  var size: int
  initTensorMetadata(result, size, shape, layout)
  cpuStorageFromBuffer(result.storage, rawBuffer, size)

proc fromBuffer*[T](rawBuffer: ptr UncheckedArray[T], shape: varargs[int]): Tensor[T] =
  ## Call `fromBuffer` with layout = rowMajor
  fromBuffer[T](rawBuffer, shape, rowMajor)

proc fromBuffer*[T](rawBuffer: pointer, shape: varargs[int], layout: static OrderType): Tensor[T] =
  ## Creates a `Tensor[T]` from a raw `pointer`. Make sure that the explicit type
  ## given to this proc actually matches the data stored behind the pointer!
  ## The size derived from the given shape must match the size of the buffer!
  ##
  ## Its counterpart ``toUnsafeView`` can be used to obtain ``ptr UncheckedArray`` from a Tensor.
  var size: int
  initTensorMetadata(result, size, shape, layout)
  cpuStorageFromBuffer(result.storage, rawBuffer, size)

proc fromBuffer*[T](rawBuffer: pointer, shape: varargs[int]): Tensor[T] =
  ## Call `fromBuffer` with layout = rowMajor
  fromBuffer[T](rawBuffer, shape, rowMajor)

func toUnsafeView*[T: KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): ptr UncheckedArray[T] {.inline.} =
  ## Returns an unsafe view of the valid data as a ``ptr UncheckedArray``.
  ## Its counterpart ``fromBuffer`` can be used to create a Tensor from``ptr UncheckedArray``.
  ##
  ## Unsafe: the pointer can outlive the input tensor.
  unsafe_raw_offset(t, aligned).distinctBase()

proc toHashSet*[T](t: Tensor[T]): HashSet[T] =
  ## Convert a Tensor into a `HashSet`
  ##
  ## Note that this is a lossy operation, since a HashSet only stores an
  ## unsorted set of unique elements.
  result = initHashSet[T](t.size)
  for x in t:
    result.incl x

func item*[T_IN, T_OUT](t: Tensor[T_IN], _: typedesc[T_OUT]): T_OUT =
  ## Returns the value of the input Tensor as a scalar of the selected type.
  ## This only works for Tensors (of any rank) that contain one single element.
  ## If the tensor has more than one element IndexDefect is raised.
  if likely(t.size == 1):
    when T_IN is Complex and T_OUT is Complex:
      # When the input and the output types are Complex, we need to find
      # the "base" type of the output type (e.g. float32 or float64),
      # and then convert the real and imaginary parts of the input value
      # into the output base type before creating the output complex type
      type TT = typeof(
        block:
          var tmp: T_OUT
          tmp.re
      )
      let val = t.squeeze[0]
      result = complex(TT(val.re), TT(val.im))
    elif T_OUT is Complex64:
      result = complex(float64(t.squeeze[0]))
    elif T_OUT is Complex32:
      result = complex(float32(t.squeeze[0]))
    else:
      result = T_OUT(t.squeeze[0])
  elif t.size > 1:
    raise newException(IndexDefect, "You cannot convert a Tensor that has more than 1 element into a scalar")
  else:
    raise newException(IndexDefect, "You cannot convert an empty Tensor into a scalar")

func item*[T](t: Tensor[T]): T {.inline.}=
  ## Returns the value of the input Tensor as a scalar (without changing its type).
  ## This only works for Tensors (of any rank) that contain one single element.
  ## If the tensor has more than one element IndexDefect is raised.
  item(t, T)
