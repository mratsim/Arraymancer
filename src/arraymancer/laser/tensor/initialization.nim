# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../openmp,
  ../compiler_optim_hints,
  ../strided_iteration/foreach,
  ../dynamic_stack_arrays,
  ../private/nested_containers,
  ./datatypes,
  # Standard library
  typetraits, sequtils,
  # Third-party
  nimblas

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

proc newTensor*[T](shape: varargs[int]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  when T is KnownSupportsCopyMem:
    # seq based tensors are zero'ed by default upon construction
    setZero(result, check_contiguous = false)

proc newTensor*[T](shape: Metadata): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  when T is KnownSupportsCopyMem:
    # seq based tensors are zero'ed by default upon construction
    setZero(result, check_contiguous = false)

proc toTensor*(a: openArray, dummy_bugfix: static[int] = 0): auto =
  ## Convert an openArray to a Tensor
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A Tensor of the same shape
  ##
  ## Note: dummy_bugfix param is unused and is a workaround a Nim bug.
  # TODO: remove 'dummy_bugfix' - https://github.com/nim-lang/Nim/issues/6343

  let shape = getShape(a)
  var data = toSeq(flatIter(a))

  if unlikely(shape.product != data.len):
    raise newException(
      IndexDefect,
      "Each nested sequence at the same level" &
        " must have the same number of elements"
      )

  type T = typeof(data[0])
  var
    t: Tensor[T]
    size: int

  initTensorMetadata(t, size, shape)
  allocCpuStorage(t.storage, size)

  when T is KnownSupportsCopyMem:
    t.copyFromRaw(data[0].unsafeAddr, data.len)
  else:
    when defined(gcArc) or defined(gcOrc):
      t.storage.raw_buffer = move data
    else:
      shallowCopy(t.storage.raw_buffer, data)

  result = t

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
