# Laser & Arraymancer
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Types and low level primitives for tensors

import
  ../dynamic_stack_arrays, ../compiler_optim_hints, ../private/memory,
  typetraits, complex

# const MAXRANK = 7

when NimVersion < "1.1.0":
  # For distinctBase
  import sugar

when not defined(nimHasCursor):
  {.pragma: cursor.}

type
  KnownSupportsCopyMem* = SomeNumber | char | Complex[float64] | Complex[float32] | bool

  RawImmutableView*[T] = distinct ptr UncheckedArray[T]
  RawMutableView*[T] = distinct ptr UncheckedArray[T]

  Metadata* = DynamicStackArray[int]

  # On CPU, the tensor datastructures and basic accessors
  # are defined in laser/tensor/datatypes
  MetadataArray* {.deprecated: "Use Metadata instead".} = Metadata

  Tensor*[T] = object                    # Total stack: 128 bytes = 2 cache-lines
    shape*: Metadata                     # 56 bytes
    strides*: Metadata                   # 56 bytes
    offset*: int                         # 8 bytes
    storage* {.cursor.}: CpuStorage[T]   # 8 bytes

  CpuStorage*[T] {.shallow.} = ref CpuStorageObj[T] # Total heap: 25 bytes = 1 cache-line
  CpuStorageObj[T] {.shallow.} = object
    # Workaround supportsCopyMem in type section - https://github.com/nim-lang/Nim/issues/13193
    when T is KnownSupportsCopyMem:
      raw_buffer*: ptr UncheckedArray[T] # 8 bytes
      memalloc*: pointer                 # 8 bytes
      isMemOwner*: bool                  # 1 byte
    else: # Tensors of strings, other ref types or non-trivial destructors
      raw_buffer*: seq[T]                # 8 bytes (16 for seq v2 backed by destructors?)

proc initMetadataArray*(len: int): MetadataArray {.inline.} =
  result.len = len

proc toMetadataArray*(s: varargs[int]): MetadataArray {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

func rank*[T](t: Tensor[T]): range[0 .. LASER_MAXRANK] {.inline.} =
  t.shape.len

func size*[T](t: Tensor[T]): Natural {.inline.} =
  t.shape.product

# note: the finalizer has to be here for ARC to like it
when not defined(gcDestructors):
  proc finalizer[T](storage: CpuStorage[T]) =
    static: assert T is KnownSupportsCopyMem, "Tensors of seq, strings, ref types and types with non-trivial destructors cannot be finalized by this proc"
    if storage.isMemOwner and not storage.memalloc.isNil:
      storage.memalloc.deallocShared()
      storage.memalloc = nil
else:
  proc `=destroy`[T](storage: var CpuStorageObj[T]) =
    when T is KnownSupportsCopyMem:
      if storage.isMemOwner and not storage.memalloc.isNil:
        storage.memalloc.deallocShared()
        storage.memalloc = nil
    else:
      `=destroy`(storage.raw_buffer)

  proc `=copy`[T](a: var CpuStorageObj[T]; b: CpuStorageObj[T]) {.error.}
  proc `=sink`[T](a: var CpuStorageObj[T]; b: CpuStorageObj[T]) {.error.}


proc allocCpuStorage*[T](storage: var CpuStorage[T], size: int) =
  ## Allocate aligned memory to hold `size` elements of type T.
  ## If T does not supports copyMem, it is also zero-initialized.
  ## I.e. Tensors of seq, strings, ref types or types with non-trivial destructors
  ## are always zero-initialized. This prevents potential GC issues.
  when T is KnownSupportsCopyMem:
    when not defined(gcDestructors):
      new(storage, finalizer[T])
    else:
      new(storage)
    storage.memalloc = allocShared(sizeof(T) * size + LASER_MEM_ALIGN - 1)
    storage.isMemOwner = true
    storage.raw_buffer = align_raw_data(T, storage.memalloc)
  else: # Always 0-initialize Tensors of seq, strings, ref types and types with non-trivial destructors
    new(storage)
    storage.raw_buffer.newSeq(size)

proc cpuStorageFromBuffer*[T: KnownSupportsCopyMem](
    storage: var CpuStorage[T],
    rawBuffer: pointer,
    size: int) =
  ## Create a `CpuStorage`, which stores data from a given raw pointer, which it does
  ## ``not`` own. The destructor/finalizer will be a no-op, because the memory is
  ## marked as not owned by the `CpuStorage`.
  ##
  ## The input buffer must be a raw `pointer`.
  when not defined(gcDestructors):
    new(storage, finalizer[T])
  else:
    new(storage)
  storage.memalloc = rawBuffer
  storage.isMemOwner = false
  storage.raw_buffer = cast[ptr UncheckedArray[T]](storage.memalloc)

func is_C_contiguous*(t: Tensor): bool =
  ## Check if the tensor follows C convention / is row major
  var cur_size = 1
  for i in countdown(t.rank - 1,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensions
    if t.shape[i] != 1 and t.strides[i] != cur_size:
        return false
    cur_size *= t.shape[i]
  return true

# ##################
# Raw pointer access
# ##################

# RawImmutableView and RawMutableView make sure that a non-mutable tensor
# is not mutated through it's raw pointer.
#
# Unfortunately there is no way to also prevent those from escaping their scope
# and outliving their source tensor (via `lent` destructors)
# and keeping the `restrict` and `alignment`
# optimization hints https://github.com/nim-lang/Nim/issues/7776
#
# Another anti-escape could be the "var T from container" and "lent T from container"
# mentionned here: https://nim-lang.org/docs/manual.html#var-return-type-future-directions

template unsafe_raw_offset_impl(offset: int) {.dirty.} =
  bind KnownSupportsCopyMem, withCompilerOptimHints, assume_aligned
  static: assert T is KnownSupportsCopyMem, "unsafe_raw access only supported for " &
    "mem-copyable types!"
  withCompilerOptimHints()
  when aligned:
    let raw_pointer{.restrict.} = assume_aligned t.storage.raw_buffer
  else:
    let raw_pointer{.restrict.} = t.storage.raw_buffer
  result = cast[type result](raw_pointer[offset].addr)

func unsafe_raw_buf*[T: KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): RawImmutableView[T] {.inline.} =
  ## Returns a view to the start of the data buffer
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(0)

func unsafe_raw_buf*[T: KnownSupportsCopyMem](t: var Tensor[T], aligned: static bool = true): RawMutableView[T] {.inline.} =
  ## Returns a view to the start of the data buffer
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(0)

func unsafe_raw_offset*[T: KnownSupportsCopyMem](t: Tensor[T], offset: int = 0, aligned: static bool = true): RawImmutableView[T] {.inline.} =
  ## Returns a view to the start of the valid data
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(t.offset+offset)

func unsafe_raw_offset*[T: KnownSupportsCopyMem](t: var Tensor[T], offset:int = 0, aligned: static bool = true): RawMutableView[T] {.inline.} =
  ## Returns a view to the start of the valid data
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(t.offset+offset)

func unsafe_raw_buf*[T: not KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): ptr UncheckedArray[T]  {.error: "Access via raw pointer forbidden for non mem copyable types!".}

func unsafe_raw_offset*[T: not KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): ptr UncheckedArray[T] {.error: "Access via raw pointer forbidden for non mem copyable types!".}

# macro raw_data_unaligned*(body: untyped): untyped =
#   ## Within this code block, all raw data accesses will not be
#   ## assumed aligned by default (LASER_MEM_ALIGN is 64 by default).
#   ## Use this when interfacing with external buffers of unknown alignment.
#   ##
#   ## ⚠️ Warning:
#   ##     At the moment Nim's builtin term-rewriting macros are not scoped.
#   ##     All processing within the file this is called will be considered
#   ##     unaligned. https://github.com/nim-lang/Nim/issues/7214#issuecomment-431567894.
#   block:
#     template trmUnsafeRawBuf{unsafe_raw_buf(x, aligned)}(x, aligned): auto =
#       {.noRewrite.}: unsafe_raw_buf(x, false)
#     template trmUnsafeRawOffset{unsafe_raw_offset(x, aligned)}(x, aligned): auto =
#       {.noRewrite.}: unsafe_raw_offset(x, false)
#     body

template `[]`*[T](v: RawImmutableView[T], idx: int): T =
  bind distinctBase
  distinctBase(type v)(v)[idx]

template `[]`*[T](v: RawMutableView[T], idx: int): var T =
  bind distinctBase
  distinctBase(type v)(v)[idx]

template `[]=`*[T](v: RawMutableView[T], idx: int, val: T) =
  bind distinctBase
  distinctBase(type v)(v)[idx] = val
