# Laser & Arraymancer
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Types and low level primitives for tensors

import
  ../dynamic_stack_arrays, ../compiler_optim_hints

when NimVersion >= "1.1.0":
  # For distinctBase
  import typetraits
else:
  import sugar

type
  RawImmutableView*[T] = distinct ptr UncheckedArray[T]
  RawMutableView*[T] = distinct ptr UncheckedArray[T]

  Metadata* = DynamicStackArray[int]

  Tensor*[T] = object                    # Total stack: 128 bytes = 2 cache-lines
    shape*: Metadata                     # 56 bytes
    strides*: Metadata                   # 56 bytes
    offset*: int                         # 8 bytes
    storage*: CpuStorage[T]              # 8 bytes

  CpuStorage*[T] {.shallow.} = ref object # Total heap: 25 bytes = 1 cache-line
    # Workaround supportsCopyMem in type section - https://github.com/nim-lang/Nim/issues/13193
    when not(T is string or T is ref):
      raw_buffer*: ptr UncheckedArray[T] # 8 bytes
      memalloc*: pointer                 # 8 bytes
      isMemOwner*: bool                  # 1 byte
    else: # Tensors of strings, other ref types or non-trivial destructors
      raw_buffer*: seq[T]                # 8 bytes (16 for seq v2 backed by destructors?)

func rank*(t: Tensor): range[0 .. LASER_MAXRANK] {.inline.} =
  t.shape.len

func size*(t: Tensor): Natural =
  t.shape.product

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
  bind supportsCopyMem, withCompilerOptimHints, assume_aligned
  when supportsCopyMem(T):
    withCompilerOptimHints()
    when aligned:
      let raw_pointer{.restrict.} = assume_aligned t.storage.raw_buffer
    else:
      let raw_pointer{.restrict.} = t.storage.raw_buffer
    result = cast[type result](raw_pointer[offset].addr)
  else:
    result = cast[type result](t.storage.raw_buffer[offset].addr)

func unsafe_raw_buf*[T](t: Tensor[T], aligned: static bool = true): RawImmutableView[T] {.inline.} =
  ## Returns a view to the start of the data buffer
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(0)

func unsafe_raw_buf*[T](t: var Tensor[T], aligned: static bool = true): RawMutableView[T] {.inline.} =
  ## Returns a view to the start of the data buffer
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(0)

func unsafe_raw_offset*[T](t: Tensor[T], aligned: static bool = true): RawImmutableView[T] {.inline.} =
  ## Returns a view to the start of the valid data
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(t.offset)

func unsafe_raw_offset*[T](t: var Tensor[T], aligned: static bool = true): RawMutableView[T] {.inline.} =
  ## Returns a view to the start of the valid data
  ##
  ## Unsafe: the pointer can outlive the input tensor
  ## For optimization purposes, Laser will hint the compiler that
  ## while the pointer is valid, all data accesses will be through it (no aliasing)
  ## and that the data is aligned by LASER_MEM_ALIGN (default 64).
  unsafe_raw_offset_impl(t.offset)

macro raw_data_unaligned*(body: untyped): untyped =
  ## Within this code block, all raw data accesses will not be
  ## assumed aligned by default (LASER_MEM_ALIGN is 64 by default).
  ## Use this when interfacing with external buffers of unknown alignment.
  ##
  ## ⚠️ Warning:
  ##     At the moment Nim's builtin term-rewriting macros are not scoped.
  ##     All processing within the file this is called will be considered
  ##     unaligned. https://github.com/nim-lang/Nim/issues/7214#issuecomment-431567894.
  block:
    template trmUnsafeRawBuf{unsafe_raw_buf(x, aligned)}(x, aligned): auto =
      {.noRewrite.}: unsafe_raw_buf(x, false)
    template trmUnsafeRawOffset{unsafe_raw_offset(x, aligned)}(x, aligned): auto =
      {.noRewrite.}: unsafe_raw_offset(x, false)
    body

template `[]`*[T](v: RawImmutableView[T], idx: int): T =
  distinctBase(type v)(v)[idx]

template `[]`*[T](v: RawMutableView[T], idx: int): var T =
  distinctBase(type v)(v)[idx]

template `[]=`*[T](v: RawMutableView[T], idx: int, val: T) =
  distinctBase(type v)(v)[idx] = val
