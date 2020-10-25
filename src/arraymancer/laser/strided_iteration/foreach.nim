# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Strided parallel iteration for tensors
# This is generic and work on any tensor types as long
# as it implement the following interface:
#
# Tensor[T]:
#   Tensors data storage backend must be shallow copied on assignment (reference semantics)
#   the macro works on aliases to ensure that if the tensor is the result of another routine
#   that routine is only called once, for example x[0..<2, _] will not slice `x` multiple times.
#
# Routine and fields used, (routines mean proc, template, macros):
#   - rank, size:
#       routines or fields that return an int
#   - shape, strides:
#       routines or fields that returns an array, seq or indexable container
#       that supports `[]`. Read-only access.
#   - unsafe_raw_offset:
#       rountine or field that returns a ptr UncheckedArray[T]
#       or a distinct type with `[]` indexing implemented.
#       The address should be the start of the raw data including
#       the eventual tensor offset for subslices, i.e. equivalent to
#       the address of x[0, 0, 0, ...]
#       Needs mutable access for var tensor.
#
# Additionally the forEach macro needs an `is_C_contiguous` routine

import
  macros,
  ./foreach_common,
  ../private/ast_utils,
  ../openmp,
  ../compiler_optim_hints

export omp_suffix # Pending https://github.com/nim-lang/Nim/issues/9365 or 9366

proc forEachContiguousImpl(
  values, raw_ptrs, size, loopBody: NimNode,
  use_openmp: static bool,
  ): NimNode =
  # Build the parallel body of a contiguous iterator

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  let body = loopBody.replaceNodes(
                  replacements = elems_contiguous,
                  to_replace = values
                  )

  if use_openmp:
    result = quote do:
      omp_parallel_for_default(`index`, `size`):
          `body`
  else:
    result = quote do:
      for `index` in 0 ..< `size`:
        `body`

proc forEachStridedImpl(
  values, aliases,
  raw_ptrs, size,
  loopBody: NimNode,
  use_openmp: static bool
  ): NimNode =
  # Build the parallel body of a strided iterator

  var iter_pos = nnkBracket.newTree()
  var init_strided_iteration = newStmtList()
  var iter_start_offset = newStmtList()
  var increment_iter_pos = newStmtList()
  var apply_backstrides = newStmtList()

  let
    alias0 = aliases[0]
    coord = genSym(nskVar, "coord_")
    j = genSym(nskForVar, "j_mem_offset_") # Setting the start offset of each tensor iterator during init
    k = genSym(nskForVar, "k_next_elem_")  # Computing the next element in main body loop
    chunk_offset = newIdentNode("chunk_offset_")
    chunk_size =  if use_openmp: newIdentNode("chunk_size_")
                  else: size

  init_strided_iteration.add quote do:
    var `coord` {.align_variable.}: array[LASER_MEM_ALIGN, int]

  stridedVarsSetup()

  # Now add the starting memory offset to the init
  if use_openmp:
    init_strided_iteration.add stridedChunkOffset()

  var elems_strided = nnkBracket.newTree()
  for i, raw_ptr in raw_ptrs:
    elems_strided.add nnkBracketExpr.newTree(raw_ptr, iter_pos[i])

  let body = loopBody.replaceNodes(replacements = elems_strided, to_replace = values)
  let stridedBody = stridedBodyTemplate()

  let omp_grain_size = newLit( # scale grain_size down for strided operation
                        OMP_MEMORY_BOUND_GRAIN_SIZE div OMP_NON_CONTIGUOUS_SCALE_FACTOR
                      )
  if use_openmp:
    result = quote do:
      omp_parallel_chunks(
        `size`, `chunk_offset`, `chunk_size`,
        `omp_grain_size`):
          `stridedBody`
  else:
    result = stridedBody

template forEachSimpleTemplate(contiguous, use_openmp: static bool){.dirty.} =
  var
    params, loopBody, values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode

  params = args
  loopBody = params.pop()

  initForEach(
        params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  let body =  if contiguous:
                forEachContiguousImpl(
                  values, raw_ptrs, size, loopBody, use_openmp
                )
              else:
                forEachStridedImpl(
                  values, aliases, raw_ptrs, size, loopBody, use_openmp
                )
  let alias0 = aliases[0]

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      `body`

template forEachTemplate(use_openmp: static bool) {.dirty.} =
  var
    params, loopBody, values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode

  params = args
  loopBody = params.pop()

  initForEach(
        params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  let contiguous_body = forEachContiguousImpl(
    values, raw_ptrs, size, loopBody, use_openmp
  )
  let strided_body = forEachStridedImpl(
    values, aliases, raw_ptrs, size, loopBody, use_openmp
  )
  let alias0 = aliases[0]
  var test_C_Contiguous = newCall(ident"is_C_contiguous", alias0)
  for i in 1 ..< aliases.len:
    test_C_Contiguous = newCall(
                      ident"and",
                      test_C_Contiguous,
                      newCall(ident"is_C_contiguous", aliases[i])
                      )

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      if `test_C_Contiguous`:
        `contiguous_body`
      else:
        `strided_body`


macro forEachContiguous*(args: varargs[untyped]): untyped =
  ## Parallel iteration over one or more **contiguous** tensors.
  ##
  ## Format:
  ## forEachContiguous x in a, y in b, z in c:
  ##    x += y * z
  ##
  ## The threshold for parallelization by default is
  ## OMP_MEMORY_BOUND_GRAIN_SIZE = 1024 elementwise operations to process per cores.
  ##
  ## Compiler will also be hinted to unroll loop for SIMD vectorization.
  ##
  ## Use ``forEachStaged`` to fine-tune those defaults.
  forEachSimpleTemplate(contiguous = true, use_openmp = true)

macro forEachContiguousSerial*(args: varargs[untyped]): untyped =
  ## Serial iteration over one or more **contiguous** tensors.
  ##
  ## Format:
  ## forEachContiguousSerial x in a, y in b, z in c:
  ##    x += y * z
  forEachSimpleTemplate(contiguous = true, use_openmp = false)

macro forEachStrided*(args: varargs[untyped]): untyped =
  ## Parallel iteration over one or more tensors of unknown strides
  ## for example resulting from most slices.
  ##
  ## Format:
  ## forEachStrided x in a, y in b, z in c:
  ##    x += y * z
  ##
  ## The threshold for parallelization by default is
  ## OMP_MEMORY_BOUND_GRAIN_SIZE div OMP_NON_CONTIGUOUS_SCALE_FACTOR =
  ##   1024/4 = 256 elementwise operations to process per cores.
  ##
  ## Use ``forEachStaged`` to fine-tune this default.
  forEachSimpleTemplate(contiguous = false, use_openmp = true)

macro forEachStridedSerial*(args: varargs[untyped]): untyped =
  ## Serial iteration over one or more tensors of unknown strides
  ## for example resulting from most slices.
  ##
  ## Format:
  ## forEachStridedSerial x in a, y in b, z in c:
  ##    x += y * z
  forEachSimpleTemplate(contiguous = false, use_openmp = false)

macro forEach*(args: varargs[untyped]): untyped =
  ## Parallel iteration over one or more tensors
  ##
  ## Format:
  ## forEach x in a, y in b, z in c:
  ##    x += y * z
  ##
  ## The iteration strategy is selected at runtime depending of
  ## the tensors memory layout. If you know at compile-time that the tensors are
  ## contiguous or strided, use forEachContiguous or forEachStrided instead.
  ## Runtime selection requires duplicating the code body.
  ##
  ## In the contiguous case:
  ##   The threshold for parallelization by default is
  ##   OMP_MEMORY_BOUND_GRAIN_SIZE = 1024 elementwise operations to process per cores.
  ##
  ##   Compiler will also be hinted to unroll loop for SIMD vectorization.
  ##
  ## Otherwise if tensor is strided:
  ##   The threshold for parallelization by default is
  ##   OMP_MEMORY_BOUND_GRAIN_SIZE div OMP_NON_CONTIGUOUS_SCALE_FACTOR =
  ##     1024/4 = 256 elementwise operations to process per cores.
  ##
  ## Use ``forEachStaged`` to fine-tune this default.
  forEachTemplate(true)

macro forEachSerial*(args: varargs[untyped]): untyped =
  ## Serial iteration over one or more tensors
  ##
  ## Format:
  ## forEachSerial x in a, y in b, z in c:
  ##    x += y * z
  ##
  ## openMP parameters will be ignored
  ##
  ## The iteration strategy is selected at runtime depending of
  ## the tensors memory layout. If you know at compile-time that the tensors are
  ## contiguous or strided, use forEachContiguousSerial or forEachStridedSerial instead.
  ## Runtime selection requires duplicating the code body.
  forEachTemplate(false)
