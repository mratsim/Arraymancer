# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# This file implements the forEachStaged macro which allows multi-stage parallel for loop
# on a variadic number of tensors

import
  macros,
  tables, sets,
  ./foreach_common,
  ../private/ast_utils,
  ../openmp,
  ../compiler_optim_hints
export omp_suffix

proc forEachStagedContiguousImpl(
  values, raw_ptrs, size, loopBody: NimNode,
  use_simd, nowait: bool,
  ): NimNode =
  # Build the body of a contiguous iterator
  # Whether this is parallelized or not should be
  # handled at a higher level

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  let body = loopBody.replaceNodes(
                    replacements = elems_contiguous,
                    to_replace = values
                  )

  let # quote do, doesn't interpolate properly to static
      # We get "undeclared identifier contiguousIndex_" otherwise
    use_simd_node = newLit use_simd
    nowait_node = newLit nowait

  result = quote do:
    omp_for(`index`, `size`, `use_simd_node`, `nowait_node`):
      `body`

proc forEachStagedStridedImpl(
  values, aliases,
  raw_ptrs, size,
  loopBody: NimNode,
  use_openmp, nowait: bool
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

  if use_openmp:
    result = newStmtList()
    result.add quote do:
      omp_chunks(
        `size`, `chunk_offset`, `chunk_size`,
        `stridedBody`
      )
    if nowait:
      result.add getAST(omp_barrier())
  else:
    result = stridedBody

template forEachStagedSimpleTemplate(contiguous: static bool){.dirty.} =
  let body =  if contiguous:
                forEachStagedContiguousImpl(
                  values, raw_ptrs, size, in_loop_body, use_simd, nowait
                )
              else:
                forEachStagedStridedImpl(
                  values, aliases, raw_ptrs, size, in_loop_body, use_openmp, nowait
                )
  let alias0 = aliases[0]
  var scaled_grain_size = newIdentNode("scaled_grain_size_")
  if contiguous:
    scaled_grain_size = newLit omp_grain_size
  else:
    scaled_grain_size = newLit(omp_grain_size div OMP_NON_CONTIGUOUS_SCALE_FACTOR)

  if use_openmp:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        let over_threshold = `scaled_grain_size` * omp_get_max_threads() < `size`
        omp_parallel_if(over_threshold):
          `before_loop_body`
          `body`
          `after_loop_body`
  else:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        `before_loop_body`
        `body`
        `after_loop_body`

template forEachStagedTemplate(){.dirty.} =
  let contiguous_body = forEachStagedContiguousImpl(
                          values, raw_ptrs, size, in_loop_body, use_simd, nowait
                        )
  let strided_body =  forEachStagedStridedImpl(
                        values, aliases, raw_ptrs, size, in_loop_body, use_openmp, nowait
                      )

  let alias0 = aliases[0]
  var test_C_Contiguous = newCall(ident"is_C_contiguous", alias0)
  for i in 1 ..< aliases.len:
    test_C_Contiguous = newCall(
                          ident"and",
                          test_C_Contiguous,
                          newCall(ident"is_C_contiguous", aliases[i])
                        )
  if use_openmp:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        let is_C_contiguous = `test_C_Contiguous`
        let over_threshold = block:
          if is_C_contiguous:
            `omp_grain_size` * omp_get_max_threads() < `size`
          else:
            (`omp_grain_size` div OMP_NON_CONTIGUOUS_SCALE_FACTOR) *
              omp_get_max_threads() < `size`
        omp_parallel_if(over_threshold):
          `before_loop_body`
          if is_C_contiguous:
            `contiguous_body`
          else:
            `strided_body`
          `after_loop_body`
  else:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        `before_loop_body`
        if `test_C_Contiguous`:
          `contiguous_body`
        else:
          `strided_body`
        `after_loop_body`

proc checkBlocks(blocks: NimNode) =
  var counts = initCountTable[string](initialSize = 8)
  let valid_blocks = [
        "openmp_config", "iteration_kind",
        "before_loop", "in_loop", "after_loop"
      ].toSet
  for node in blocks:
    node.expectKind nnkCall
    let section = $node[0]
    assert section in valid_blocks, "Only the following sections are allowed: " & $valid_blocks
    counts.inc section
  for key, count in counts:
    if count > 1:
      error "\"" & key & "\" can only be defined once but is defined " & $count & " times."

type IterKind = enum
  Contiguous, Strided, Both

proc parseBlocks(
    use_openmp, use_simd, nowait: var NimNode,
    omp_grain_size: var NimNode,
    iter_kind: var IterKind,
    before_loop_body, in_loop_body, after_loop_body: var NimNode,
    dslBlocks: NimNode
  ) =
  var params = [
    "use_openmp", "use_simd", "nowait",
    "omp_grain_size",
    "iteration_kind",
    "before_loop", "in_loop", "after_loop"
  ].toOrderedSet # after_loop needs to be processed after nowait

  ## Parsing
  for blck in dslBlocks:
    blck[0].expectKind nnkIdent
    blck[1].expectKind nnkStmtList
    if eqIdent(blck[0], "openmp_config"):
      for param in blck[1]:
        param[1].expectKind nnkStmtList
        if eqIdent(param[0], "use_openmp"):
          let missing = missingOrExcl(params, $param[0])
          assert missing.not, "`use_openmp` is defined more than once."
          use_openmp = param[1]
        elif eqIdent(param[0], "use_simd"):
          let missing = missingOrExcl(params, $param[0])
          assert missing.not, "`use_simd` is defined more than once."
          use_simd = param[1]
        elif eqIdent(param[0], "nowait"):
          let missing = missingOrExcl(params, $param[0])
          assert missing.not, "`nowait` is defined more than once."
          nowait = param[1]
        elif eqIdent(param[0], "omp_grain_size"):
          let missing = missingOrExcl(params, $param[0])
          assert missing.not, "`omp_grain_size` is defined more than once."
          omp_grain_size = param[1]
        else:
          error "In \"openmp_config\" only the following parameters are allowed: use_openmp, use_simd, nowait, omp_grain_size"
    elif eqIdent(blck[0], "iteration_kind"):
      assert blck[1].len == 1
      params.excl "iteration_kind"
      if blck[1][0].kind in {nnkIdent, nnkSym}:
        if blck[1][0].eqIdent("contiguous"):
          iter_kind = Contiguous
        elif blck[1][0].eqIdent("strided"):
          iter_kind = Strided
        else:
          error "Invalid iteration kind " & $blck[1][0]
      elif blck[1][0].kind == nnkCurly:
        let valid_iter_kind = block:
          blck[1][0].len == 2 and (
            ($blck[1][0][0] == "contiguous" and $blck[1][0][1] == "strided") or
            ($blck[1][0][0] == "serial" and $blck[1][0][0] == "contiguous")
          )
        assert valid_iter_kind, "Invalid iteration kind " & blck[1][0].repr
      else:
        error "Invalid iteration kind " & blck[1][0].repr
    elif eqIdent(blck[0], "before_loop"):
      params.excl "before_loop"
      before_loop_body = blck[1]
    elif eqIdent(blck[0], "in_loop"):
      params.excl "in_loop"
      in_loop_body = blck[1]
    elif eqIdent(blck[0], "after_loop"):
      params.excl "after_loop"
      after_loop_body = blck[1]
    else:
      error "Invalid section " & $blck[0] # This should be caught by checkBlocks

  ## Default values
  for param in params:
    case param
    of "use_openmp": use_openmp = newLit true
    of "use_simd": use_simd = newLit true
    of "nowait": nowait = newLit false
    of "omp_grain_size": omp_grain_size = newLit OMP_MEMORY_BOUND_GRAIN_SIZE
    of "iteration_kind": iter_kind = Both
    of "before_loop":
      before_loop_body =  nnkStmtList.newTree(
                            nnkDiscardStmt.newTree(newEmptyNode())
                          )
    of "in_loop": error "`in_loop` section is required"
    of "after_loop":
      after_loop_body = nnkStmtList.newTree(
                            nnkDiscardStmt.newTree(newEmptyNode())
                        )
      nowait = newLit true # There is always a barrier after #pragma omp parallel, no need to double it.

macro forEachStagedAux(
    use_openmp, use_simd, nowait: static bool,
    omp_grain_size: static Natural,
    iteration_kind: static IterKind,
    before_loop_body, in_loop_body, after_loop_body: untyped,
    params: varargs[untyped]
  ): untyped =

  var
    values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode

  initForEach(
        params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  case iteration_kind
  of Contiguous: forEachStagedSimpleTemplate(contiguous = true)
  of Strided: forEachStagedSimpleTemplate(contiguous = false)
  of Both: forEachStagedTemplate()

macro forEachStaged*(args: varargs[untyped]): untyped =
  ## Staged optionally parallel iteration over one or more tensors
  ## This is useful if you need thread-local initialization or cleanup before the parallel loop
  ## Example usage for reduction
  ##
  ## forEachStaged xi in x, yi in y:
  ##   openmp_config:
  ##     use_openmp: true
  ##     use_simd: false
  ##     nowait: true
  ##     omp_grain_size: OMP_MEMORY_BOUND_GRAIN_SIZE
  ##   iteration_kind:
  ##     {contiguous, strided} # Default, "contiguous", "strided" are also possible
  ##   before_loop:
  ##     var local_sum = 0.T
  ##   in_loop:
  ##     local_sum += xi + yi
  ##   after_loop:
  ##     omp_critical:
  ##       result += local_sum

  var
    params, dslBlocks: NimNode

    use_openmp, use_simd, nowait: NimNode
    omp_grain_size: NimNode
    iteration_kind: IterKind
    before_loop_body, in_loop_body, after_loop_body: NimNode

  params = args
  dslBlocks = params.pop()

  checkBlocks dslBlocks
  parseBlocks(
    use_openmp, use_simd, nowait,
    omp_grain_size,
    iteration_kind,
    before_loop_body, in_loop_body, after_loop_body,
    dslBlocks
  )

  result = quote do:
    forEachStagedAux(
      `use_openmp`, `use_simd`, `nowait`,
      `omp_grain_size`,
      IterKind(`iteration_kind`),
      `before_loop_body`, `in_loop_body`, `after_loop_body`,
      `params`
    )
