# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  std/[macros, strutils],
  ../compiler_optim_hints

template isVar[T](x: T): bool =
  ## Workaround due to `is` operator not working for `var`
  ## https://github.com/nim-lang/Nim/issues/9443
  compiles(addr(x))

proc aliasTensor(id: int, tensor: NimNode): tuple[alias: NimNode, isVar: NimNode] =
  ## Produce an alias variable for a tensor
  ## Supports:
  ## - identifiers
  ## - dot call for field access
  ## - foo[1] for array access
  if tensor.kind notin {nnkIdent, nnkSym, nnkDotExpr, nnkBracketExpr, nnkCall}:
    error "Expected a variable identifier or field access or array indexing but found \"" & tensor.repr() &
      "\". Please assign to a variable first."

  var t = tensor

  # nnkCall handles the annoying typed AST we get in generics
  # Call
  #   OpenSymChoice
  #     Sym "[]"
  #     Sym "[]"
  #     ...
  #   DotExpr
  #     Ident "self"
  #     Ident "first_moments"
  #   Ident "i"
  if tensor.kind == nnkCall:
    tensor[0].expectKind nnkOpenSymChoice
    tensor[0][0].expectKind nnkSym
    doAssert tensor[0][0].eqIdent("[]")

    # Rewrite the AST to untyped
    t = nnkBracketExpr.newTree(
      tensor[1]
    )
    for i in 2 ..< tensor.len:
      t.add tensor[i]

  let isVar = block:
    # Handle slicing cases like foo[0..<1, 0..<2]
    # that do not return `var` but are technically `var`
    # if `foo` is var
    if t.kind in {nnkDotExpr, nnkBracketExpr}:
      let t0 = t[0]
      quote do: isVar(`t0`)
    else:
      quote do: isVar(`t`)

  proc flattenedName(node: NimNode): string =
    if node.kind in {nnkDotExpr, nnkBracketExpr}:
      result = flattenedName(node[0])
      result &= '_'
      result &= flattenedName(node[1])
    elif node.kind in {nnkIdent, nnkSym}:
      result = $node
    else:
      error "Expected a field nameor dot expression or array access but found \"" &
        t.repr()

  let alias = flattenedName(t)

  return (newIdentNode($alias & "_alias" & $id & '_'), isVar)

proc initForEach*(
        params: NimNode,
        values, aliases, raw_ptrs: var NimNode,
        aliases_stmt, raw_ptrs_stmt: var NimNode,
        test_shapes: var NimNode
      ) =

  ### Parse the input
  values = nnkBracket.newTree()
  var tensors = nnkBracket.newTree()

  template syntaxError() {.dirty.} =
    error "Syntax error: argument " &
      ($arg.kind).substr(3) & " in position #" & $i & " was unexpected." &
      "\n    " & arg.repr()

  for i, arg in params:
    if arg.kind == nnkInfix:
      if eqIdent(arg[0], "in"):
        values.add arg[1]
        tensors.add arg[2]
    elif arg.kind == nnkStmtList:
      # In generic proc, symbols are resolved early
      # the "in" symbol will be transformed into an opensymchoice of "contains"
      # Note that arg order in "contains" is inverted compared to "in"
      if arg[0].kind == nnkCall and arg[0][0].kind == nnkOpenSymChoice and eqident(arg[0][0][0], "contains"):
        values.add arg[0][2]
        tensors.add arg[0][1]
      else:
        syntaxError()
    else:
      syntaxError()

  ### Initialization
  # First we need to alias the tensors, in our macro scope.
  # This is to ensure that an input like x[0..2, 1] isn't called multiple times
  # With move semantics this shouldn't cost anything.
  # We also take a pointer to the data
  aliases_stmt = newStmtList()
  aliases = nnkBracket.newTree()
  raw_ptrs_stmt = newStmtList()
  raw_ptrs = nnkBracket.newTree()

  aliases_stmt.add newCall(bindSym"withCompilerOptimHints")

  for i, tensor in tensors:
    let (alias, detectVar) = aliasTensor(i, tensor)
    aliases.add alias
    aliases_stmt.add quote do:
      when `detectVar`:
        var `alias`{.align_variable.} = `tensor`
      else:
        let `alias`{.align_variable.} = `tensor`

    let raw_ptr_i = genSym(nskLet, $alias & "_raw_data" & $i & '_')
    raw_ptrs_stmt.add quote do:
      let `raw_ptr_i`{.restrict.} = `alias`.unsafe_raw_offset()
    raw_ptrs.add raw_ptr_i

  let alias0 = aliases[0]
  test_shapes = newStmtList()
  for i in 1 ..< aliases.len:
    let alias_i = aliases[i]
    test_shapes.add quote do:
      assert `alias0`.shape == `alias_i`.shape,
        "\n " & astToStr(`alias0`) & ".shape is " & $`alias0`.shape & "\n" &
        "\n " & astToStr(`alias_i`) & ".shape is " & $`alias_i`.shape & "\n"

template stridedVarsSetup*(): untyped {.dirty.} =
  for i, alias in aliases:
    let iter_pos_i = gensym(nskVar, "iter" & $i & "_pos_")
    iter_pos.add iter_pos_i
    init_strided_iteration.add newVarStmt(iter_pos_i, newLit 0)
    iter_start_offset.add quote do:
      `iter_pos_i` += `coord`[`j`] * `alias`.strides[`j`]
    increment_iter_pos.add quote do:
      `iter_pos_i` += `alias`.strides[`k`]
    apply_backstrides.add quote do:
      `iter_pos_i` -= `alias`.strides[`k`] * (`alias`.shape[`k`]-1)


template stridedChunkOffset*(): untyped {.dirty.} =
  quote do:
    if `chunk_offset` != 0:
      var accum_size = 1
      for `j` in countdown(`alias0`.rank - 1, 0):
        `coord`[`j`] = (`chunk_offset` div accum_size) mod `alias0`.shape[`j`]
        `iter_start_offset`
        accum_size *= `alias0`.shape[`j`]

template stridedBodyTemplate*(): untyped {.dirty.} =
  quote do:
    # Initialisation
    `init_strided_iteration`

    # Iterator loop
    for _ in 0 ..< `chunk_size`:
      # Apply computation
      `body`

      # Next position
      for `k` in countdown(`alias0`.rank - 1, 0):
        if `coord`[`k`] < `alias0`.shape[`k`] - 1:
          `coord`[`k`] += 1
          `increment_iter_pos`
          break
        else:
          `coord`[`k`] = 0
          `apply_backstrides`
