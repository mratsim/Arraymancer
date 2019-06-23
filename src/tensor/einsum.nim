import macros, sequtils, sets, algorithm
import tensor

template `^^`(s, i: untyped): untyped =
  (when i is BackwardsIndex: s.len - int(i) else: int(i))

proc slice[T, U](n: NimNode, s: HSlice[T, U]): seq[NimNode] =
  ## returns the slice `s` of the children of `n`
  let a = n ^^ s.a
  let b = n ^^ s.b
  doAssert n.len > b, " N " & $n.len & " and b " & $b
  doAssert a >= 0
  for i in a .. b:
    result.add n[i]

proc buildLoops(rank: int,
                idxIdentPairs: seq[(string, int)],
                shapeIdent: NimNode, innerStatement: NimNode): NimNode =
  # generate the for loops
  var forLoops = nnkForStmt.newTree()
  var stmtInLoop = newNimNode(nnkNilLit)
  for i in 0 ..< rank:
    let shapeIdx = idxIdentPairs[i][1]
    let forIdx = ident(idxIdentPairs[i][0])
    let toIdx = quote do:
      `shapeIdent`[`shapeIdx`]
    var loop = nnkForStmt.newTree(
      forIdx,
      nnkInfix.newTree(
        ident"..<",
        newLit 0,
        toIdx
      )
    )
    if stmtInLoop.kind == nnkNilLit:
      stmtInLoop = innerStatement
    else:
      stmtInLoop = forLoops

    loop.add stmtInLoop
    forLoops = loop
  result = forLoops

proc getTensors(tensors: NimNode): seq[NimNode] =
  ## extracts all tensors from the `tensors: varargs[typed]` argument of
  ## the macro and checks if they are symbols. Returns them as a seq.
  # NOTE: if an argument to `einsum` contains an undefined identifier, the
  # compiler will error out with `undeclared identifier` before we get here
  for t in tensors:
    if t.kind == nnkSym:
      result.add t
    else:
      error("Argument to `einsum` must be a number of defined tensors!")

type
  StatementKind = enum
    skAssign, # specific assignment of summation to an existing tensor
    skAuto # automatic deduction of the resulting tensor

proc checkStatement(stmts: NimNode): StatementKind =
  ## checks what kind of statement `einsum` was given. Either a simple product
  ## of `nnkInfix` using `*` without assignment (deduce resulting tensor
  ## automatically) or `nnkAsgn` assigning to an existing tensor.
  if stmts.len > 1 or stmts.len == 0:
    error("Only a single statement allowed for `einsum`!")
  let stmt = stmts[0]
  case stmt.kind
  of nnkInfix:
    # TODO: also check nested infix for multiple tensors
    doAssert stmt[0].ident == toNimIdent("*"), "It ``must`` be a product `*` " &
      "between the tensors!"
    result = skAuto
  of nnkAsgn:
    result = skAssign
  else:
    error("`einsum` statement must not be of kind `" & $stmt.kind & "`!")

type
  TensorIdx = object
    t: NimNode # the tensor ident
    idx: seq[NimNode] # the corresponding indices

proc getTensorIdx(tensors: NimNode, tensorArgument: seq[NimNode]): seq[TensorIdx] =
  ## Iterate over the `nnkInfix` RHS of the `einsum` statement.
  ## Also compares the tensors in the statement to the tensors the user gave
  ## as the `typed` argument to the macro.
  ## Returns a sequence of TensorIdx objects, i.e. a tensor ident together with the
  ## associated indices.
  proc extractIdentIdx(n: NimNode, compare: NimNode): TensorIdx =
    let tC = n[0]
    doAssert $tC == $compare
    let tIdx = slice(n, 1 .. ^1)
    result = TensorIdx(t: tC, idx: tIdx)
  case tensors.kind
  of nnkBracketExpr:
    # only a single tensor, probably in an `skAssign` statement
    doAssert tensorArgument.len == 1, "If only a single tensor is used in the " &
      "statement of `einsum`, only a single argument may be given!"
    result = @[extractIdentIdx(tensors, tensorArgument[0])]
  of nnkInfix:
    doAssert tensors[0].ident == toNimIdent"*"
    if tensors[1].kind == nnkInfix:
      result.add getTensorIdx(tensors[1], tensorArgument)
      result.add getTensorIdx(tensors[2], @[tensorArgument[^1]])
    else:
      result.add getTensorIdx(tensors[1], @[tensorArgument[0]])
      result.add getTensorIdx(tensors[2], @[tensorArgument[1]])
  else:
    error("Unsupported kind " & $tensors.kind)

proc findAxes(idxSet: HashSet[string], tensors: seq[TensorIdx]): seq[(int, int)] =
  ## Find the first axis in which each index in `idxSet` appears in the
  ## tensors. E.g.
  ## idxSet = {i, j, k}
  ## tensors: @[(t: a, idx: @[i, j]]), (t: b, idx: @[j, k])]
  ## index i: a contains i, is at position 0 in `tensors`. `i` itself is
  ##          index 0 of `a.idx`
  ## index j: a contains j, is at position 0 in `tensors`. `j` itself is
  ##          index 1 of `a.idx`. `b` contains `j` too, but since a's axis 1 and
  ##          b's axis 0 ``must`` match in length anyways, we can ignore b.
  ## index i: a contains i, is at position 0 in `tensors`. `i` itself is
  ##          index 0 of `a.idx`
  ## -> @[(0, 0), (0, 1), (1, 0)]
  for idx in idxSet:
    for i, tIdx in tensors:
      let idxStr = tIdx.idx.mapIt($it)
      let at = find(idxStr, $idx)
      if at >= 0:
        result.add (i, at)
        # found this index, can break from search
        break

proc toDuplicates[T](s: seq[T]): HashSet[T] =
  ## creates a set of elements ``only`` consisting of the duplicate elements
  ## in `s`. Unique elements will ``not`` show up in the resulting set.
  var tmp = initHashSet[T]()
  for x in s:
    if x notin tmp:
      tmp.incl x
    else:
      # already in `tmp`, so it's a duplicate. Add it to result
      result.incl x

proc toUnique[T](s: seq[T]): HashSet[T] =
  ## creates a set of elements, which ``only`` contains the unique
  ## elements of `s`. Any duplicate will ``not`` appear in the resulting
  ## set.
  let duplicates = toDuplicates(s)
  for x in s:
    if x notin duplicates:
      result.incl x

proc splitLhsRhs(stmtKind: StatementKind,
                 stmt: NimNode): (NimNode, OrderedSet[string], NimNode) =
  ## Returns the einsum statement of the LHS, the LHS indices in an ordered set
  ## and the RHS statements. If `stmtKind` is `skAuto` however, `lhsStmt` will
  ## be a nnkNilLit and the OrderedSet the empty set.
  # node holding RHS of `stmt`
  var rhsStmt: NimNode
  # node of LHS, ``iff`` `stmtKind` is `skAssign`
  var lhsStmt: NimNode
  var idxLHS: OrderedSet[string]
  if stmtKind == skAssign:
    # in case of assign, slice off the infix part
    rhsStmt = stmt[0][1]
    lhsStmt = stmt[0][0]
    case lhsStmt.kind
    of nnkIdent:
      # left is an ident, so result supposed to be a scalar. Indidces empty set
      idxLHS = initOrderedSet[string]()
    of nnkBracketExpr:
      idxLHS = toOrderedSet(slice(lhsStmt, 1 .. ^1).mapIt($it))
    else:
      error("Unsupported kind for `einsum` LHS statement " & $lhsStmt.kind)
  else:
    rhsStmt = stmt[0]
  result = (lhsStmt, idxLhs, rhsStmt)

proc shapeAssertions(tensorIdxSeq: seq[TensorIdx]): NimNode =
  ## generates the shape assertions for the tensor ranks that are required,
  ## i.e. that the number of supplied indices corresponds to the rank of the
  ## input tensors.
  for tIdx in tensorIdxSeq:
    let t = tIdx.t
    let idx = tIdx.idx
    let nIdx = idx.len
    result = quote do:
      doAssert `t`.rank == `nIdx`

iterator enumerateIdx(s: seq[TensorIdx]): string =
  ## enumerates all indices of the tensors in `s` as strings
  for t in s:
    for idx in t.idx:
      yield $idx

macro einsum*(tensors: varargs[typed], stmt: untyped): untyped =
  ## Performs Einstein summation of the given `tensors` defined by the
  ## `stmt`.
  doAssert stmt.len == 1, "There may only be a single statement in `einsum`!"
  result = newStmtList()
  # extract all tensors by checking if they are all symbols
  let ts = getTensors(tensors)

  # determine what kind of statement is given, e.g.
  # skAssign: res[i,j] = a[i,j] * b[i,j]
  # skAuto: a[i,j] * b[i,j]
  let stmtKind = checkStatement(stmt)
  # get LHS, RHS statements, possible LHS indices
  let (lhsStmt, idxLhs, rhsStmt) = splitLhsRhs(stmtKind, stmt)

  let tensorIdxSeq = getTensorIdx(rhsStmt, ts)
  # first build a union of all indices
  let idxAllSeq = concat(tensorIdxSeq.mapIt(it.idx)).mapIt($it)

  # add shape assertion statements
  result.add shapeAssertions(tensorIdxSeq)

  # use to create sets of resulting and contracting indices
  var idxRes = toUnique(idxAllSeq)
  var idxContr = toDuplicates(idxAllSeq)

  # compare `idxContr` deduced from the RHS with the indices of LHS, if assignment
  if stmtKind == skAssign:
    # for the assignment case we may have to modify the `idxRes` and `idxContr` based on what
    # `idxLhs` shows. Any index that still appears in `idxLhs` must be taken out of
    # `idxContr` and added to `idxRes`, because this means the user wishes to exclude
    # contraction of that index. I.e. the case for the `Hadamard product`:
    # res[i,j] = m[i,j] * n[i,j]
    # product wise multiplication
    for idx in idxLhs:
      if idx in idxContr:
        idxContr.excl idx
        idxRes.incl idx
    # on the other hand for any index in union(`idxRes`, `idxContr`), but not
    # in `idxLhs`, must be removed from `idxRes` and added to `idxContr`
    for idx in union(idxRes, idxContr):
      if idx notin idxLhs:
        idxContr.incl idx
        idxRes.excl idx

  # now we can safely calculate the rank of the tensor
  let rank = idxRes.card

  # find each axis of each tensor, which will surive
  # seq of tuples of `tensor`, `axis` pairs
  let tensorAxes = findAxes(idxRes, tensorIdxSeq)
  let contractionAxes = findAxes(idxContr, tensorIdxSeq)

  # the sequence storing the NimNode for the `i`, `j`,... einstein index
  # and corresponding it to the correct index for the `shape*Idents` sequence
  var idxIdentPairs = newSeq[(string, int)]()


  # generate the code to get the shape of the resulting tensor
  let shapeIdents = ident"shapes"
  if rank > 0:
    # add a `shapes` variable, only if the resulting shape is
    result.add quote do:
      var `shapeIdents` = newSeq[int](`rank`)

  case stmtKind
  of skAssign:
    for i, idx in idxLhs:
      # since `shapeIdents` corresponds to the shape of the resulting
      # tensor, use the `LHS` (if skAssign) to order them in the
      # correct way
      var idxArg: int
      var t: NimNode
      for ax in tensorAxes:
        if $tensorIdxSeq[ax[0]].idx[ax[1]] == idx:
          idxArg = ax[1]
          t = tensorIdxSeq[ax[0]].t
          idxIdentPairs.add (idx, i)
          break
      doAssert t.kind != nnkNilLit, "Something went wrong, could not find tensor!"
      if rank > 0:
        # only add to `shapes` variable, if rank > 0
        result.add quote do:
          `shapeIdents`[`i`] = `t`.shape[`idxArg`]
  of skAuto:
    var i = 0
    for idx in enumerateIdx(tensorIdxSeq):
      if idx in idxRes:
        # since `shapeIdents` corresponds to the shape of the resulting
        # tensor, use the `LHS` (if skAssign) to order them in the
        # correct way
        var idxArg: int
        var t: NimNode
        for ax in tensorAxes:
          if $tensorIdxSeq[ax[0]].idx[ax[1]] == idx:
            idxArg = ax[1]
            t = tensorIdxSeq[ax[0]].t
            idxIdentPairs.add (idx, i)
            break
        result.add quote do:
          `shapeIdents`[`i`] = `t`.shape[`idxArg`]
        inc i

  var idxIdentContrPairs = newSeq[(string, int)]()
  for i, idx in toSeq(idxContr):
    for ax in contractionAxes:
      if $tensorIdxSeq[ax[0]].idx[ax[1]] == idx:
        idxIdentContrPairs.add (idx, i)
        break

  # generate the code to get the shape of the contraction
  let shapeContrIdents = ident"shapesContr"
  let rankContr = idxContr.card
  result.add quote do:
    var `shapeContrIdents` = newSeq[int](`rankContr`)
    echo "Rank contr ", `rankContr`
  for i, ax in contractionAxes:
    let t = tensorIdxSeq[ax[0]].t
    let idx = ax[1]
    result.add quote do:
      `shapeContrIdents`[`i`] = `t`.shape[`idx`]


  # identifier for the variable storing the temporary result (tensor / scalar),
  # which will be the result of the macro's `block`
  let resIdent = ident"tmp"
  # generate the result tensor
  if rank == 0:
    result.add quote do:
      echo `shapeContrIdents`
      var `resIdent` = 0.0
  else:
    result.add quote do:
      var `resIdent` = newTensor[float](`shapeIdents`)
      echo "Shapes ", `shapeIdents`.len
      echo "Shapes ", `shapeIdents`
      echo "Tmp ", `resIdent`.shape

  # now either get the assignment from the user or build it from
  # `idxRes`
  # TODO: Check whether user's indices in the assignment are the same
  # as our `idxRes` calculation
  var asgnTo: NimNode
  case stmtKind
  of skAssign:
    asgnTo = lhsStmt
    # replace the identifier, use the `tmp` instead of the actual `a`
    # TODO: have an inplace version?
    case asgnTo.kind
    of nnkIdent:
      # case of scalar result, use `resIdent` as total result
      asgnTo = resIdent
    of nnkBracketExpr:
      # result is a tensor
      asgnTo[0] = resIdent
    else:
      error("Unsupported kind for assignment " & $asgnTo.kind)
  else:
    if rank > 0:
      # generate bracket to acceess element
      asgnTo = nnkBracketExpr.newTree(resIdent)
      # now assign the indices we access by the order in which they appear
      # in the input statement
      for idx in enumerateIdx(tensorIdxSeq):
        if idx in idxRes:
          asgnTo.add ident(idx)
    else:
      # scalar result from implicit call
      asgnTo = resIdent

  let prod = rhsStmt
  let contrRes = ident"res"

  var contractionLoops: NimNode
  if rankContr > 0:
    let innerStmt = quote do:
      `contrRes` += `prod`

    contractionLoops = newStmtList()
    contractionLoops.add quote do:
      var `contrRes`: float
    contractionLoops.add buildLoops(rankContr,
                                    idxIdentContrPairs,
                                    shapeContrIdents,
                                    innerStmt)
    contractionLoops.add quote do:
      `asgnTo` = `contrRes`
  else:
    # in this case we have no contraction. Use variable for inner
    # stamtent
    contractionLoops = quote do:
      `asgnTo` = `rhsStmt` # we could just assign `stmt`, but this for clarity

  if rank > 0:
    let forLoops = buildLoops(rank,
                              idxIdentPairs,#idxRes,
                              shapeIdents, contractionLoops)
    result.add forLoops
  else:
    # since we build no loop outer loop, also have to assign the result of the
    # contraction loop
    result.add contractionLoops
    result.add quote do:
      `asgnTo` = `contrRes`

  # put everything into a block and return tmp tensor
  result = quote do:
    block:
      `result`
      `resIdent`

  echo "Result ", result.repr

when isMainModule:
  let c0 = toSeq(11..34).toTensor.astype(float)
  let d0 = toSeq(1..6).toTensor.astype(float)
  let c = c0.reshape(2, 2, 3, 2)
  let d = d0.reshape(3, 2)

  echo c
  echo d

  let t = einsum(c, d):
    let i, j, k, l = EinsumIndex
    c[i,j,k,l] * d[k,l]

  echo t.shape
  echo t
