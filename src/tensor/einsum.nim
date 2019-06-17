import macros, sequtils, sets, algorithm
import arraymancer

proc getEinsumIdx(n: NimNode): seq[NimNode] =
  for i in 0 ..< n.len - 1:
    # don't want last element, since that is `EinsumIndex`
    let ch = n[i]
    if ch.kind == nnkIdent:
      result.add ch

template `^^`(s, i: untyped): untyped =
  (when i is BackwardsIndex: s.len - int(i) else: int(i))

proc slice[T, U](n: NimNode, s: HSlice[T, U]): seq[NimNode] =
  ## returns the slice `s` of the children of `n`
  echo "s ", s.repr
  let a = n ^^ s.a
  let b = n ^^ s.b
  doAssert n.len > b, " N " & $n.len & " and b " & $b
  doAssert a >= 0
  for i in a .. b:
    result.add n[i]

proc buildLoops(rank: int,
                idxIdentPairs: seq[(string, int)], #idxRes: HashSet[string],
                shapeIdent: NimNode, innerStatement: NimNode): NimNode =
  # generate the for loops
  var forLoops = nnkForStmt.newTree()
  #let idxResSeq = toSeq(idxRes)
  var stmtInLoop = newNimNode(nnkNilLit)
  for i in 0 ..< rank:
    echo "idxIdentPairs, ", idxIdentPairs
    let shapeIdx = idxIdentPairs[i][1]  #rank - i - 1
    let forIdx = ident(idxIdentPairs[i][0]) #ident(idxResSeq[i])
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
    #let idxI = ident(idxResSeq[i])
    if stmtInLoop.kind == nnkNilLit:
      #  echo `resIdent`[`idxI`, `idxI`]
      stmtInLoop = innerStatement
    else:
      stmtInLoop = forLoops

    loop.add stmtInLoop
    forLoops = loop
  result = forLoops

proc getTensors(tensors: NimNode): seq[NimNode] =
  ## extracts all tensors from the `tensors: varargs[typed]` argument of
  ## the macro and checks if they are symbols. Returns them as a seq.
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

proc getTensorSeq(tensors: NimNode, tensorArgument: seq[NimNode]): seq[(NimNode, seq[NimNode])] =
  ## Iterate over the `nnkInfix` RHS of the `einsum` statement.
  ## Also compares the tensors in the statement to the tensors the user gave
  ## as the `typed` argument to the macro.
  proc extractIdentIdx(n: NimNode, compare: NimNode): (NimNode, seq[NimNode]) =
    echo n.treeRepr
    let tC = n[0]
    doAssert $tC == $compare
    let tIdx = slice(n, 1 .. ^1)
    result = (tC, tIdx)
  case tensors.kind
  of nnkBracketExpr:
    # only a single tensor, probably in an `skAssign` statement
    doAssert tensorArgument.len == 1, "If only a single tensor is used in the " &
      "statement of `einsum`, only a single argument may be given!"
    result = @[extractIdentIdx(tensors, tensorArgument[0])]
  of nnkInfix:
    echo "Tensors ", tensors.treeRepr
    echo "Args ", tensorArgument
    echo tensors.len
    echo tensorArgument.len
    doAssert tensors[0].ident == toNimIdent"*"
    #doAssert tensors.len == tensorArgument.len, "Every tensor given as an argument " &
    #  "to `einsum` must be used in the statement in the body! "
    if tensors[1].kind == nnkInfix:
      result.add getTensorSeq(tensors[1], tensorArgument)
      result.add getTensorSeq(tensors[2], @[tensorArgument[^1]])
    else:
      result.add getTensorSeq(tensors[1], @[tensorArgument[0]])
      result.add getTensorSeq(tensors[2], @[tensorArgument[1]])
    #for i in 1 ..< tensors.len:
    #  echo "Index ", i, " at "
    #  echo tensors[i].repr
    #  echo " and ", tensorArgument[i - 1].repr
    #  result.add getTensorSeq(tensors[i], @[tensorArgument[i - 1]])
  else:
    error("Unsupported kind " & $tensors.kind)
  echo "Result ", result

proc findAxes(idxRes: HashSet[string], tensors: seq[(NimNode, seq[NimNode])]): seq[(int, int)] =
  for idx in idxRes:
    for i, tup in tensors:
      let tIdx = tup[1].mapIt($it)
      let at = find(tIdx, $idx)
      if at >= 0:
        result.add (i, at)
        # found this index, can break from search
        break
  #result = result.reversed

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

macro einsum*(tensorInput: varargs[typed], stmts: untyped): untyped =
  echo tensorInput.treeRepr
  let ts = getTensors(tensorInput)

  var
    t1: NimNode
    t2: NimNode
  if ts.len > 1:
    t1 = ts[0]
    t2 = ts[1]
  else:
    t1 = ts[0]
  echo stmts.treeRepr

  let stmtKind = checkStatement(stmts)
  var rhsStmt: NimNode
  var lhsStmt: NimNode
  var idxLHS: OrderedSet[string]
  echo stmtKind
  if stmtKind == skAssign:
    # in case of assign, slice off the infix part
    rhsStmt = stmts[0][1]
    echo rhsStmt.treeRepr
    lhsStmt = stmts[0][0]

    case lhsStmt.kind
    of nnkIdent:
      # left is an ident, so result supposed to be a scalar. Indidces empty set
      idxLHS = initOrderedSet[string]()
    of nnkBracketExpr:
      idxLHS = toOrderedSet(slice(lhsStmt, 1 .. ^1).mapIt($it))
    else:
      error("Unsupported kind for `einsum` LHS statement " & $lhsStmt.kind)
  else:
    rhsStmt = stmts[0]

  # TODO: allow nested infix w/ more than 2 tensors
  let tensorIdxPairs = getTensorSeq(rhsStmt, ts)
  echo "TENSOR IDX PAIRS ", tensorIDxPairs

  let inputOrderIdents = toOrderedSet(
    concat(tensorIdxPairs.mapIt(it[1])).mapIt($it)
  )

  let resIdent = ident"tmp"

  result = newStmtList()

  # generate shape assertion statements
  for tup in tensorIdxPairs:
    let t = tup[0]
    let idx = tup[1]
    let nIdx = idx.len
    echo "T ", t.treeRepr
    echo "idx ", idx
    echo "nIdx ", nIdx
    result.add quote do:
      doAssert `t`.rank == `nIdx`
  echo result.repr

  # generate the rank of the resulting tensor
  # TODO: avoid that `mapIt`
  # Init both with the first tensor
  # TODO: how to keep order sane? Walk the actual statements again
  # after extracting the set? Do we even need it?
  #var idxRes = toSet(tensorIdxPairs[0][1].mapIt($it))
  #var idxContr = toSet(tensorIdxPairs[0][1].mapIt($it))
  #if tensorIdxPairs.len == 1:
  #  # `idxRes` is unchanged, all indices are the resulting indices
  #  # for `idxContr` there cannot be contracted indices in case of a single argument
  #  idxContr = initHashSet[string]()

  # first build a union of all indices
  let idxAllSeq = concat(tensorIdxPairs.mapIt(it[1])).mapIt($it)
  var idxRes = toUnique(idxAllSeq)
  var idxContr = toDuplicates(idxAllSeq)

  echo "Idx Res ", idxRes
  echo "Idx Contr ", idxContr

  # compare `idxContr` deduced from the RHS with the indices of LHS, if assignment
  if stmtKind == skAssign:
    # for the assignment case we may have to modify the `idxRes` and `idxContr` based on what
    # `idxLHS` shows. Any index that still appears in `idxLHS` must be taken out of
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

    # special case, Hadamard product, LHS indices exactly the same as RHS indices:
    # TODO: is it a good idea to introduce special cases like this? Is this actually
    # a special case?
    #if toSet(toSeq(idxLhs)) == union(idxRes, idxContr):
    #  # Means that idxRes must be union of those
    #  idxRes = union(idxRes, idxContr)
    #  idxContr = initHashSet[string]()
    #else:
    #  # if we have an assignment, we take the LHS for fact
    #  # Thus `idxContr` then is actually the `symmetricDifference` of the LHS and the
    #  # union of `idxRes` u `idxContr` (i.e. all indices on the RHS)
    #  idxContr = symmetricDifference(
    #    union(idxRes, idxContr),
    #    toSet(toSeq(idxLHS))
    #  )
    #  # `idxRes` thus has to get rid of the indices in `idxContr`
    #  idxRes = difference(idxRes, idxContr)


    #if idxContr.len == 0 and idxLHS.len == 0:
    #  # If both are empty, we have to fix the contraction indices. This means we
    #  # have a scalar assignment, so no IDX on the LHS is correct. However, on the
    #  # RHS we ``must`` only have a single tensor, which is to be fully contracted
    #  # TODO: Theoretically we could allow any number of tensors, meaning a scalar
    #  # on the LHS implies we want to perform full contraction of the /resulting/
    #  # tensor after normal Einstein summation
    #  if true:
    #    echo  rhsStmt.treeRepr
    #  doAssert rhsStmt.kind == nnkBracketExpr, "there must only be a single " &
    #    "tensor on the RHS to do full contraction of the tensor!"
    #  # thus swap `idxContr` and `idxRes`
    #  let tmp = idxRes
    #  idxRes = idxContr
    #  idxContr = tmp

  # now we can safely calculate the rank of the tensor
  let rank = idxRes.card
  echo "rank ", rank, " idx rhs ", idxRes




  # find each axis of each tensor, which will surive
  # seq of tuples of `tensor`, `axis` pairs
  echo "Finding tensors ", idxRes, " from ", tensorIdxPairs
  let tensorAxes = findAxes(idxRes, tensorIdxPairs)
  let contractionAxes = findAxes(idxContr, tensorIdxPairs)

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
    #for i, ax in tensorAxes:
      # since `shapeIdents` corresponds to the shape of the resulting
      # tensor, use the `LHS` (if skAssign) to order them in the
      # correct way
      #doAssert
      echo "LHS ", idxLhs
      #echo "IDX PAIRS ", ax[1]
      echo tensorAxes
      echo tensorIdxPairs
      #echo ax

      # TODO: `i` should not be the index, but rather the
      # index of `idxLhs` for the index that corresponds to
      # ax
      var idxArg: int
      var t: NimNode
      for tIdx, ax in tensorAxes:
        echo "iter i ", i, " ", ax, "tensor ", tIdx, " looking for ", idx
        if $tensorIdxPairs[ax[0]][1][ax[1]] == idx:
          idxArg = ax[1]
          t = tensorIdxPairs[ax[0]][0]
          echo "IDX ARG ", idxArg
          echo "TENSOR ", t
          echo "IDX ", idx
          idxIdentPairs.add (idx, i) #idxArg)
          break
      if t.kind == nnkNilLit:
        echo "DId not find " , idx, " in ", tensorAxes
        echo "tensor idx pairs ", tensorIdxPairs
      #let t = tensorIdxPairs[ax[0]][0]
      #let idx = ax[1]
      if rank > 0:
        # only add to `shapes` variable, if rank > 0
        result.add quote do:
          `shapeIdents`[`i`] = `t`.shape[`idxArg`]
  of skAuto:
    echo "*#*&#*&#&*#*&#&*#*&#&*"
    var i = 0
    for idx in inputOrderIdents:
      if idx in idxRes:
        # since `shapeIdents` corresponds to the shape of the resulting
        # tensor, use the `LHS` (if skAssign) to order them in the
        # correct way
        #doAssert
        echo "LHS ", idxLhs
        #echo "IDX PAIRS ", ax[1]
        echo tensorAxes
        echo tensorIdxPairs
        #echo ax

        # TODO: `i` should not be the index, but rather the
        # index of `idxLhs` for the index that corresponds to
        # ax
        var idxArg: int
        var t: NimNode
        for tIdx, ax in tensorAxes:
          if $tensorIdxPairs[ax[0]][1][ax[1]] == idx:
            idxArg = ax[1]
            t = tensorIdxPairs[ax[0]][0]
            idxIdentPairs.add (idx, i)
            break
        result.add quote do:
          `shapeIdents`[`i`] = `t`.shape[`idxArg`]

        inc i

  var idxIdentContrPairs = newSeq[(string, int)]()
  for i, idx in toSeq(idxContr):
    for tIdx, ax in contractionAxes:
      if $tensorIdxPairs[ax[0]][1][ax[1]] == idx:
        idxIdentContrPairs.add (idx, i)
        break

  # generate the code to get the shape of the contraction
  let shapeContrIdents = ident"shapesContr"
  let rankContr = idxContr.card
  echo "rankContr ", rankContr, " idx contr ", idxContr
  result.add quote do:
    var `shapeContrIdents` = newSeq[int](`rankContr`)
    echo "Rank contr ", `rankContr`
  for i, ax in contractionAxes:
    let t = tensorIdxPairs[ax[0]][0]
    let idx = ax[1]
    result.add quote do:
      `shapeContrIdents`[`i`] = `t`.shape[`idx`]

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
    echo "Accessing ", lhsStmt.treeRepr
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
    echo "Or not?!"
    if rank > 0:
      # generate bracket to acceess element
      asgnTo = nnkBracketExpr.newTree(resIdent)
      # now assign the indices we access by the order in which they appear
      # in the input statement
      for x in inputOrderIdents:
        if x in idxRes:
          asgnTo.add ident(x)
    else:
      # scalar result from implicit call
      asgnTo = resIdent

  echo "Asgn to is ", asgnTo.repr

  #doAssert t1.shape[2] == t2.shape[0]
  #doAssert t1.shape[3] == t2.shape[1]
  #var res: T
  #for k in 0 ..< t1.shape[2]:
  #  for l in 0 ..< t1.shape[3]:
  #     res += t1[i, j, k, l] * t2[k, l]
  let prod = rhsStmt # [0]
  let contrRes = ident"res"

  var contractionLoops: NimNode
  if rankContr > 0:
    let innerStmt = quote do:
      `contrRes` += `prod`

    contractionLoops = newStmtList()
    contractionLoops.add quote do:
      var `contrRes`: float
    contractionLoops.add buildLoops(rankContr,
                                    idxIdentContrPairs,#idxContr,
                                    shapeContrIdents,
                                    innerStmt)
    contractionLoops.add quote do:
      `asgnTo` = `contrRes`
  else:
    # in this case we have no contraction. Use variable for inner
    # stamtent
    contractionLoops = quote do:
      `asgnTo` = `rhsStmt` # we could just assign `stmts`, but this for clarity
  echo "Contr loops ", contractionLoops.repr

  if rank > 0:
    let forLoops = buildLoops(rank,
                              idxIdentPairs,#idxRes,
                              shapeIdents, contractionLoops)
    echo "FOR LOOPS ", forLoops.repr
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

  #echo result.treeRepr
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
