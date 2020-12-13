import macros, sequtils, sets, algorithm
import ../private/ast_utils
import ./shapeshifting
  # Note: importing shapeshifting_cuda will trigger a Nim inference bug
  #       in genContiguous with no workaround

## This module provides Einstein summation for an arbitrary number of tensors.
##
## Einstein summation describes a special application of
## `index notation <https://en.wikipedia.org/wiki/Index_notation>`_
## in which indices that appear more than once are implicitly summed over.
## This allows for a concise notation of many vector / matrix / tensor calculations,
## while exactly representing the required calculation.
##
## In general Einstein summation is a subset of
## `Ricci calculus <https://en.wikipedia.org/wiki/Ricci_calculus>`_.
##
## The implementation of `einsum` in different languages however, typically goes
## above and beyond actual Einstein summation, allowing for many aspects of
## Ricci calculus.
##
## Simple Einstein summation examples
## ==================================
##
## Typical examples include matrix-vector multiplcation, matrix-matrix multiplication
## or the cross product. The examples below use the `einsum` / notation for the
## elements of tensors, namely `m[i,j]` for element `i,j` of the matrix ``m``, instead of
## the more mathematical notation `m_ij`.
##
## Matrix-vector multiplication
## ----------------------------
##
## Let ``m`` be an `NxM` matrix and ``v`` a `M` vector. Then matrix-vector multiplication
## `m * v` is defined as:
## `w[i] = \sum_j m[i,j] * v[j]`.
## The result is an `N` vector ``w`` consisting of elements `w[i]`.
## Since `j` appears twice on the RHS of the equation, Einstein summation implies that
## the sum over `j` is implicit, hence we can write:
##
## `w[i] = m[i,j] * v[j]`.
##
## Matrix-matrix multiplication
## ----------------------------
##
## The same can be applied to matrix-matrix multiplication. Let ``m``, ``n`` be two
## compatible matrices (both `NxN` or `NxM` and `MxN`) with elements `m[i,j]` and
## `n[i,j]`. Matrix-matrix multiplication is defined as
##
## `a[i,k] = \sum_j m[i,j] * n[j,k]`
##
## and thus in Einstein summation:
##
## `a[i,k] = m[i,j] * n[j,k]`.
##
## Cross-product of two vectors
## ----------------------------
##
## The cross product of two 3 vectors ``v``, ``w`` can be conveniently defined using
## the `Levi-Civita symbol <https://en.wikipedia.org/wiki/Levi-Civita_symbol#Three_dimensions>`_
## `\epsilon_{ijk}`:
##
## `a[i] = \epsilon_{ijk} v[j] * w[k]`,
##
## which implies `j` and `k` are summed over, while `i` is kept for the resulting tensor.
##
## More complex examples
## =====================
##
## In this implementation of `einsum` (similar to other `einsum` implementations),
## it's also possible to explicitly keep different dimensions of the multiplied
## tensors or even perform calculations without a single index appearing mutliple
## times, for instance to transpose a tensor. For these cases the explicit form
## of the `einsum` macro has to be used, see below.
##
## Transposition of a matrix
## -------------------------
##
## Transposition of a matrix can be expressed in index notation simply as an
## exchange of indices, namely let ``m`` be an `NxM` matrix, the transposed
## `MxN` matrix ``m^T`` is written as:
##
## `m[j,i] = m[i,j]`.
##
## Hadamard product
## ----------------
##
## The Hadamard product defines the product of two `NxM` matrices ``n``, ``m``
## in which the matrices are multiplied element wise. It is a good example
## of the extension of `einsum` over standard Einstein summation:
##
## `a[i,j] = m[i,j] * n[i,j]`.
##
## Naive Einstein summation would demand a sum over both `i` and `j`, resulting
## in a scalar on the LHS instead of another `NxM` matrix.
##
## Contracting a whole matrix
## --------------------------
##
## Contraction of a full matrix describes summing all elements of a matrix
## ``m``, resulting in a scalar `a`. It is expressed by:
##
## `a = m[i,i]`.
##
## The `einsum` macro
## ==================
##
## The `einsum` macro provides two different usage paradigms.
## * implicit <- normal Einstein summation
## * explicit <- potential extended Einstein summation
##
## The macro takes a `varargs[Tensor]` and a single statement. It
## returns a `Tensor[T]`, where `T` is deduced from the subtype of the
## given tensors, if the result is not a scalar. For a scalar result
## the return value is of type `T`. Note that the type of all given tensors
## must match!
##
## The statement given to the macro is just a single line making use of
## Einstein summation as in all the examples above. As a matter of fact
## all examples above are valid statements for the `einsum` macro!
##
## Of course only tensors, which are given to the macro in the `varargs`
## may be used in the statement.
##
## If only the `RHS` of the examples above are given, the required indices
## for the resulting tensor are automatically calculated using pure Einstein
## summation. Assuming `a`, `b` are two 2D arraymancer tensors , we could
## express their matrix mutliplcation as
##
## .. code:: nim
##    let c = einsum(a, b):
##      a[i,j] * b[j,k]
##
## Of course the same can be written in explicit form:
##
## .. code:: nim
##    let c = einsum(a, b):
##      c[i,k] = a[i,j] * b[j,k]
##
## A few things must be noted here for the explicit case:
## * the indices on the LHS are taken as "the truth"! Any index appearing here
##   will ``not`` be summed over.
## * the order on the LHS is taken into account, allowing for transposing
##   dimensions.
## * the identifier used on the LHS is arbitrary. It can match what the user assigns
##   to, but need not.
##
## For many more examples for typical applications, take a look at the test case
## `<../../tests/tensor/test_einsum.nim>`_.
##
## Implementation details
## ----------------------
##
## The macro calculates, which indices must be contracted and which remain in the
## final tensor. For each appearing index (of either case) we create a for loop,
## while the contracting for loops appear within the non contracting indices.
##
## The macro creates a `block`, in which the code is produced and returns the
## temporary tensor used in it.
##
## It also forces the tensors into contiguous, row major form by creating
## local copies with `asContiguous`.


type
  # enum which stores whether an `einsum` call is explicit `skAssign` (statement
  # contains an nnkAsgn node) or implicit `skAuto` (statement is purely nnkIndix
  # nodes)
  StatementKind = enum
    skAssign, # specific assignment of summation to an existing tensor
    skAuto # automatic deduction of the resulting tensor
  # a helper object, which stores a tensor ident node together with the applied
  # indices
  TensorIdx = object
    t: NimNode # the tensor ident
    idx: seq[NimNode] # the corresponding indices

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
    doAssert stmt[0].eqIdent"*", "It ``must`` be a product `*` " &
      "between the tensors!"
    result = skAuto
  of nnkAsgn:
    result = skAssign
  else:
    error("`einsum` statement must not be of kind `" & $stmt.kind & "`!")

proc getTensorIdx(tensors: NimNode, tensorArgument: seq[NimNode]): seq[TensorIdx] =
  ## Iterate over the `nnkInfix` RHS of the `einsum` statement.
  ## Also compares the tensors in the statement to the tensors the user gave
  ## as the `typed` argument to the macro.
  ## Returns a sequence of TensorIdx objects, i.e. a tensor ident together with the
  ## associated indices.
  proc extractIdentIdx(n: NimNode, compare: NimNode): TensorIdx =
    let tC = n[0]
    doAssert $tC == $compare, " was " & $tC & " and "  & $compare
    let tIdx = slice(n, 1 .. ^1)
    result = TensorIdx(t: tC, idx: tIdx)
  case tensors.kind
  of nnkBracketExpr:
    # only a single tensor, probably in an `skAssign` statement
    doAssert tensorArgument.len == 1, "If only a single tensor is used in the " &
      "statement of `einsum`, only a single argument may be given!"
    result = @[extractIdentIdx(tensors, tensorArgument[0])]
  of nnkInfix:
    doAssert tensors[0].eqIdent"*"
    if tensors[1].kind == nnkInfix:
      result.add getTensorIdx(tensors[1], tensorArgument)
      result.add getTensorIdx(tensors[2], @[tensorArgument[^1]])
    else:
      result.add getTensorIdx(tensors[1], @[tensorArgument[0]])
      result.add getTensorIdx(tensors[2], @[tensorArgument[1]])
  else:
    error("Unsupported kind " & $tensors.kind)

proc findIdx(tensorSeq: seq[TensorIdx], idx: string): (NimNode, int) =
  ## returns a tensor ident NimNode and the corresponding axis index that
  ## the string index `idx` corresponds to
  for tIdx in tensorSeq:
    let idxStr = tIdx.idx.mapIt($it)
    let resIdx = find(idxStr, idx)
    if resIdx >= 0:
      result = (tIdx.t, resIdx)

proc toDuplicates[T](s: seq[T]): OrderedSet[T] =
  ## creates a set of elements ``only`` consisting of the duplicate elements
  ## in `s`. Unique elements will ``not`` show up in the resulting set.
  var tmp = initHashSet[T]()
  for x in s:
    if x notin tmp:
      tmp.incl x
    else:
      # already in `tmp`, so it's a duplicate. Add it to result
      result.incl x

proc toUnique[T](s: seq[T]): OrderedSet[T] =
  ## creates a set of elements, which ``only`` contains the unique
  ## elements of `s`. Any duplicate will ``not`` appear in the resulting
  ## set.
  let duplicates = toDuplicates(s)
  for x in s:
    if x notin duplicates:
      result.incl x

proc union[T](s1, s2: OrderedSet[T]): OrderedSet[T] =
  ## returns the union of two OrderedSets. The first arguments order
  ## will be the dominant order. We iterate both and incl each element.
  for x in s1:
    result.incl x
  for x in s2:
    result.incl x

proc makeContigIdent(x: NimNode): NimNode =
  doAssert x.kind == nnkSym or x.kind == nnkIdent
  result = ident(x.strVal & "Cont")

proc replaceRhsByContig(rhs: NimNode): NimNode =
  ## Replaces the tensor identifiers of the RHS statement by those
  ## of the local contiguous tensors.
  result = rhs
  case result.kind
  of nnkInfix:
    for i in 0 ..< result.len:
      case result[i].kind
      of nnkIdent: discard
      of nnkInfix:
        result[i] = replaceRhsByContig(result[i])
      of nnkBracketExpr:
        result[i][0] = makeContigIdent(result[i][0])
      else:
        error("Unsupported kind for `einsum` RHS statement " & $result[i].kind)
  of nnkBracketExpr:
    result[0] = makeContigIdent(result[0])
  else:
    error("Unsupported kind for `einsum` RHS statement " & $result.kind)

proc splitLhsRhs(stmtKind: StatementKind,
                 stmt: NimNode): (NimNode, OrderedSet[string], NimNode) =
  ## Returns the einsum statement of the LHS, the LHS indices in an ordered set
  ## and the RHS statements. If `stmtKind` is `skAuto` however, `lhsStmt` will
  ## be a nnkNilLit and the OrderedSet the empty set.
  ## In addition the `rhsStmt` tensor identifiers will be replaced by the
  ## local contiguous tensors (identifier & "Cont").
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
  # now patch `rhsStmt` to use the local contiguous tensors
  rhsStmt = replaceRhsByContig(rhsStmt)
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

proc genShapes(idxIdentPairs: var seq[(string, int)],
               idxSet: OrderedSet[string],
               shapeIdent: NimNode,
               tensorSeq: seq[TensorIdx]): NimNode =
  ## Generates the tensor shape assignment statements, to assign the
  ## correct tensor axis dimensions to the shape and contraction shape
  ## sequences.
  ## Also fills  the `idxIdentPairs` sequence, which maps the Einstein
  ## index identifier to the correct axis to generate the for loops later.
  result = newStmtList()
  for i, idx in idxSet:
    let (t, idxArg) = findIdx(tensorSeq, idx)
    idxIdentPairs.add (idx, i)
    result.add quote do:
      `shapeIdent`[`i`] = `t`.shape[`idxArg`]
  # Reverse the `idxIdentPairs` so that the inner most loops
  # cover the right most indices
  idxIdentPairs.reverse

proc genAssignTo(resIdent: NimNode,
                 stmtKind: StatementKind,
                 lhsStmt: NimNode,
                 idxRes: OrderedSet[string]): NimNode =
  ## generates the correct assignment for the `resIdent` variable (the temporary
  ## result variable) based on the `stmtKind` (assign / auto), the potential
  ## `lhsStmt` and the indices of the resulting tensor `idxRes`.
  ## Either:
  ## `tmp` <- our `resIdent` for a scalar result
  ## `tmp[i,j,...]` <- our `resIdent` for a tensor result. Indices in `[]` those
  ##                   of `idxRes` or `lhsStmt` depending on `stmtKind`
  case stmtKind
  of skAssign:
    result = copyNimTree(lhsStmt)
    # replace the identifier, use the `tmp` instead of user defined LHS ident
    case result.kind
    of nnkIdent:
      # case of scalar result, use `resIdent` as total result
      result = resIdent
    of nnkBracketExpr:
      # result is a tensor, replace identifier before `[]`
      result[0] = resIdent
    else:
      error("Unsupported kind for assignment " & $result.kind)
  else:
    if idxRes.card > 0:
      # generate bracket to access element
      result = nnkBracketExpr.newTree(resIdent)
      # now assign the indices we access by the order in which they appear
      # in the input statement
      for idx in idxRes:
        result.add ident(idx)
    else:
      # scalar result from implicit call
      result = resIdent

proc genResContrIndices(
  stmtKind: StatementKind,
  tensorSeq: seq[TensorIdx],
  idxLhs: OrderedSet[string]): (OrderedSet[string], OrderedSet[string]) =
  ## generates the OrderedSets for the indices of the contraction and result
  ## indices based on the `stmtKind` and all indices on the RHS and LHS
  # extract all indices from `tensorSeq`
  let idxAllSeq = concat(tensorSeq.mapIt(it.idx)).mapIt($it)
  # starting point for result indices: all unique indices
  var idxRes = toUnique(idxAllSeq)
  # starting point for contraction indices: all duplicate indices
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
  result = (idxRes, idxContr)

macro typeName(x: typed): untyped =
  let str = x.getTypeInst[1].repr
  result = quote do:
    `str`

proc extractType(ts: seq[NimNode]): (NimNode, NimNode) =
  ## first of all checks whether all tensors in `ts` have the same
  ## data type. If so, returns the type. If not stops compilation.
  proc genSubTypeNode(t: NimNode): NimNode =
    result = quote do:
      getSubType(type(`t`))

  let t0 = ts[0]
  # get string value for error message
  let t0IdentStr = t0.strVal
  let t0Ident = genSym(nskType, "T0Type")
  let t0SubType = genSubTypeNode(t0)
  # res will contain the `when` statement plus type declaration
  var res = newStmtList()
  res.add quote do:
    type `t0Ident` = `t0SubType`
  var whenStmt = nnkWhenStmt.newTree()
  for t in ts:
    # string value for error message
    let tIdentStr = t.strVal
    let subType = genSubTypeNode(t)
    var elifBranch = nnkElifBranch.newTree()
    elifBranch.add quote do:
      `t0Ident` isnot `subType`
    elifBranch.add quote do:
      {.error: "All tensors must be of the same type! " & $`t0IdentStr` & " is of " &
        "type " & $typeName(`t0SubType`) & " while " & $`tIdentStr` & " is of type " &
        $typeName(`subType`) & "!".}
    whenStmt.add elifBranch
  res.add whenStmt
  result = (t0Ident, res)

proc genContiguous(ts: seq[NimNode], subType: NimNode): (seq[NimNode], NimNode) =
  var res = newStmtList()
  var tsCont: seq[NimNode]
  for t in ts:
    let tCIdent = makeContigIdent(t)
    res.add quote do:
      # TODO: Nim inference bug that require the subtype
      let `tcIdent` = asContiguous[`subType`](`t`, layout = rowMajor, force = true)
    tsCont.add tcIdent
  result = (tsCont, res)
  # echo res.treeRepr

macro einsum*(tensors: varargs[typed], stmt: untyped): untyped =
  ## Performs Einstein summation of the given `tensors` defined by the `stmt`.
  ## See the top of the module for an explanation on Einstein summation.
  ##
  ## Let `a`, `b` some 2D tensors (matrices), then the usage to perform
  ## matrix multiplication of the two might look like:
  ##    .. code:: nim
  ##       # implicit Einstein summation
  ##       let c = einsum(a, b):
  ##         a[i,j] * b[j,k]
  ##       # explicit Einstein summation. Note that identifier `d` in statement
  ##       # is arbitrary and need not match what will be assigned to.
  ##       let d = einsum(a, b):
  ##         d[i,k] = a[i,j] * b[j,k] # explicit Einstein summation
  doAssert stmt.len == 1, "There may only be a single statement in `einsum`!"
  result = newStmtList()
  # extract all tensors by checking if they are all symbols
  let tsRaw = getTensors(tensors)
  # generate the type check code and extract the subtype of all tensors
  let (typeIdent, typeGen) = extractType(tsRaw)
  result.add typeGen

  # create contiguous, row ordered versions of the tensors
  let (ts, contiguousTensors) = genContiguous(tsRaw, typeIdent)
  result.add contiguousTensors

  # determine what kind of statement is given, e.g.
  # skAssign: res[i,j] = a[i,j] * b[i,j]
  # skAuto: a[i,j] * b[i,j]
  let stmtKind = checkStatement(stmt)
  # get LHS, RHS statements, possible LHS indices
  let (lhsStmt, idxLhs, rhsStmt) = splitLhsRhs(stmtKind, stmt)

  let tensorIdxSeq = getTensorIdx(rhsStmt, ts)
  # add shape assertion statements
  result.add shapeAssertions(tensorIdxSeq)

  # use to create sets of resulting and contracting indices
  let (idxRes, idxContr) = genResContrIndices(stmtKind, tensorIdxSeq, idxLhs)
  # now we can safely calculate the rank of the tensor
  let rank = idxRes.card

  # the sequence storing the NimNode for the `i`, `j`,... einstein index
  # and corresponding it to the correct index for the `shape*Idents` sequence
  var idxIdentPairs = newSeq[(string, int)]()
  # generate the code to get the shape of the resulting tensor
  let shapeIdents = genSym(nskVar, "shapes")
  if rank > 0:
    # add a `shapes` variable, only if the resulting shape is
    result.add quote do:
      var `shapeIdents` = newSeq[int](`rank`)
  case stmtKind
  of skAssign:
    result.add genShapes(idxIdentPairs,
                         idxLhs,
                         shapeIdents,
                         tensorIdxSeq)
  of skAuto:
    result.add genShapes(idxIdentPairs,
                         idxRes,
                         shapeIdents,
                         tensorIdxSeq)

  var idxIdentContrPairs = newSeq[(string, int)]()
  # generate the code to get the shape of the contraction
  let shapeContrIdents = genSym(nskVar, "shapesContr")
  let rankContr = idxContr.card
  if rankContr > 0:
    result.add quote do:
      var `shapeContrIdents` = newSeq[int](`rankContr`)
    result.add genShapes(idxIdentContrPairs,
                         idxContr,
                         shapeContrIdents,
                         tensorIdxSeq)

  # identifier for the variable storing the temporary result (tensor / scalar),
  # which will be the result of the macro's `block`
  let resIdent = genSym(nskVar, "tmp")
  # generate the result tensor
  if rank == 0:
    result.add quote do:
      var `resIdent` = `typeIdent`(0)
  else:
    result.add quote do:
      var `resIdent` = newTensor[`typeIdent`](`shapeIdents`)

  # generate the LHS of the variable assignment after contraction, e.g.
  # `tmp[i, j]`
  let asgnTo = genAssignTo(resIdent, stmtKind, lhsStmt, idxRes)

  # now build the for loops. Starting with the inner loops performing the
  # tensor contraction
  let contrRes = genSym(nskVar, "res")
  var contractionLoops: NimNode
  if rankContr > 0:
    let innerStmt = quote do:
      `contrRes` += `rhsStmt`

    contractionLoops = newStmtList()
    contractionLoops.add quote do:
      var `contrRes`: `typeIdent`
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
  # then build the outer non contracting for loops, using the `contractionLoops`
  # as the inner loop body
  if rank > 0:
    let forLoops = buildLoops(rank,
                              idxIdentPairs,
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
  # echo result.repr

when isMainModule:
  import
    ./data_structure, ./init_cpu, ./ufunc,
    ./accessors_macros_read, ./accessors_macros_write

  let c0 = toSeq(11..34).toTensor.astype(float)
  let d0 = toSeq(1..6).toTensor.astype(float)
  let c = c0.reshape(2, 2, 3, 2)
  let d = d0.reshape(3, 2)

  echo c
  echo d

  let t = einsum(c, d):
    c[i,j,k,l] * d[k,l]

  echo t.shape
  echo t
