
const MAX_ARITY = 2 # Max arity/number of input of autograd operations

type
  Gate*[TT] = ref object {.inheritable.}
    # Base operator or layer
    # Inherit from it and add a forward and backward method.
    # Additional fields like weights, cache for bprop should be added too.

  Node[TT] = ref NodeObj[TT]
  ParentsArray[TT] = array[MAX_ARITY, Node[TT]]
  SmallDiffArray[TT] = array[MAX_ARITY, TT]

  NodeObj {.acyclic.} [TT] = object
    # Store an operator/layer + its parent
    gate: Gate[TT]
    parents: ParentsArray[TT]

  Context*[TT] = ref object
    ## Tape / Wengert list. Contains the list of applied operations or layers
    nodes: seq[Node[TT]]

  Variable*[TT] = object
    ## Wrapper for values
    tape: Context[TT]
    parent: Node[TT] # Absence of parents will be represented by the null value. TODO: Option type with no overhead: https://forum.nim-lang.org/t/3082
    value: TT # TT should be a Tensor[T] or CudaTensor[T] or a scalar
    index: int

  Grad[TT] = object
    ## Wrapper for the list of gradients with regards to each inputs
    derivs: seq[TT]

# Somehow if you declare forward before backward, you get invalid declaration order
# https://github.com/nim-lang/Nim/issues/5325
method backward*[TT](self: Gate[TT], gradient: TT): SmallDiffArray[TT] {.base.} =
  raise newException(ValueError, "backward method is not implemented for " & $self.type.name)

method forward*[TT](self: Gate[TT], a, b: Variable[TT]): Variable[TT] {.base.} =
  raise newException(ValueError, "forward method is not implemented for " & $self.type.name)

proc newContext*(TT: typedesc): Context[TT] {.noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  new result
  result.nodes = newSeq[Node[TT]]()

proc variable*[TT](ctx: Context[TT], value: TT): Variable[TT] {.noSideEffect.} =
  ## Wrap a variable to the context
  ## T is a Tensour[T, CudaTensor[T] or scalar T
  return Variable[TT](tape: ctx, parent: nil, value: value)

proc len[TT](t: Context[TT]): int =
  ## Returns the number of operations applied in the context
  t.nodes.len()

template push[TT](t: Context[TT], node: Node[TT]) =
  ## Append a new operation to the context
  t.nodes.add(node) #Appending in Nim is add not push

template value*[TT](v: Variable[TT]): TT  =
  ## Unwrap the value from its context
  v.value

proc grad*[TT](v: Variable[TT]): Grad[TT] =
  ## Compute the gradients
  # Computation is done with gradient set to 1 for the final output value
  let len = v.tape.len
  let nodes = addr v.tape.nodes

  let T: typedesc = getSubType(TT)

  result.derivs = newSeqWith[TT](zeros[T](1,1))

  result.derivs[v.index] = ones[T](1) #by default 1 Tensor

  #### todo ...