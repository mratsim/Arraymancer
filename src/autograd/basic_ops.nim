import macros


proc check_ctx[TT](a, b: Variable[TT]) =
  if a.tape != b.tape:
    raise newException(ValueError, "You cannot combine variable from different contexts")


type AddGate* {.final.} [TT] = ref object of Gate[TT]
  a_shape*: seq[int]
  b_shape*: seq[int]

method forward*[TT](self: AddGate[TT], a, b: Variable[TT]): Variable[TT] {.inline.}=
  Variable[TT](tape: a.tape, value: a.value + b.value)

method backward*[TT](self: AddGate[TT], gradient: TT): SmallDiffs[TT] =
  result[0] = ones[getSubType(TT)](self.a_shape)
  result[1] = ones[getSubType(TT)](self.b_shape)

proc `+`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Gate
  var gate: AddGate[TT]
  new gate
  gate.arity = 2
  gate.a_shape = a.value.shape
  gate.b_shape = b.value.shape

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a
  node.parents[1] = b

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a, b)
  result.ancestor = node
  node.child = result
