# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typetraits, macros, strformat

# ############################################################
#
#                        Datatypes
#
# ############################################################

type
  Context*[TT] = ref ContextObj[TT]
    ## An autograd context is a record of operations or layers.
    ## It holds the following fields:
    ##   - ``nodes``: This records the list of operations(``Node``) applied in the context
    ##   - ``no_grad``: This disable tracing the list of operations altogether.
    ##     This is useful to save memory when you don't need the gradient
    ##     (for validation or prediction for example)
    ##
    ## A context is also called a tape or a Wengert list.
    ##
    ## Note: backpropagation empties the list of operations.

  ContextPtr[TT] = ptr ContextObj[TT]
    ## A ``ContextPtr`` is a weak reference to a ``ContextObj``.
    ## A ``Context`` refers to ``Nodes`` which refers to ``Variable``
    ## which will weakly refer back to a ``Context``.
    ## This avoids strong circular references with the associated garbage collection costs.

  ContextObj[TT] = object
    nodes: seq[Node[TT]]
    no_grad: bool

  Variable*[TT] = ref VariableObj[TT]
    ## A variable is a wrapper for Tensors that tracks operations applied to it.
    ## It consists of:
    ##    - A weak reference to a record of operations ``context``
    ##    - The tensor being tracked ``value``
    ##    - The gradient of the tensor ``grad``
    ##
    ## Warning  ⚠: Make sure the ``Context`` outlives the ``Variable``.
    ## In the future ``grad`` will be optional: ``Option[TT]`` or ``opt[TT]``

  VariablePtr*[TT] = ptr VariableObj[TT]
    ## A ``VariablePtr`` is almost the same as a ``Variable``
    ## except that it is not traced by the garbage collector.
    ##
    ## It is an optimization to break from the following cyclic reference:
    ## parent --(ref)--> child
    ## child  --(ref)--> parent
    ## A naive (but fast) garbage collector cannot delete child because parent refers to it
    ## nor can it delete parent because child refers to it.

  VariableObj[TT] = object
    context*: ContextPtr[TT]
    value*: TT
    grad*: TT
    requires_grad*: bool
    # TODO make the grad initialization optional to optimize memory use

  Gate*[TT] = ref object of RootObj
    ## Base operator or layer. You can describe your custom operations or layers
    ## by inheriting from Gate and add a forward and optionally a backward method.
    ## Each operations should set the number of gradients produced during backpropagation.
    ## Additional fields specific to the operations like weights or inputs cache should be added too.

  PayloadKind* = enum
    pkVar, pkSeq
  Payload*[TT] = object
    case kind*: PayloadKind
    of pkVar: variable*: Variable[TT]
    of pkSeq: sequence*: seq[Variable[TT]]

  Node*[TT] = ref object
    ## A node consist of:
    ##   - The description of the operator or layer (``gate``)
    ##   - A weak reference to the ``parents`` VariableObj
    ##   - The actual value of the node (``payload``)
    gate*: Gate[TT]
    parents*: Parents[TT]
    payload*: Payload[TT]

  Parents[TT] = seq[VariablePtr[TT]]
  SmallDiffs*[TT] = seq[TT]

# ############################################################
#
#                      Debugging
#
# ############################################################

# TODO: don't export that publicly

method debugGateName*[TT](self: Gate[TT]): string {.noInit, base, inline.} =
  raise newException(ValueError, "debugGateName method is not implemented for one of the gates. (And obviously we can't print its name)")

func debugContext(ctx: Context or ContextPtr) =
  ## Debug the autograd context

  debugecho "\n######"
  for i, node in ctx.nodes:
    var s = &"Node {i:>4}: {debugGateName(node.gate):>25} - "
    if node.parents.len <= 1:
      s &= $node.parents[0].value.shape
    else:
      s &= '('
      for p, parent in node.parents:
        if p != 0:
          s &= ", "
        s &= $parent.value.shape
      s &= ')'
    s &= " ===>> "
    case node.payload.kind
    of pkVar: s &= $node.payload.variable.value.shape
    of pkSeq:
      s &= "( "
      for p, payload in node.payload.sequence:
        if p != 0:
          s &= ", "
        s &= $payload.value.shape
      s &= ")"
    debugecho s

# ############################################################
#
#                Autograd procedure
#
# ############################################################

method backward*[TT](self: Gate[TT], payload: Payload[TT]): seq[TT] {.noInit, base, inline.} =
  raise newException(ValueError, "backward method is not implemented for " & $self.type.name)

func newContext*(TT: typedesc): Context[TT] =
  ## Initialize a context
  new result
  result.nodes = newSeq[Node[TT]]()

func variable*[TT](ctx: Context[TT], value: TT, requires_grad = false): Variable[TT] =
  ## Wrap a variable to the context
  ## T is a Tensor[T, CudaTensor[T] or scalar T
  # TODO make the grad initialization optional to optimize memory use
  new result
  result.context = ctx.weakRef
  result.value = value
  result.grad = value.zeros_like
  result.requires_grad = requires_grad

func len[TT](ctx: ContextPtr[TT]): int {.inline.}=
  ## Returns the number of operations applied in the context
  ctx.nodes.len

func push*[TT](ctx: ContextPtr[TT], node: Node[TT]) {.inline.}=
  ## Append a new operation to the context
  if not ctx.no_grad:
    ctx.nodes.add(node)

func peek[TT](ctx: ContextPtr[TT]): Node[TT] {.inline.}=
  ctx.nodes[ctx.len - 1]

func pop[TT](ctx: ContextPtr[TT]): Node[TT] {.inline.}=
  ctx.nodes.pop

template no_grad_mode*(ctx: Context, body: untyped): untyped =
  ## Within this block, the context will not track the operations applied
  ## to each Variable.
  ##
  ## This should be used for validation or prediction to optimize memory.
  let prev_state = ctx.no_grad
  ctx.no_grad = true

  body

  ctx.no_grad = prev_state

func is_grad_needed*(v: Variable): bool {.inline.} =
  ## Depending on the input variable and its context no_grad_mode,
  ## returns true if gradient computation is needed and false otherwise
  v.requires_grad and not v.context.no_grad

func check_ctx*(a, b: Variable) {.inline.} =
  if unlikely(a.context != b.context):
    raise newException(ValueError, "You cannot combine variable from different contexts")

proc backprop*[TT](v: Variable[TT]) =
  ## Differentiate the chain of operations w.r.t to this variable.
  ## Context will be reset

  if unlikely(not v.requires_grad):
    raise newException(ValueError, "Operations leading to this variable were not fully traced.\nDid you forget to set a `requires_grad` or to disable the context `no_grad_mode`?")

  # We initialize the Variable we want to backpropagate on with a Tensor of ones.
  # TODO, restrict to scalar backprop?
  v.grad = v.value.ones_like

  # We pop the context until we find the gate that produced our Variable
  while v.context.len > 0 and v.context.peek.payload.variable != v:
    discard v.context.pop

  v.context.debugContext

  # Now, until the context is been all backpropagated through we update
  # each intermediate variables with its accumulated gradient and then pop the node
  # TODO: Count Toward Zero memory optimization:
  # https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation and https://github.com/Rufflewind/revad/blob/de509269fe878bc9d564775abc25c4fa663d8a5e/src/chain.rs

  while v.context.len > 0:
    let curNode = v.context.pop
    let curGate = curNode.gate
    let diffs = curGate.backward(curnode.payload)

    echo debugGateName(curGate)

    for i, diff in diffs:
      let parent_i = curNode.parents[i]
      echo &"Parent {i} shape: {$parent_i.value.shape}, gradient shape = {$parent_i.value.shape}, diff shape: {$diff.shape}"
      if parent_i.requires_grad:
        parent_i.grad += diff

func weakRef*[TT](v: Variable[TT]): VariablePtr[TT] {.inline.} =
  ## Get a weak/untraced reference to a Variable
  ## This is intended for library writers and Neural Network graphs
  ## to avoid strong cyclic references.
  cast[VariablePtr[TT]](v)

func weakRef*[TT](ctx: Context[TT]): ContextPtr[TT] {.inline.} =
  ## Get a weak/untraced reference to a Variable
  ## This is intended for library writers and Neural Network graphs
  ## to avoid strong cyclic references.
  cast[ContextPtr[TT]](ctx)
