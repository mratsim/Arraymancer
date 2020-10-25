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

import
  typetraits, macros,
  ../private/sequninit

# ############################################################
#
#                        Datatypes
#
# ############################################################

# Design tradeoff:
#
# Nim default GC is deferred reference counting, meaning it doesn't deal well with
# cyclic references where A refers to B and B refers to A.
#
# To solve that it must use the dreaded mark-and-sweep, i.e. stop the world
# copy all live objects into a new location, everything that was not copied
# was unused.
#
# Deep learning is already extremely memory-bandwidth intensive.
# We can't use a direct pointer to the object as the GC can move it (https://github.com/mratsim/Arraymancer/pull/329)
# so we need a pointer to the ref the GC is holding to break the ref cycle.
#
# The cost is that using objects through that requires a double pointer indirection,
# so much more cache misses.
# Though it's certainly much less costly than mark and sweep.
#
# This is pending https://github.com/nim-lang/Nim/issues/9974

type
  Context*[TT] = ref object # {.acyclic.}
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
    nodes: seq[Node[TT]]
    no_grad: bool

  Variable*[TT] = ref object # {.acyclic.}
    ## A variable is a wrapper for Tensors that tracks operations applied to it.
    ## It consists of:
    ##    - A weak reference to a record of operations ``context``
    ##    - The tensor being tracked ``value``
    ##    - The gradient of the tensor ``grad``
    ##    - a flag that indicates if gradient is needed
    context*: Context[TT]
      # Variables shouldn't own their Context
    value*: TT
    grad*: TT
    requires_grad*: bool

  Gate*[TT] = ref object of RootObj # {.acyclic.}
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

  Backward*[TT] = proc(self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] {.nimcall.}
    ## ⚠️ Warning: make sure the identifier is not overloaded
    ## https://github.com/nim-lang/Nim/issues/9997

  Node[TT] = object
    ## A node consist of:
    ##   - The description of the operator or layer (``gate``)
    ##   - The corresponding proc that handles backpropagation
    ##   - A weak reference to the ``parents`` VariableObj
    ##   - The actual value of the node (``payload``)
    gate: Gate[TT]
    backward: Backward[TT]
    parents: Parents[TT]
    payload: Payload[TT]
    when defined(debug):
      name: string

  Parents[TT] = seq[Variable[TT]]
    # Children nodes shouldn't own their parents
  SmallDiffs*[TT] = seq[TT]

# ############################################################
#
#                      Debugging
#
# ############################################################

func debug[TT](ctx: Context[TT]) =
  ## Debug the autograd context

  debugecho "\n######"
  for i, node in ctx.nodes:
    # strformat doesn't work in generics
    # var s = &"Node {i:>4}: {node.name:>12} - "
    var s = "Node " & $i & ": " & node.name & " - "
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

func newContext*(TT: typedesc): Context[TT] =
  ## Initialize a context
  new result
  result.nodes = newSeq[Node[TT]]()

proc variable*[TT](ctx: Context[TT], value: TT, requires_grad = false): Variable[TT] =
  ## Wrap a variable to the context
  ## T is a Tensor[T, CudaTensor[T] or scalar T
  new result
  result.context = ctx
  result.value = value
  if requires_grad:
    result.requires_grad = true
    result.grad = value.zeros_like

template len[TT](ctx: Context[TT]): int =
  ## Returns the number of operations applied in the context
  ctx.nodes.len

template push[TT](ctx: Context[TT], node: Node[TT]) =
  ## Append a new operation to the context
  ctx.nodes.add(node)

template peek[TT](ctx: Context[TT]): Node[TT] =
  ctx.nodes[ctx.len - 1]

template pop[TT](ctx: Context[TT]): Node[TT] =
  ctx.nodes.pop

func register_node*[TT](
        name: static string,
        gate: Gate[TT],
        backward: Backward[TT],
        result: Variable[TT] or seq[Variable[TT]],
        parents: varargs[Variable[TT]]) =
  ## Add an operation / gate as a new node in the computation graph
  var node: Node[TT]

  node.gate = gate
  node.backward = backward
  node.parents = @parents

  when result is Variable:
    node.payload = Payload[TT](kind: pkVar, variable: result)
  else:
    node.payload = Payload[TT](kind: pkSeq, sequence: result)

  when defined(debug):
    node.name = name

  parents[0].context.push node

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

  while v.context.len > 0:
    let curNode = v.context.pop
    let diffs = curNode.backward(curNode.gate, curNode.payload)
    for i, diff in diffs:
      let parent_i = curNode.parents[i]
      if parent_i.requires_grad:
        parent_i.grad += diff

func newParents*[TT](num: Natural): Parents[TT] {.inline.} =
  newSeqUninit[Variable[TT]](num)

func newDiffs*[TT](num: Natural): SmallDiffs[TT] {.inline.} =
  newSeqUninit[TT](num)
