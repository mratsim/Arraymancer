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

import typetraits, macros

const MAX_NB_GRADS = 3 # Max number of gradients output of autograd operations

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

  VariablePtr[TT] = ptr VariableObj[TT]
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

  Gate*[TT] = ref object {.inheritable.}
    nb_grads*: int
    ## Base operator or layer. You can describe your custom operations or layers
    ## by inheriting from Gate and add a forward and optionally a backward method.
    ## Each operations should set the number of gradients produced during backpropagation.
    ## Additional fields specific to the operations like weights or inputs cache should be added too.

  Node*[TT] = ref object
    ## A node consist of:
    ##   - The description of the operator or layer (``gate``)
    ##   - A weak reference to the ``parents`` VariableObj
    ##   - The actual value of the node (``payload``)
    gate*: Gate[TT]
    parents*: Parents[TT]
    payload*: Variable[TT]

  Parents[TT] = array[MAX_NB_GRADS, VariablePtr[TT]]
  SmallDiffs*[TT] = array[MAX_NB_GRADS, TT]

# Somehow if you declare forward before backward, you get invalid declaration order
# https://github.com/nim-lang/Nim/issues/5325
method backward*[TT](self: Gate[TT], gradient: TT): SmallDiffs[TT] {.noInit, base, inline.} =
  raise newException(ValueError, "backward method is not implemented for " & $self.type.name)

method forward*[TT](self: Gate[TT], a, b: Variable[TT]): Variable[TT] {.base, inline.} =
  # Binary forward
  raise newException(ValueError, "forward method is not implemented for " & $self.type.name)

method forward*[TT](self: Gate[TT], a: Variable[TT]): Variable[TT] {.base, inline.}=
  # Unary forward
  raise newException(ValueError, "forward method is not implemented for " & $self.type.name)

proc newContext*(TT: typedesc): Context[TT] {.noSideEffect.} =
  ## Initialize a context
  new result
  result.nodes = newSeq[Node[TT]]()

proc variable*[TT](ctx: Context[TT], value: TT, requires_grad = false): Variable[TT] {.noSideEffect.} =
  ## Wrap a variable to the context
  ## T is a Tensor[T, CudaTensor[T] or scalar T
  # TODO make the grad initialization optional to optimize memory use
  new result
  result.context = ctx.weakRef
  result.value = value
  result.grad = value.zeros_like
  result.requires_grad = requires_grad

proc len[TT](ctx: ContextPtr[TT]): int {.noSideEffect, inline.}=
  ## Returns the number of operations applied in the context
  ctx.nodes.len()

proc push*[TT](ctx: ContextPtr[TT], node: Node[TT]) {.noSideEffect, inline.}=
  ## Append a new operation to the context
  if not ctx.no_grad:
    ctx.nodes.add(node)

proc peek[TT](ctx: ContextPtr[TT]): Node[TT] {.noSideEffect, inline.}=
  ctx.nodes[ctx.len - 1]

proc pop[TT](ctx: ContextPtr[TT]): Node[TT] {.noSideEffect, inline.}=
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

proc is_grad_needed*(v: Variable): bool {.noSideEffect, inline.} =
  ## Depending on the input variable and its context no_grad_mode,
  ## returns true if gradient computation is needed and false otherwise
  v.requires_grad and not v.context.no_grad

proc check_ctx*(a, b: Variable) {.noSideEffect, inline.} =
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
  while v.context.len > 0 and v.context.peek.payload != v:
    discard v.context.pop

  # Now, until the context is been all backpropagated through we update
  # each intermediate variables with its accumulated gradient and then pop the node
  # TODO: Count Toward Zero memory optimization:
  # https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation and https://github.com/Rufflewind/revad/blob/de509269fe878bc9d564775abc25c4fa663d8a5e/src/chain.rs

  while v.context.len > 0:
    let curNode = v.context.pop
    let curGate = curNode.gate
    let diffs = curGate.backward(curnode.payload.grad)

    for i in 0 ..< curGate.nb_grads:
      let parent_i = curNode.parents[i]
      if parent_i.requires_grad:
        parent_i.grad += diffs[i]

macro `[]`*[TT](v: Variable[TT], args: varargs[untyped]): Variable[TT] or VariableObj[TT]=
  ## Slice the tensor contained by the dynamic graph Variable
  ## Input:
  ##   - a Variable
  ## Output:
  ##   - a sliced Variable

  if args.len == 0:
    # This is the dereference operator
    result = nnkDerefExpr.newTree(v)
  else:
    result = quote do:
      var v = `v` # Shadow to make sure that if v is an expression it is called only once

      var sliced: type(v)
      new sliced

      sliced.context       = v.context
      sliced.value         = v.value[`args`]
      sliced.grad          = v.grad[`args`]
      sliced.requires_grad = v.requires_grad

      sliced
      # TODO: tests for slicing correspondence

proc weakRef*[TT](v: Variable[TT]): VariablePtr[TT] {.inline.} =
  ## Get a weak/untraced reference to a Variable
  ## This is intended for library writers and Neural Network graphs
  ## to avoid strong cyclic references.
  addr v[]

proc weakRef*[TT](ctx: Context[TT]): ContextPtr[TT] {.inline.} =
  ## Get a weak/untraced reference to a Variable
  ## This is intended for library writers and Neural Network graphs
  ## to avoid strong cyclic references.
  addr ctx[]
