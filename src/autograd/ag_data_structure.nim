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

import typetraits

const MAX_NB_GRADS = 3 # Max number of gradients output of autograd operations

type
  Context*[TT] = ref object
    ## An autograd context is a record of operations or layers.
    ## Note: backpropagation empties the list of operations.
    ##
    ## A context is also called a tape or a Wengert list.
    nodes: seq[Node[TT]]

  Variable*[TT] = ref VariableObj[TT]
    ## A variable is a wrapper for Tensors that tracks operations applied to it.
    ## It consists of:
    ##    - An autograd context ``tape``
    ##    - The tensor being tracked ``value``
    ##    - The gradient of the tensor ``grad``
  VariablePtr*[TT] = ptr VariableObj[TT]
    ## A ``VariablePtr`` is almost the same as a ``Variable``
    ## except that it is not traced by the garbage collector.
    ##
    ## It is an optimization to break from the following cyclic reference:
    ## parent --(ref)--> child
    ## child  --(ref)--> parent
    ## A naive (but fast) garbage collector cannot delete child because parent refers to it
    ## nor can it delete parent because child refers to it.


  VariableObj {.acyclic.} [TT] = object {.acyclic.}
    tape*: Context[TT]
    value*: TT
    grad*: TT
    # TODO make the grad initialization optional to optimize memory use


  Gate*[TT] = ref object {.inheritable.}
    nb_grads*: int
    ## Base operator or layer. You can describe your custom operations or layers
    ## by inheriting from Gate and add a forward and optionally a backward method.
    ## Each operations should set the number of gradients produced during backpropagation.
    ## Additional fields specific to the operations like weights or inputs cache should be added too.

  Node*[TT] = ref object {.acyclic.}
    ## A node consist of:
    ##   - The description of the operator or layer (``gate``)
    ##   - A weak reference to the ``parents`` VariableObj
    ##   - The actual value of the node (``payload``)
    gate*: Gate[TT]
    parents*: Parents[TT]
    payload*: Variable[TT]

  Parents*[TT] = array[MAX_NB_GRADS, VariablePtr[TT]]
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

proc newContext*(TT: typedesc): Context[TT] {.inline, noSideEffect.} =
  ## Initialize a context (Tape / Wengert list)
  new result
  result.nodes = newSeq[Node[TT]]()

proc variable*[TT](ctx: Context[TT], value: TT): Variable[TT] {.inline, noSideEffect.} =
  ## Wrap a variable to the context
  ## T is a Tensour[T, CudaTensor[T] or scalar T
  # TODO make the grad initialization optional to optimize memory use
  return Variable[TT](tape: ctx, value: value, grad: value.zeros_like)

proc weakRef*[TT](v: Variable[TT]): VariablePtr[TT] {.inline.} =
  ## Get a weak/untraced reference to a Variable
  ## This is intended for library writer and Neural Network graphs
  ## to avoid strong cyclic references between parent operations/Variables and node.payload.
  cast[VariablePtr[TT]](v)

proc len[TT](ctx: Context[TT]): int {.inline.}=
  ## Returns the number of operations applied in the context
  ctx.nodes.len()

proc push*[TT](ctx: Context[TT], node: Node[TT]) {.inline.}= #TODO: how not to export that
  ## Append a new operation to the context
  ctx.nodes.add(node) #Appending in Nim is add not push

proc peek[TT](ctx: Context[TT]): Node[TT] {.inline.}=
  ctx.nodes[ctx.len - 1]

proc check_ctx*(a, b: Variable) {.noSideEffect, inline.} =
  if unlikely(a.tape != b.tape): # Compare pointer address directly
    raise newException(ValueError, "You cannot combine variable from different contexts")

proc backprop*[TT](v: Variable[TT]) =
  ## Differentiate the chain of operations w.r.t to this variable.
  ## Context will be reset

  # We initialize the Variable we want to backpropagate on with a Tensor of ones.
  # TODO, restrict to scalar backprop?
  v.grad = v.value.ones_like

  # We pop the context until we find the gate that produced our Variable
  while v.tape.len > 0 and v.tape.peek.payload != v:
    discard v.tape.nodes.pop

  # Now, until the context is been all backpropagated through we update
  # each intermediate variables with its accumulated gradient and then pop the node
  # TODO: Count Toward Zero memory optimization:
  # https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation and https://github.com/Rufflewind/revad/blob/de509269fe878bc9d564775abc25c4fa663d8a5e/src/chain.rs

  while v.tape.len > 0:
    let curNode = v.tape.nodes.pop
    let curVar = curnode.payload

    let diffs = curNode.gate.backward(curVar.grad)

    for i in 0 ..< curNode.gate.nb_grads:
      curNode.parents[i].grad += diffs[i]