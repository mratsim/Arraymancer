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

const MAX_ARITY = 3 # Max arity/number of input of autograd operations

type
  Gate*[TT] = ref object {.inheritable.}
    arity*: int
    # Base operator or layer
    # Inherit from it and add a forward and backward method.
    # Each operations should set its arity (number of input)
    # Additional fields like weights, cache for bprop should be added too.

  Parents*[TT] = array[MAX_ARITY, VariablePtr[TT]] # Children have a weak (non-GC) reference to their parents operands
  SmallDiffs*[TT] = array[MAX_ARITY, TT]           #TODO: how not to export that

  Node*[TT] = ref object
    # Store an operator/layer + its parent
    gate*: Gate[TT] #TODO: how not to export that
    parents*: Parents[TT] #TODO: how not to export that
    child*: Variable[TT] # Todo: avoid reference to child and add {.acyclic.}

  Context*[TT] = ref object
    ## Tape / Wengert list. Contains the list of applied operations or layers
    nodes: seq[Node[TT]]

  ## Considerations
  ## A variable can be used in 2 different computations, in that case both gate will point to it

  Variable*[TT] = ref VariableObj[TT]
    ## Wrapper for values
  VariablePtr*[TT] = ptr VariableObj[TT]
    ## Weak reference to a VariableObj

  VariableObj[TT] = object
    tape*: Context[TT] #TODO: how not to export that
    value*: TT # TT should be a Tensor[T] or CudaTensor[T] or a scalar
    grad*: TT # gradient wrt to the last back propagation done
    # TODO make the grad initialization optional to optimize memory use


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
  ## to avoid strong cyclic references between parent operations/Variables and child.
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
  while v.tape.len > 0 and v.tape.peek.child != v:
    discard v.tape.nodes.pop

  # Now, until the context is been all backpropagated through we update
  # each intermediate variables with its accumulated gradient and then pop the node
  # TODO: Count Toward Zero memory optimization:
  # https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation and https://github.com/Rufflewind/revad/blob/de509269fe878bc9d564775abc25c4fa663d8a5e/src/chain.rs

  while v.tape.len > 0:
    let curNode = v.tape.nodes.pop
    let curVar = curNode.child

    let diffs = curNode.gate.backward(curVar.grad)

    for i in 0 ..< curNode.gate.arity:
      curNode.parents[i].grad += diffs[i]