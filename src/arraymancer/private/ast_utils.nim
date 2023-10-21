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

# Tools to manipulate Nim Abstract Syntax Tree

import macros

proc hasType*(x: NimNode, t: static[string]): bool {. compileTime .} =
  ## Compile-time type checking
  sameType(x, bindSym(t))

proc isInt*(x: NimNode): bool {. compileTime .} =
  ## Compile-time type checking
  hasType(x, "int")

proc isBool*(x: NimNode): bool {. compileTime .} =
  ## Compile-time type checking
  hasType(x, "bool")

proc isOpenArray*(x: NimNode): bool {. compileTime .} =
  ## Compile-time type checking
  doAssert false, "This is broken for generics https://github.com/nim-lang/Nim/issues/14021"
  hasType(x, "array") or hasType(x, "seq") or hasType(x, "openArray")

proc isAllInt*(slice_args: NimNode): bool {. compileTime .} =
  ## Compile-time type checking
  result = true
  for child in slice_args:
    # We don't use early return here as everything is evaluated at compile-time,
    # has no run-time impact and there are very few slice_args
    result = result and isInt(child)

proc pop*(tree: var NimNode): NimNode {. compileTime .}=
  ## varargs[untyped] consumes all arguments so the actual value should be popped
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

macro getSubType*(TT: typedesc): untyped =
  # Get the subtype T of an AnyTensor[T] input
  getTypeInst(TT)[1][1]

template letsGoDeeper* =
  var rTree = node.kind.newTree()
  for child in node:
    rTree.add inspect(child)
  return rTree

proc replaceSymsByIdents*(ast: NimNode): NimNode =
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      return ident($node)
    of nnkEmpty:
      return node
    of nnkLiterals:
      return node
    of nnkHiddenStdConv: # see `test_fancy_indexing.nim` why needed
      expectKind(node[1], nnkSym)
      return ident($node[1])
    else:
      letsGoDeeper()
  result = inspect(ast)

proc replaceNodes*(ast: NimNode, replacements: NimNode, to_replace: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - an array of replacement value
  #   - an array of identifiers to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      for i, c in to_replace:
        if node.eqIdent($c):
          return replacements[i]
      return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)
