# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import macros

proc pop*(tree: var NimNode): NimNode =
  ## varargs[untyped] consumes all arguments so the actual value should be popped
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

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
