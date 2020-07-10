# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ../autograd/autograd,
  ../private/ast_utils,
  ./dsl_types

proc replaceInputNodes*(self: TopoTable, in_shape: NimNode): NimNode =
  # Args:
  #   - The topology table
  #   - the input shape
  # Returns:
  #   - An AST input shape with "x.out_shape" replaced by the actual x.out_shape
  #     taken from the topology table

  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of nnkDotExpr:
      if eqIdent(node[1], "out_shape"):
        return self[node[0]].out_shape
      else:
        letsGoDeeper()
    of {nnkIdent, nnkSym, nnkEmpty}:
      return node
    of nnkLiterals:
      return node
    else:
      letsGoDeeper()
  result = inspect(in_shape)

macro ctxSubtype*(context: Context): untyped =
  ## Extract the subtype from a Context
  result = replaceSymsByIdents(context.getTypeInst[1])
