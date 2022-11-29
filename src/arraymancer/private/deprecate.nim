# Copyright 2017-2020 Mamy André-Ratsimbazafy
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

import macros, ./ast_utils

# Handle deprecation and replacement
# --------------------------------------------------------------------------------------

proc overloadSingleSym(oldName, replacement: NimNode, exported: bool): NimNode =
  let impl = replacement.getImpl()
  impl.expectKind {nnkProcDef, nnkFuncDef}

  impl[3].expectKind nnkFormalParams
  let cleanedParams = impl[3].replaceSymsByIdents()

  var body = newCall(replacement)

  for paramIdx in 1 ..< cleanedParams.len:
    for identIdx in 0 ..< cleanedParams[paramIdx].len - 2:
      body.add cleanedParams[paramIdx][identIdx]

  impl[4].expectKind({nnkEmpty, nnkPragma})
  var pragmas = impl[4]
  if pragmas.kind == nnkEmpty:
    pragmas = nnkPragma.newTree()

  # pragmas.add ident"inline"
  # pragmas.add ident"noinit"
  pragmas.add nnkExprColonExpr.newTree(
    ident"deprecated",
    newLit("Use `" & $replacement & "` instead")
  )

  # Generic params
  var generics = newEmptyNode()
  if impl[2].kind != nnkEmpty:
    generics = nnkGenericParams.newTree()
    for genParam in impl[2]:
      if genParam.kind == nnkSym: # Bound symbol
        generics.add nnkIdentDefs.newTree(
          genParam.replaceSymsByIdents(),
          newEmptyNode(),
          newEmptyNode()
        )
      else: # unbound - can this happen?
        genParam.expectKind(nnkIdentDefs)

  let name = if exported: nnkPostfix.newTree(ident"*", oldName)
            else: oldName

  result = nnkProcDef.newTree(
    name,
    newEmptyNode(),  # Term-rewriting
    generics,        # Generic param
    cleanedParams,
    pragmas,
    newEmptyNode(),  # Reserved for future use
    body
  )

macro implDeprecatedBy*(oldName: untyped, replacement: typed, exported: static bool): untyped =
  ## Implement a proc that is deprecated
  if replacement.kind == nnkSym:
    result = overloadSingleSym(oldName, replacement, exported)
  else:
    replacement.expectKind(nnkClosedSymChoice)
    result = newStmtList()
    for overload in replacement:
      result.add overloadSingleSym(oldName, overload, exported)

  when false:
    # View proc signature.
    echo result.treerepr

# Sanity check
# -----------------------------------------------

when isMainModule:
  block:
    proc foo(x: int): int =
      result = x + 2

    implDeprecatedBy(bar, foo, exported = false)

    let z = bar(10)
    echo z

  block:
    proc foo[T](x: T): T =
      result = x + 2

    implDeprecatedBy(bar, foo, exported = false)

    let z = bar(10)
    echo z

  block:
    proc foo[T](x, y: T): T =
      result = x + y + 2

    implDeprecatedBy(bar, foo, exported = false)

    let z = bar(10, 100)
    echo z
