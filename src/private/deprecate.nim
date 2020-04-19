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

import macros

# Handle deprecation and replacement
# --------------------------------------------------------------------------------------

macro implDeprecatedBy*(procName: untyped, replacement: typed{nkSym}): untyped =
  ## Implement a proc that is deprecated
  let impl = replacement.getImpl()
  impl.expectKind {nnkProcDef, nnkFuncDef}

  impl[3].expectKind nnkFormalParams
  var
    params = @[impl[3][0]] # Return value
    body = newCall(replacement)

  for idx in 1 ..< impl[3].len:
    params.add impl[3][idx]
    body.add impl[3][idx][0]

  impl[4].expectKind({nnkEmpty, nnkPragma})
  var pragmas = impl[4]
  if pragmas.kind == nnkEmpty:
    pragmas = nnkPragma.newTree()

  pragmas.add ident"inline"
  pragmas.add ident"noInit"
  pragmas.add nnkExprColonExpr.newTree(
    ident"deprecated",
    newLit("Use `" & $replacement & "` instead")
  )

  result = newProc(
    name = procName,
    params = params,
    body = body,
    procType = nnkProcDef,
    pragmas = pragmas
  )

  when false:
    # View proc signature.
    echo result.toStrLit

# Sanity check
# -----------------------------------------------

when isMainModule:

  proc foo(x: int): int =
    result = x + 2

  implDeprecatedBy(bar, foo)

  let z = bar(10)
  echo z
