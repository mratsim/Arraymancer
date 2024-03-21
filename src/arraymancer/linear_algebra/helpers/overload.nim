import std / macros

# Alias / overload generator
# --------------------------------------------------------------------------------------
macro overload*(overloaded_name: untyped, lapack_name: typed{nkSym}): untyped =
  let impl = lapack_name.getImpl()
  impl.expectKind {nnkProcDef, nnkFuncDef}

  # We can't just `result[0] = overloaded_name`
  # as libName (lapack library) is not defined in this scope

  var
    params = @[newEmptyNode()] # No return value for all Lapack proc
    body = newCall(lapack_name)

  impl[3].expectKind nnkFormalParams
  for idx in 1 ..< impl[3].len:
    # Skip arg 0, the return type which is always empty
    params.add impl[3][idx]
    body.add impl[3][idx][0]

  result = newProc(
    name = overloaded_name,
    params = params,
    body = body,
    procType = nnkTemplateDef
    # pragmas = nnkPragma.newTree(ident"inline")
  )

  when false:
    # View proc signature.
    # Some procs like syevr have over 20 parameters
    echo result.toStrLit
