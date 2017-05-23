### Package
version       = "0.0.1"
author        = "Mamy André-Ratsimbazafy"
description   = "Nim tensors / multi-dimensional arrays"
license       = "Apache License 2.0"

### Dependencies
requires "nim >= 0.15.1", "nimblas >= 0.1.3"

### BLAS support
## OSX
# switch("define","openblas")
# switch("clibdir", "/usr/local/opt/openblas/lib")
# switch("cincludes", "/usr/local/opt/openblas/include")

### BLIS support
# switch("define","blis")

## Testing tasks
proc test(name: string) =
  if not dirExists "bin":
    mkDir "bin"
  if not dirExists "nimcache":
    mkDir "nimcache"
  --run
  --nimcache: "nimcache"
  switch("out", ("./bin/" & name))
  setCommand "c", "tests/" & name & ".nim"


task test, "Run all tests - Default BLAS":
  test "all_tests"

task test_openblas, "Run all tests - OpenBLAS":
  ## Should work but somehow Nim doesn't find libopenblas.dylib on MacOS
  when defined(macosx):
    switch("define","openblas")
    switch("clibdir", "/usr/local/opt/openblas/lib")
    switch("cincludes", "/usr/local/opt/openblas/include")
  test "all_tests"

task test_blis, "Run all tests - BLIS":
  switch("define","blis")
  test "all_tests"