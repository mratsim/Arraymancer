# Package
version       = "0.0.1"
author        = "Mamy André-Ratsimbazafy"
description   = "Nim tensors / multi-dimensional arrays"
license       = "Apache License 2.0"

srcDir = "src"

# Dependencies
requires "nim >= 0.15.1", "nimblas >= 0.1.3"

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


task test, "Run all tests - internal":
  test "all_tests"