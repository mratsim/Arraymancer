# Benchmark for https://github.com/mratsim/Arraymancer/issues/164
# Place it in "Arraymancer/build" folder

# Inspired by paper: https://arxiv.org/pdf/1608.00099
#
# Instead of using template recursion at compile-time to create nested for loops
# followed by a search at run-time that would find the appropriate
# looping sequence,
# we use the powerful Nim macros to autogenerate the nested loop statements.
# the code would be as fast as using hardcoded for loop at the price of compiling speed

# If we have an array of shape [4, 9, 7, 5] (and strides [...])
# We want to output index:
# for i in 0 ..< 4:
#   for j in 0 ..< 9:
#     for k in 0 ..< 7:
#       for l in 0 ..< 5:
#         yield i*strides[0] + j*strides[1] + ....


import ../../src/tensor/backend/metadataArray
import std / macros

proc shape_to_strides*(shape: MetadataArray): MetadataArray {.noSideEffect.} =
  var accum = 1
  result.len = shape.len

  for i in countdown(shape.len-1,0):
    result[i] = accum
    accum *= shape[i]

const MAXRANK = 7

type TestTensor = object
  rank: int
  shape, strides: MetadataArray

################################################################################################
#### new nested for loop iteration scheme

iterator iterShape(dim: int): int =
  # Special case if dim = 0 or 1
  # We still need to do a single iteration
  # Index will be 0 computed realIdx will be correct

  let ending = if dim == 0: 0
               else: dim-1
  for i in 0 .. ending:
    yield i

macro forParams(dim: static[int], shape, strides: MetadataArray, statement: untyped): untyped =
  result = newNimNode(nnkStmtList)

  if dim < MAXRANK:
    result.add nnkForStmt.newTree(
      newIdentNode("i" & $dim),
      nnkCall.newTree(
        newIdentNode("iterShape"),
        nnkBracketExpr.newTree(
            shape,
            newLit(dim)
        )
      ),
      getAST forParams(dim + 1, shape, strides, statement)
    )
  else:
    result.add statement

macro genNestedFor(shape, strides: MetadataArray): untyped =

  # 1. Store the loop indexes [i0, i1, i2, i3, i4, ...]
  var iters: array[MAXRANK, NimNode]
  for i in 0 ..< MAXRANK:
    iters[i] = ident("i" & $i)

  # 2. Construct the AST of the inner most statement (after the nested for loops)
  #
  #   var realIdx = 0
  #   realIdx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] ...
  #   yield realIdx
  var realIdxStmt = newNimNode(nnkStmtList)
  realIdxStmt.add(
    # var realIdx = 0
    nnkVarSection.newTree(
        nnkIdentDefs.newTree(
          newIdentNode("realIdx"),
          newEmptyNode(),
          newLit(0)
      )
    )
  )
  for i in 0 ..< MAXRANK:
    # realIdx += i0 * strides[0]
    # realIdx += i1 * strides[1]
    # ...
    realIdxStmt.add(
      newTree(
        nnkInfix,
        ident("+="),
        ident("realIdx"),
        newTree(
          nnkInfix,
          ident("*"),
          iters[i],
          newTree(
            nnkBracketExpr,
            strides,
            newLit(i)
          )
        )
      )
    )
  # yield realIdx
  realIdxStmt.add(
    nnkYieldStmt.newTree(
      ident("realIdx")
    )

  )

  # 3. Generate the nested for loop and embed the realIdx statements
  result = newNimNode(nnkStmtList)
  result.add(
    getAST forParams(
      0,
      shape,
      strides,
      realIdxStmt
    )
  )

  # TODO: currently all computations happen in the innermost loop.
  # Consideration:
  # - Part of it could be hoisted out.
  # - Given that each computation for each loop is done separately the compiler can probably
  #   detect the invariant.
  # - Hoisting the computations would mean holding the intermediate results in registers
  #   This will require 7 intermediate variables (while we already have 7 iteration variables)
  #   and might cause register pressure and register spilling
  # - Fused multiply-add might avoid register pressure and be just as fast as add

iterator nFor_triot(t: TestTensor): int =
  genNestedFor(t.shape, t.strides)

################################################################################################
#### original iteration scheme

import ../../src/tensor/backend/memory_optimization_hints

template initStridedIteration*(coord, backstrides, iter_pos: untyped, t, iter_offset, iter_size: typed): untyped =
  ## Iterator init
  var iter_pos = 0
  withMemoryOptimHints() # MAXRANK = 8, 8 ints = 64 Bytes, cache line = 64 Bytes --> profit !
  var coord {.align64, noInit.}: array[MAXRANK, int]
  var backstrides {.align64, noInit.}: array[MAXRANK, int]
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

  # Calculate initial coords and iter_pos from iteration offset
  if iter_offset != 0:
    var z = 1
    for i in countdown(t.rank - 1,0):
      coord[i] = (iter_offset div z) mod t.shape[i]
      iter_pos += coord[i]*t.strides[i]
      z *= t.shape[i]

template advanceStridedIteration*(coord, backstrides, iter_pos, t, iter_offset, iter_size: typed): untyped =
  ## Computing the next position
  for k in countdown(t.rank - 1,0):
    if coord[k] < t.shape[k]-1:
      coord[k] += 1
      iter_pos += t.strides[k]
      break
    else:
      coord[k] = 0
      iter_pos -= backstrides[k]

template stridedIterationYield*(i, iter_pos: typed) =
  ## Iterator the return value

  # Simplified for testing
  yield iter_pos

template stridedIteration*(t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)
  for i in iter_offset..<(iter_offset+iter_size):
    stridedIterationYield(i, iter_pos)
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

iterator nFor_current(t: TestTensor): int =

  var size = 1
  for i in 0..<t.rank:
    size *= t.shape[i]
  stridedIteration(t, 0, size)


proc warmup() =
  # Raise CPU to max perf even if using ondemand CPU governor
  # (mod is a costly operation)
  var foo = 123
  for i in 0 ..< 100000000:
    foo += i*i mod 456
    foo = foo mod 789
  echo foo

################################################################################################
#### Bench

import std / times


let a = [2, 3, 4].toMetadataArray
let small = TestTensor(rank: 3, shape: a, strides: a.shape_to_strides)

const
  dz = 0.01
  z = 100
  spaceSteps = int(z / dz) # 10000
  timeSteps = 50000
  b = [timeSteps, spaceSteps].toMetadataArray

let big = TestTensor(rank: 2, shape: b, strides: b.shape_to_strides)


proc small_tensor_test_triot(t: TestTensor): uint {.noinline, noSideEffect.}=
  for _ in 0..<100_000_000:
    for i in nFor_triot(t):
      result += i.uint

proc small_tensor_test_current(t: TestTensor): uint {.noinline, noSideEffect.}=
  for _ in 0..<100_000_000:
    for i in nFor_current(t):
      result += i.uint

proc big_tensor_test_triot(t: TestTensor): uint {.noinline, noSideEffect.}=
  for i in nFor_triot(t):
    result += i.uint

proc big_tensor_test_current(t: TestTensor): uint {.noinline, noSideEffect.}=
  for i in nFor_current(t):
    result += i.uint

##################
# Warmup
var start = cpuTime()
warmup()
var stop = cpuTime()
echo "Warmup: " & $(stop - start) & "s"
#################

start = cpuTime()
echo small_tensor_test_current(small)
stop = cpuTime()
echo "Current implementation - small tensors: " & $(stop - start) & "s"

start = cpuTime()
echo small_tensor_test_triot(small)
stop = cpuTime()
echo "Macros generated nested for loop - small tensors: " & $(stop - start) & "s"

start = cpuTime()
echo big_tensor_test_current(big)
stop = cpuTime()
echo "Current implementation - big tensor: " & $(stop - start) & "s"

start = cpuTime()
echo big_tensor_test_triot(big)
stop = cpuTime()
echo "Macros generated nested for loop - big tensor: " & $(stop - start) & "s"

##############
# On i5-5257U (Broadwell mobile dual core)

# 369
# Warmup: 0.413704s
# 27600000000
# Current implementation - small tensors: 5.74213s
# 27600000000
# Macros generated nested for loop - small tensors: 23.201846s
# 124999999750000000
# Current implementation - big tensor: 1.15879s
# 124999999750000000
# Macros generated nested for loop - big tensor: 5.242488999999999s
# 35.93s, 1.0Mb
