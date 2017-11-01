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

# Importing this library provides fusion operations.
# Those gives either a performance improvements (Fused Multiply-Add)
# Or accuracy improvement by avoiding catastrophic cancellation

proc ln1p*(x: float32): float32 {.importc: "log1pf", header: "<math.h>".}
proc ln1p*(x: float64): float64 {.importc: "log1p", header: "<math.h>".}
  ## Compute ln( 1+x ) and avoids catastrophic cancellation if x << 1
  ## i.e. if x << 1 ln(1+x) ~= x but normal float rounding would do ln(1) = 0 instead.

proc expm1*(x: float32): float32 {.importc: "expm1f", header: "<math.h>".}
proc expm1*(x: float64): float64 {.importc: "expm1", header: "<math.h>".}
  ## Compute exp(x) - 1 and avoids catastrophic cancellation if x ~= 0
  ## i.e. if x ~= 0 exp(x) - 1 ~= x but normal float rounding would do exp(0) - 1 = 0 instead.


## The auto-rewrite do not seem to work :/

# template rewriteLn1p*{ln(`+`(1,x))}(x: typed): type(x) =
#   ## Fuse ``ln(1 + x)`` into a single operation.
#   ##
#   ## Operation fusion leverage the Nim compiler and should not be called explicitly.
#   ln1p x
# 
# template rewriteLn1p*{ln(`+`(x,1))}(x: typed): type(x) =
#   ## Fuse ``ln(x + 1)`` into a single operation.
#   ##
#   ## Operation fusion leverage the Nim compiler and should not be called explicitly.
#   ln1p x

# Note: we don't create the FMA proc as detection of hardware support happens at GCC
# compilation time. Furthermore, the fallback is slower than doing (x*y) + z
# because the fallback do the intermediate computation at full precision.
# So, trust the compiler