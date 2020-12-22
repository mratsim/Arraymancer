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


# Arraymancer-defined pragma
# Due to different scoping rule for pragmas one does not just declare pragmas and use "import" or "include"
# They must be redeclared within each proc.
# As a workaround we use a template

import ../../std_version_types

template withMemoryOptimHints*() =
  when not defined(js):
    {.pragma: align64, codegenDecl: "$# $# __attribute__((aligned(64)))".}
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: align64.}
    {.pragma: restrict.}

const withBuiltins = defined(gcc) or defined(clang)

when withBuiltins:
  proc builtin_assume_aligned[T](data: ptr T, n: csize_t): ptr T {.importc: "__builtin_assume_aligned",noDecl.}

when defined(cpp):
  proc static_cast[T](input: T): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, n: csize_t): ptr T =
  when defined(cpp) and withBuiltins: # builtin_assume_aligned returns void pointers, this does not compile in C++, they must all be typed
    static_cast builtin_assume_aligned(data, n)
  elif withBuiltins:
    builtin_assume_aligned(data, n)
  else:
    data
