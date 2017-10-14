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


# Explanation:
#
# - Cuda C++ code is handled by p_kernels_cuda
# - p_kernels_cuda is compiled separately as a C++ file to avoid polluting the whole project with C++ semantics
# - The compiled C++ is renamed to .cu to make NVCC happy
# - We tell Nim not to forget to compile/link to this .cu

mode = ScriptMode.Verbose

# We must go back "./src/tensor/private/"
const par_nimcache = thisDir() & "/../../../nimcache/"
const compiled_name = par_nimcache & "arraymancer_p_kernels_cuda"

const par_nimcache_esc = "\"" & thisDir() & "\"/../../../nimcache/"

# Compile the Cuda kernels as stand-alone
selfExec "cpp --nimcache:" & par_nimcache_esc & " -c p_kernels_cuda"

# Rename the .cpp to .cu to keep NVCC happy
if fileExists(compiled_name & ".cpp"):
  mvFile(compiled_name & ".cpp", compiled_name & ".cu")
else:
  echo "ERROR: File " & compiled_name & ".cpp was not found"