# Copyright 2017 Mamy André-Ratsimbazafy
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

when defined(blis):
  static: echo "--USING BLIS--"
  include ./blis
  let blis_status = bli_init()
  echo "Blis initiatialization status: " & $blis_status
  ### Should add a quit proc but the following doesn't compile
  # proc quit_blis() =
  #   echo "Blis quit status: " & $bli_finalize()
  #
  # addQuitProc quit_blis

else:
  static: echo "Consider adding BLIS from \"https://github.com/flame/blis\" " &
          "and compile Arraymancer with \"-d:blis\" " &
          "for operations on array slices without copy. " &
          "OSX users can install it through Homebrew."

