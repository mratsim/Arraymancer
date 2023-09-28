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

# from Nim https://github.com/nim-lang/Nim/pull/22739 on the stdlib provides a
# `newSeqUninit` for types supporting `supportsCopyMem`
when not declared(newSeqUninit):
  func newSeqUninit*[T](len: Natural): seq[T] {.inline.} =
    ## Creates an uninitialzed seq.
    ## Contrary to newSequnitialized in system.nim this works for any subtype T
    result = newSeqOfCap[T](len)
    result.setLen(len)
