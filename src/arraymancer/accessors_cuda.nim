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



# For element-wise operations, instead of a sequential iterator like for CPU,
# it will be faster to have many threads compute the index -> offset and update
# the data at this offset.
#
# For this we need:
#   - to store strides and offset on the cuda device to avoid copies
#   - a way to convert element #10 of the tensor to the real offset (column major),
#     the kernels won't use tensor[2,5] as an index


proc getIndexOfElementID[T](t: Tensor[T], element_id: int): int {.noSideEffect,used.} =
  ## Convert "Give me element 10" to the real index/memory offset.
  ## Reference Nim CPU version
  ## This is not meant to be used on serial architecture due to the division overhead.
  ## On GPU however it will allow threads to address the real memory addresses independantly.

  result = 0
  var reminderOffset = element_id
  var dimIndex: int

  for k in countdown(t.rank - 1,0):
    ## hopefully the compiler doesn't do division twice ...
    dimIndex = reminderOffset mod t.shape[k]
    reminderOffset = reminderOffset div t.shape[k]

    result += dimIndex * t.strides[k]