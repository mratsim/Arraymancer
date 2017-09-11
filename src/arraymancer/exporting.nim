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



proc export_tensor*[T](t: Tensor[T]): tuple[shape: seq[int], strides: seq[int], data: seq[T]] =
  ## Export the tensor as a tuple containing
  ## - shape
  ## - strides
  ## - data
  ## If the tensor was not contiguous (a slice for example), it is reshaped to keep and export only useful data.
  ## Data is exported in C order (last index changes the fastest, column in 2D case)

  let contig_t = t.asContiguous

  result.shape = contig_t.shape
  result.strides = contig_t.strides
  result.data = contig_t.data