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


import  ./data_structure,
        ./shapeshifting


# TODO, find a way to test that that moves works properly (global counter  for testing?)

template asContiguous*[T](t: Tensor[T]{call}, layout: OrderType = rowMajor, force: bool = false): Tensor[T] =
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default tensor will be rowMajor.
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout
  ##
  ## This is a move optimization for function chaining to avoid copying value returned by the previous function

  unsafeContiguous(t, layout, force)

template permute*[T](t: Tensor[T]{call}, dims: varargs[int]): Tensor[T] =
  ## Permute dimensions of a tensors
  ## Input:
  ##   - a tensor
  ##   - the new dimension order
  ## Returns:
  ##   - a tensor with re-order dimension
  ## Usage:
  ##  .. code:: nim
  ##     a.permute(0,2,1) # dim 0 stays at 0, dim 1 becomes dim 2 and dim 2 becomes dim 1
  ##
  ## This is a move optimization for function chaining to avoid copying value returned by the previous function
  unsafePermute(t, dims)