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

#[

########### Arraymancer Exceptions ############

Author: Kaan Eraslan

Purpose: Implement exceptions that are more specific to tensors and
their usecases.
This file also contains the hierarchy of arraymancer specific exceptions.


Exception Hierarchy
---------------------

1. IndexError

    1. SliceError

        1. SliceStepError

    2. ArgRankError

2. ValueError

    1. IncompatibleShapeError

        1. ElementWiseShapeError

    2. IncompatibleSizeError

3. TypeError

    1. TensorLayoutTypeError

]#

###### Exceptions inheriting IndexError #######

ArgRankError* = object of IndexError
  ## Raised if the number of arguments does not match the tensor rank

AxisRankError* = object of IndexError
  ## Raised if the number of axis does not match the tensor rank

SliceError* =  object of IndexError
  ## Raised if there is an error in slicing.

SliceStepError* = object of SliceError
  ## Raised if the error concerns slicing in a process that operates with steps

###### Exceptions inheriting ValueError ########

IncompatibleShapeError* = object of ValueError
  ## Raised if the shapes of tensors are incompatible for some reason.

ElementWiseShapeError* =  object of IncompatibleShapeError
  ## Raised if the shapes of tensors are incompatible for elementwise operations

IncompatibleSizeError* = object of ValueError
  ## Raised if the sizes of tensors are incompatible for some reason.

IncompatibleSizeReshapeError* = object of IncompatibleSizeError
  ## Raised if the sizes of tensors are incompatible for reshaping


####### Exceptions inheriting TypeError #########

ObjectTypeError* = object of Exception
  ## Raised when the object does not satisfy the necessary conditions
  ## for an operation

TensorLayoutTypeError* = object of ObjectTypeError
  ## Raised when tensor layout is not contigous.
  # Data_structure_helpers.nim de kaldın
