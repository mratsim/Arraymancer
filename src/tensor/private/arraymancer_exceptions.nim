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

Purpose: Implement exceptions that are more specific to tensors and
their usecases.
This file also contains the hierarchy of arraymancer specific exceptions.


Exception Hierarchy
---------------------

1. IndexError

    1. SliceError
    2. AxisIndexError

2. ValueError

    1. IncompatibleShapeError
    2. IncompatibleSizeError
    3. RankError
    4. TensorLayoutError
    5. IncompatibleRankError

]#

###### Exceptions inheriting IndexError #######

SliceError* =  object of IndexError
  ## Raised if there is an error in slicing.

AxisIndexError* = object of IndexError
  ## Raised if the number of axis does not match the tensor rank

LevelLengthError* = object of IndexError
  ## Raised if the lengths of sequences at the same level differs.


###### Exceptions inheriting ValueError ########

IncompatibleShapeError* = object of ValueError
  ## Raised if the shapes of tensors are incompatible for some reason.

IncompatibleSizeError* = object of ValueError
  ## Raised if the sizes of tensors are incompatible for some reason.

RankError* = object of ValueError
  ## Raised if the number of arguments does not match the tensor rank



####### Exceptions inheriting TypeError #########

TensorLayoutError* = object of ValueError
  ## Raised when tensor layout is not contigous.

IncompatibleRankError* = object of ValueError
  ## Raised when tensor's rank does not satisfy the necessary conditions
  ## for an operation.
