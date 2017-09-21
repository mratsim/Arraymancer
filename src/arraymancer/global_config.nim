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




# This configures the maximum number of dimensions supported by Arraymancer
# It should improve performance on Cuda and for iterator by storing temporary shape/strides
# that will be used extensively in the loop on the stack.
const MAXDIMS = 8 # 8 because it's a nice number, more is possible upon request.