# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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

# Split tests to save on memory
# (https://github.com/mratsim/Arraymancer/issues/359#issuecomment-500107895)

import
  ../tensor/test_init,
  ../tensor/test_operators_comparison,
  ../tensor/test_accessors,
  ../tensor/test_accessors_slicer,
  ../tensor/test_selectors,
  ../tensor/test_fancy_indexing,
  ../tensor/test_display
