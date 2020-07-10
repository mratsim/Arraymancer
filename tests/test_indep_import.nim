#[ This test simply checks whether independent imports of the
   tensor submodule work as expected
]#

import arraymancer/tensor

let t = zeros[float](100)

doAssert t.size == 100
