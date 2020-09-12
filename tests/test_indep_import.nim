#[ This test simply checks whether independent imports of the
   tensor submodule work as expected
]#

import ../src/arraymancer/tensor
import ../src/arraymancer/stats/kde

let t = zeros[float](100)

doAssert t.size == 100
doAssert kde(t).size == 1000
