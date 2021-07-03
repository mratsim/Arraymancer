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

import ../../src/arraymancer, ../testutils
import math, unittest, sequtils, strutils

proc compareStrings(t1, t2: string) =
  let t1S = t1.splitLines
  let t2S = t2.splitLines
  check t1S.len == t2S.len
  for (x, y) in zip(t1S, t2S):
    check x.strip == y.strip

proc main() =
  suite "Displaying tensors":
    test "Display compiles":
      const
        a = @[1, 2, 3, 4, 5]
        b = @[1, 2, 3, 4, 5]

      var
        vandermonde: seq[seq[int]]
        row: seq[int]

      vandermonde = newSeq[seq[int]]()

      for i, aa in a:
        row = newSeq[int]()
        vandermonde.add(row)
        for j, bb in b:
          vandermonde[i].add(aa^bb)

      # @[@[1, 1, 1, 1, 1], @[2, 4, 8, 16, 32], @[3, 9, 27, 81, 243], @[4, 16, 64, 256, 1024], @[5, 25, 125, 625, 3125]]


      let t_van = vandermonde.toTensor()
      when not compiles(echo t_van): check: false

      check $t_van == """
Tensor[system.int] of shape "[5, 5]" on backend "Cpu"
|1       1     1     1     1|
|2       4     8    16    32|
|3       9    27    81   243|
|4      16    64   256  1024|
|5      25   125   625  3125|
"""

    test "Disp3d + Concat + SlicerMut bug with empty tensors":
      let a = [4, 3, 2, 1, 8, 7, 6, 5].toTensor.reshape(2, 1, 4)
      discard $a

    test "Display 3D tensor":
      let t = toSeq(1..24).toTensor().reshape(2,3,4)
      compareStrings($t, """
Tensor[system.int] of shape "[2, 3, 4]" on backend "Cpu"
          0                      1
|1      2     3     4| |13    14    15    16|
|5      6     7     8| |17    18    19    20|
|9     10    11    12| |21    22    23    24|
""")

    test "Display 4D tensor":
      let t = toSeq(1..72).toTensor().reshape(2,3,4,3)
      compareStrings($t, """
Tensor[system.int] of shape "[2, 3, 4, 3]" on backend "Cpu"
         0                1                2
  |1      2     3| |13    14    15| |25    26    27|
0 |4      5     6| |16    17    18| |28    29    30|
  |7      8     9| |19    20    21| |31    32    33|
  |10    11    12| |22    23    24| |34    35    36|
  --------------------------------------------------
         0                1                2
  |37    38    39| |49    50    51| |61    62    63|
1 |40    41    42| |52    53    54| |64    65    66|
  |43    44    45| |55    56    57| |67    68    69|
  |46    47    48| |58    59    60| |70    71    72|
  --------------------------------------------------
""")

    test "Display 4D tensor with float values":
      let t = linspace(0.0, 100.0, 72).reshape(2,3,4,3)
      ## NOTE: I hope this one isn't too flaky
      when (NimMajor, NimMinor, NimPatch) >= (1, 4, 0):
        compareStrings($t, """
  Tensor[system.float] of shape "[2, 3, 4, 3]" on backend "Cpu"
                                  0                                                             1                                                             2
    |0.000000000000000    1.408450704225352    2.816901408450704| |16.90140845070422    18.30985915492958    19.71830985915493| |33.80281690140845    35.21126760563380    36.61971830985915|
  0 |4.225352112676056    5.633802816901408    7.042253521126760| |21.12676056338028    22.53521126760563    23.94366197183098| |38.02816901408450    39.43661971830986    40.84507042253521|
    |8.450704225352112    9.859154929577464    11.26760563380282| |25.35211267605634    26.76056338028169    28.16901408450704| |42.25352112676056    43.66197183098591    45.07042253521126|
    |12.67605633802817    14.08450704225352    15.49295774647887| |29.57746478873239    30.98591549295774    32.39436619718310| |46.47887323943662    47.88732394366197    49.29577464788732|
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                  0                                                             1                                                             2
    |50.70422535211267    52.11267605633802    53.52112676056338| |67.60563380281690    69.01408450704224    70.42253521126759| |84.50704225352104    85.91549295774638    87.32394366197173|
  1 |54.92957746478873    56.33802816901408    57.74647887323943| |71.83098591549293    73.23943661971828    74.64788732394362| |88.73239436619707    90.14084507042242    91.54929577464776|
    |59.15492957746478    60.56338028169014    61.97183098591549| |76.05633802816897    77.46478873239431    78.87323943661966| |92.95774647887310    94.36619718309845    95.77464788732379|
    |63.38028169014084    64.78873239436619    66.19718309859155| |80.28169014084500    81.69014084507035    83.09859154929569| |97.18309859154914    98.59154929577448    99.99999999999983|
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  """)

    test "Display 4D tensor with float values and custom precision":
      let t = linspace(0.0, 100.0, 72).reshape(2,3,4,3)
      when (NimMajor, NimMinor, NimPatch) >= (1, 4, 0):
        compareStrings(t.pretty(4), """
  Tensor[system.float] of shape "[2, 3, 4, 3]" on backend "Cpu"
                0                         1                         2
    |0.000    1.408    2.817| |16.90    18.31    19.72| |33.80    35.21    36.62|
  0 |4.225    5.634    7.042| |21.13    22.54    23.94| |38.03    39.44    40.85|
    |8.451    9.859    11.27| |25.35    26.76    28.17| |42.25    43.66    45.07|
    |12.68    14.08    15.49| |29.58    30.99    32.39| |46.48    47.89    49.30|
    -----------------------------------------------------------------------------
                0                         1                         2
    |50.70    52.11    53.52| |67.61    69.01    70.42| |84.51    85.92    87.32|
  1 |54.93    56.34    57.75| |71.83    73.24    74.65| |88.73    90.14    91.55|
    |59.15    60.56    61.97| |76.06    77.46    78.87| |92.96    94.37    95.77|
    |63.38    64.79    66.20| |80.28    81.69    83.10| |97.18    98.59    100.0|
    -----------------------------------------------------------------------------
  """)

    test "Display 5D tensor":
      let t1 = toSeq(1..144).toTensor().reshape(2,3,4,3,2)
      compareStrings($t1, """
Tensor[system.int] of shape "[2, 3, 4, 3, 2]" on backend "Cpu"
                          0                            |                            1
       0            1            2            3        |         0            1            2            3
  |1        2| |7        8| |13      14| |19      20|  |    |73      74| |79      80| |85      86| |91      92|
0 |3        4| |9       10| |15      16| |21      22|  |  0 |75      76| |81      82| |87      88| |93      94|
  |5        6| |11      12| |17      18| |23      24|  |    |77      78| |83      84| |89      90| |95      96|
  ---------------------------------------------------  |    ---------------------------------------------------
       0            1            2            3        |         0            1            2            3
  |25      26| |31      32| |37      38| |43      44|  |    |97      98| |103    104| |109    110| |115    116|
1 |27      28| |33      34| |39      40| |45      46|  |  1 |99     100| |105    106| |111    112| |117    118|
  |29      30| |35      36| |41      42| |47      48|  |    |101    102| |107    108| |113    114| |119    120|
  ---------------------------------------------------  |    ---------------------------------------------------
       0            1            2            3        |         0            1            2            3
  |49      50| |55      56| |61      62| |67      68|  |    |121    122| |127    128| |133    134| |139    140|
2 |51      52| |57      58| |63      64| |69      70|  |  2 |123    124| |129    130| |135    136| |141    142|
  |53      54| |59      60| |65      66| |71      72|  |    |125    126| |131    132| |137    138| |143    144|
  ---------------------------------------------------  |    ---------------------------------------------------
""")

      let t2 = toSeq(1..72*3).toTensor().reshape(3,3,4,3,2)
      compareStrings($t2, """
Tensor[system.int] of shape "[3, 3, 4, 3, 2]" on backend "Cpu"
                          0                            |                            1                            |                            2
       0            1            2            3        |         0            1            2            3        |         0            1            2            3
  |1        2| |7        8| |13      14| |19      20|  |    |73      74| |79      80| |85      86| |91      92|  |    |145    146| |151    152| |157    158| |163    164|
0 |3        4| |9       10| |15      16| |21      22|  |  0 |75      76| |81      82| |87      88| |93      94|  |  0 |147    148| |153    154| |159    160| |165    166|
  |5        6| |11      12| |17      18| |23      24|  |    |77      78| |83      84| |89      90| |95      96|  |    |149    150| |155    156| |161    162| |167    168|
  ---------------------------------------------------  |    ---------------------------------------------------  |    ---------------------------------------------------
       0            1            2            3        |         0            1            2            3        |         0            1            2            3
  |25      26| |31      32| |37      38| |43      44|  |    |97      98| |103    104| |109    110| |115    116|  |    |169    170| |175    176| |181    182| |187    188|
1 |27      28| |33      34| |39      40| |45      46|  |  1 |99     100| |105    106| |111    112| |117    118|  |  1 |171    172| |177    178| |183    184| |189    190|
  |29      30| |35      36| |41      42| |47      48|  |    |101    102| |107    108| |113    114| |119    120|  |    |173    174| |179    180| |185    186| |191    192|
  ---------------------------------------------------  |    ---------------------------------------------------  |    ---------------------------------------------------
       0            1            2            3        |         0            1            2            3        |         0            1            2            3
  |49      50| |55      56| |61      62| |67      68|  |    |121    122| |127    128| |133    134| |139    140|  |    |193    194| |199    200| |205    206| |211    212|
2 |51      52| |57      58| |63      64| |69      70|  |  2 |123    124| |129    130| |135    136| |141    142|  |  2 |195    196| |201    202| |207    208| |213    214|
  |53      54| |59      60| |65      66| |71      72|  |    |125    126| |131    132| |137    138| |143    144|  |    |197    198| |203    204| |209    210| |215    216|
  ---------------------------------------------------  |    ---------------------------------------------------  |    ---------------------------------------------------
""")

    test "Display 5D tensor with large numbers":
      let t = toSeq(1_000_000_000..1_000_000_000+72*3 - 1).toTensor().reshape(3,3,4,3,2)
      compareStrings($t, """
Tensor[system.int] of shape "[3, 3, 4, 3, 2]" on backend "Cpu"
                                                      0                                                        |                                                        1                                                        |                                                        2
              0                          1                          2                          3               |                0                          1                          2                          3               |                0                          1                          2                          3
  |1000000000    1000000001| |1000000006    1000000007| |1000000012    1000000013| |1000000018    1000000019|  |    |1000000072    1000000073| |1000000078    1000000079| |1000000084    1000000085| |1000000090    1000000091|  |    |1000000144    1000000145| |1000000150    1000000151| |1000000156    1000000157| |1000000162    1000000163|
0 |1000000002    1000000003| |1000000008    1000000009| |1000000014    1000000015| |1000000020    1000000021|  |  0 |1000000074    1000000075| |1000000080    1000000081| |1000000086    1000000087| |1000000092    1000000093|  |  0 |1000000146    1000000147| |1000000152    1000000153| |1000000158    1000000159| |1000000164    1000000165|
  |1000000004    1000000005| |1000000010    1000000011| |1000000016    1000000017| |1000000022    1000000023|  |    |1000000076    1000000077| |1000000082    1000000083| |1000000088    1000000089| |1000000094    1000000095|  |    |1000000148    1000000149| |1000000154    1000000155| |1000000160    1000000161| |1000000166    1000000167|
  -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------
              0                          1                          2                          3               |                0                          1                          2                          3               |                0                          1                          2                          3
  |1000000024    1000000025| |1000000030    1000000031| |1000000036    1000000037| |1000000042    1000000043|  |    |1000000096    1000000097| |1000000102    1000000103| |1000000108    1000000109| |1000000114    1000000115|  |    |1000000168    1000000169| |1000000174    1000000175| |1000000180    1000000181| |1000000186    1000000187|
1 |1000000026    1000000027| |1000000032    1000000033| |1000000038    1000000039| |1000000044    1000000045|  |  1 |1000000098    1000000099| |1000000104    1000000105| |1000000110    1000000111| |1000000116    1000000117|  |  1 |1000000170    1000000171| |1000000176    1000000177| |1000000182    1000000183| |1000000188    1000000189|
  |1000000028    1000000029| |1000000034    1000000035| |1000000040    1000000041| |1000000046    1000000047|  |    |1000000100    1000000101| |1000000106    1000000107| |1000000112    1000000113| |1000000118    1000000119|  |    |1000000172    1000000173| |1000000178    1000000179| |1000000184    1000000185| |1000000190    1000000191|
  -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------
              0                          1                          2                          3               |                0                          1                          2                          3               |                0                          1                          2                          3
  |1000000048    1000000049| |1000000054    1000000055| |1000000060    1000000061| |1000000066    1000000067|  |    |1000000120    1000000121| |1000000126    1000000127| |1000000132    1000000133| |1000000138    1000000139|  |    |1000000192    1000000193| |1000000198    1000000199| |1000000204    1000000205| |1000000210    1000000211|
2 |1000000050    1000000051| |1000000056    1000000057| |1000000062    1000000063| |1000000068    1000000069|  |  2 |1000000122    1000000123| |1000000128    1000000129| |1000000134    1000000135| |1000000140    1000000141|  |  2 |1000000194    1000000195| |1000000200    1000000201| |1000000206    1000000207| |1000000212    1000000213|
  |1000000052    1000000053| |1000000058    1000000059| |1000000064    1000000065| |1000000070    1000000071|  |    |1000000124    1000000125| |1000000130    1000000131| |1000000136    1000000137| |1000000142    1000000143|  |    |1000000196    1000000197| |1000000202    1000000203| |1000000208    1000000209| |1000000214    1000000215|
  -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------  |    -----------------------------------------------------------------------------------------------------------
""")

    test "Display 5D tensor with string elements":
      let t = toSeq(1..72).mapIt("Value: " & $it).toTensor.reshape(2,3,3,4)
      compareStrings($t, """
Tensor[system.string] of shape "[2, 3, 3, 4]" on backend "Cpu"
                          0                                                  1                                                  2
  |Value: 1      Value: 2     Value: 3     Value: 4| |Value: 13    Value: 14    Value: 15    Value: 16| |Value: 25    Value: 26    Value: 27    Value: 28|
0 |Value: 5      Value: 6     Value: 7     Value: 8| |Value: 17    Value: 18    Value: 19    Value: 20| |Value: 29    Value: 30    Value: 31    Value: 32|
  |Value: 9     Value: 10    Value: 11    Value: 12| |Value: 21    Value: 22    Value: 23    Value: 24| |Value: 33    Value: 34    Value: 35    Value: 36|
  --------------------------------------------------------------------------------------------------------------------------------------------------------
                          0                                                  1                                                  2
  |Value: 37    Value: 38    Value: 39    Value: 40| |Value: 49    Value: 50    Value: 51    Value: 52| |Value: 61    Value: 62    Value: 63    Value: 64|
1 |Value: 41    Value: 42    Value: 43    Value: 44| |Value: 53    Value: 54    Value: 55    Value: 56| |Value: 65    Value: 66    Value: 67    Value: 68|
  |Value: 45    Value: 46    Value: 47    Value: 48| |Value: 57    Value: 58    Value: 59    Value: 60| |Value: 69    Value: 70    Value: 71    Value: 72|
  --------------------------------------------------------------------------------------------------------------------------------------------------------
""")

main()
GC_fullCollect()
