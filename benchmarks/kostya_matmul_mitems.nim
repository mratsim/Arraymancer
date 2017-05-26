# From: https://github.com/kostya/benchmarks

import os, strutils
import ../arraymancer

proc divmod[T: SomeInteger](n: T, b: T): (T, T) =
    ## return (n div base, n mod base)
    return (n div b, n mod b)

proc matgen(n: int): auto =
    result = newTensor(@[n,n],float64,Backend.Cpu)
    let tmp = 1.0 / (n*n).float64
    var counter = 0
    for val in result.mitems:
        let (i, j) = counter.divmod(n)
        val = (i - j).float64 * (i + j).float64 * tmp
        inc counter

var n = 100
if paramCount()>0:
    n = parseInt(paramStr(1))
n = n div 2 * 2

let a, b = matgen n
let c = a * b

echo formatFloat(c[n div 2, n div 2], ffDefault, 8)

# run with kostya_matmul 1500