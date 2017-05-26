# From: https://github.com/kostya/benchmarks

import os, strutils, future
import ../arraymancer

proc matgen(n: int): auto =
    result = newTensor(@[n,n],float64,Backend.Cpu)
    let tmp = 1.0 / (n*n).float64
    return lc[tmp * (i - j).float64 * (i + j).float64 | (i <- 0..<n, j<- 0..<n), float64].toTensor(Cpu).reshape(n, n)


var n = 100
if paramCount()>0:
    n = parseInt(paramStr(1))
n = n div 2 * 2

let a, b = matgen n
let c = a * b

echo formatFloat(c[n div 2, n div 2], ffDefault, 8)

# run with kostya_matmul 1500