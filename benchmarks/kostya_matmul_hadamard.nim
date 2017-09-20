# From: https://github.com/kostya/benchmarks

import os, strutils, sequtils
import ../arraymancer

proc matgen(n: int): auto =
    result = newTensor(@[n,n],float64,Backend.Cpu)
    let tmp = 1.0 / (n*n).float64
    let j_idx = @[toSeq(0..<n)].toTensor(Cpu).astype(float64).broadcast([n,n])
    let i_idx = j_idx.transpose
    ## TODO: +, -, .* are very slow
    return (i_idx - j_idx) .* (i_idx + j_idx) * tmp

var n = 100
if paramCount()>0:
    n = parseInt(paramStr(1))
n = n div 2 * 2

let a, b = matgen n
let c = a * b

echo formatFloat(c[n div 2, n div 2], ffDefault, 8)

# run with kostya_matmul 1500