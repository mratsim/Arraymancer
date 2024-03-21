import std / [os, strutils, random]
import ../src/arraymancer

{.passC: "-march=native" .}
# {.passl: "-fopenmp".}
# {.passc: "-fopenmp".}

proc main(n: int) =
  let even_n = n div 2 * 2

  let a, b = randomTensor(n,n, 100.int - 1) # Nim ints are int64 on x86_64
  let c = a*b
  echo $c[even_n div 2, even_n div 2]

when isMainModule:
  if paramCount()>0:
    main(parseInt(paramStr(1)))
  else:
    main(100)

#########
# Results on i9-9980XE
# Skylake-X overclocked 4.1GHz all-core turbo,
# AVX2 4.0GHz all-coreAVX-512 3.5GHz all-core
# Input 1500x1500 random large int64 matrix

# Nim 1.0.4. Compilation option: "-d:danger -d:openmp"
# Julia v1.3.1
# Python 3.8.1 + Numpy-MKL 1.18.0

# Nim: 0.14s, 22.7Mb
# Julia: 1.67s, 246.5Mb
# Python Numpy: 5.69s, 75.9Mb
