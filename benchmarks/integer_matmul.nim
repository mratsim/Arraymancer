import os, strutils, random
import ../src/arraymancer

{.passC: "-march=native" .}
# {.passl: "-fopenmp".}
# {.passc: "-fopenmp".}

proc main(n: int) =
  let even_n = n div 2 * 2

  let a, b = randomTensor(n,n, 100_000_000.int - 1) # Nim ints are int64 on x86_64
  let c = a*b
  echo $c[even_n div 2, even_n div 2]

when isMainModule:
  if paramCount()>0:
    main(parseInt(paramStr(1)))
  else:
    main(100)

#########
# Results on MacOS + i5-5257U (Broadwell dual-core mobile 2.7GHz, turbo 3.1)
# Input 1500x1500 random large int64 matrix

# Nim 0.17.3 (devel) NO openMP. Compilation option: "-d:release --passc:-march=native"
# Julia v6.0
# Python 3.5.2 + Numpy 1.12

# Nim: 1.72s, 90 MB
# Julia: 4.49s, 185.2 MB
# Python Numpy: 9.49s, 55.8 MB

#########
# Results on Linux + E3-1230v5 (Skylake quad-core 3.4 GHz, turbo 3.8)
# Input 1500x1500 random large int64 matrix

# Nim 0.17.3 (devel) WITH OpenMP. Compilation option: "-d:release --passc:-march=native -d:openmp"
# Julia v6.0
# Python 3.6.2 + Numpy 1.12 compiled from source with march=native and openBLAS

# Nim + OpenMP: 0.36s, 55.5 MB
# Julia: 3.11s, 207.6 MB
# Python Numpy: 8.03s, 58.9 MB
