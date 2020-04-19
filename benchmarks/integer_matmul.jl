function main(n)
  n = round(Int, n / 2 * 2)
  a = rand(0:100, n, n)
  b = a
  c = a * b
  v = round(Int, n/2) + 1
  println(c[v, v])
end

function when_isMainModule()
  n = 100
  if length(ARGS) >= 1
    n = parse(Int, ARGS[1])
  end
  main(n)
end

when_isMainModule()

#########
# Results on i9-9980XE
# Skylake-X overclocked 4.1GHz all-core turbo,
# AVX2 4.0GHz all-core, AVX-512 3.5GHz all-core
# Input 1500x1500 random large int64 matrix

# Nim 1.0.4. Compilation option: "-d:danger -d:openmp"
# Julia v1.3.1
# Python 3.8.1 + Numpy-MKL 1.18.0

# Nim: 0.14s, 22.7Mb
# Julia: 1.67s, 246.5Mb
# Python Numpy: 5.69s, 75.9Mb
