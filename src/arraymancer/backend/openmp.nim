
const OMP_FOR_ANNOTATION = "if(ompsize > " & $OMP_FOR_THRESHOLD & ")"

template omp_parallel_countup(i: untyped, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(0, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_forup(i: untyped, start, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(start, ompsize, OMP_FOR_ANNOTATION):
    body
