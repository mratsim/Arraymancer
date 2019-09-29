import
  times, strformat, math,
  ../../src/arraymancer,
  ../../src/linear_algebra/helpers/[decomposition_lapack, auxiliary_blas, init_colmajor]

# Benchmarks of full SVD implementations
# The goal is to understand the best implementations
# especially in case of tall-and-skinny matrices
# and fat matrices, i.e. when m << n or n << m for m-by-n matrices

# Implementations
# ---------------------------------------------------------------------------------

proc svd_direct[T](A: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  ## SVD using the raw lapack routine gesdd
  ## A = U Σ Vt
  assert A.rank == 2

  var A = A.clone(colMajor) # gesdd destroys its input
  var scratchspace: seq[T]
  gesdd(A, result.U, result.S, result.Vh, scratchspace)

proc svd_transpose[T](A: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  ## SVD using the raw lapack routine gesdd
  ## on the transposed matrix
  ## At = V Σ Ut (Σ = Σt as it is a rectangular diagonal matrix)
  assert A.rank == 2

  var A = A.clone(colMajor) # gesdd destroys its input
  var scratchspace: seq[T]
  gesdd(A, result.Vh, result.S, result.U, scratchspace)

  result.Vh = result.Vh.transpose()
  result.U = result.U.transpose()

proc svd_AAt[T](A: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  ## SVD using eigen values decomposition of A*At
  ## This will significantly improve performance
  ## if A is a M-by-N matrix and M << N
  ## as decomposition will be done on a M-by-M matrix
  ##
  ## However, if a matrix is ill-conditionned (like Hilbert matrix)
  ## the condition number is squared, which will be an issue
  ## if the matrix inverse is needed.
  ## (Condition: a small change in the matrix induces a huge one in its inverse)
  ##
  ## A At = U Σ² Ut   # with cheaper eigenval decomposition
  ## V = At U Σ-1     # recover SVD param
  ##
  ## Shape:
  ## - U: MxM
  ## - Σ: MxM
  ## - Ut: MxM
  ## - V: NxM
  assert A.rank == 2

  var aat: Tensor[T]
  var scratchspace: seq[T]

  let k = min(A.shape[0], A.shape[1])
  let K = max(A.shape[0], A.shape[1])

  # Compute A * At
  aat.newMatrixUninitColMajor(A.shape[0], A.shape[0])
  syrk(1.0, A, AAt, 0, aat, 'U')

  # Decompose A * At = U Σ² Ut
  syevr(aat, 'U', return_eigenvectors = true,
        low_idx = K-k-1, high_idx = K-1,
        eigenval = result.S,
        eigenvec = result.U,
        scratchspace
      )

  # Syevr returns everything in ascending order, we want descending
  result.S = result.S[^1..0|-1]
  result.U = result.U[_, ^1..0|-1]

  # Compute Σ
  apply2_inline(S):
    sqrt(x)

  # V = At U Σ-1
  discard # TODO - need a strided diagonal multiplication.
          #        Due to the numerical stability issue
          #        on ill-conditioned matrices, we would
          #        lose one of the main advantage of SVD
          #        by implementing this scheme.
          #        This is also quite complex.

proc svd_AtA[T](A: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  ## SVD using eigen values decomposition of At*
  ## This will significantly improve performance
  ## if A is a M-by-N matrix and N << M
  ## as decomposition will be done on a N-by-N matrix
  ##
  ## However, if a matrix is ill-conditionned (like Hilbert matrix)
  ## the condition number is squared, which will be an issue
  ## if the matrix inverse is needed.
  ## (Condition: a small change in the matrix induces a huge one in its inverse)
  ##
  ## At A = V Σ² Vt    # with cheaper eigenval decomposition
  ## U = A V Σ-1       # recover SVD param
  ##
  ## Shape:
  ## - A: MxN
  ## - U: MxM
  ## - Σ: MxM
  ## - Ut: MxM
  ## - V: NxM
  assert A.rank == 2

  var ata: Tensor[T]
  var scratchspace: seq[T]

  let k = min(A.shape[0], A.shape[1])
  let K = max(A.shape[0], A.shape[1])

  # Compute At * A
  ata.newMatrixUninitColMajor(A.shape[1], A.shape[1])
  syrk(1.0, A, AtA, 0, ata, 'U')

  # Decompose A * At = U Σ² Ut
  syevr(ata, 'U', return_eigenvectors = true,
        low_idx = K-k-1, high_idx = K-1,
        eigenval = result.S,
        eigenvec = result.V,
        scratchspace
      )

  # Syevr returns everything in ascending order, we want descending
  result.S = result.S[^1..0|-1]
  result.V = result.V[_, ^1..0|-1]

  # Compute Σ
  apply2_inline(S):
    sqrt(x)

  # U = A V Σ-1
  discard # TODO - need a strided diagonal multiplication.
          #        Due to the numerical stability issue
          #        on ill-conditioned matrices, we would
          #        lose one of the main advantage of SVD
          #        by implementing this scheme.
          #        This is also quite complex.

# Setup
# ---------------------------------------------------------------------------------

const Sizes = [100, 200, 400, 800, 1600, 3200]
const Display = false

type FloatType = float64

for Observations in Sizes:
  for Features in Sizes:
    let Max = max(Observations, Features)
    let Min = min(Observations, Features)

    echo "\n###########################"
    echo "Starting a new experiment\n"
    echo &"Matrix of shape:       [{Observations}, {Features}]"
    echo &"Target SVD (U, S, Vh): [{Observations}, {Min}], [{Min}], [{Min}, {Features}]\n"

    var start, stop: float64

    template profile(body: untyped): untyped {.dirty.} =
      start = epochTime()
      body
      stop = epochTime()

    template checkError(x, y: untyped) =
      let err = mean_absolute_error(x.S, y.S)
      let err_negy = mean_absolute_error(x.S, -y.S)
      let xStr = x.astToStr
      let yStr = y.astToStr
      # Strformat not working in templates grrr....

      # Note that signs are completely unstable so errors are not reliable
      # without a deterministic way to fix the signs
      echo "mean_absolute_error(", xStr, ", ", yStr,") = ", err
      echo "---------------------------------------------"

    # Create a known ill-conditionned matrix for testing
    let m = Observations
    let n = Features
    profile:
      let H = hilbert(Max, FloatType)[0..<Observations, 0..<Features]
    echo &"Hilbert matrix creation took: {stop-start:>4.4f} seconds"

    profile:
      let svd_direct = svd_direct(H)
    echo &"svd_direct took:     {stop-start:>4.4f} seconds"

    profile:
      let svd_transpose = svd_transpose(H)
    echo &"svd_transpose took: {stop-start:>4.4f} seconds"

    echo &"\nChecking that we have the same results."
    checkError(svd_direct, svd_transpose)

    when Display:
      echo "\n--- svd_direct -------------------"
      echo svd_direct
      echo "\n--- svd_transpose -------------------"
      echo svd_transpose

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 100]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0001 seconds
# svd_direct took:     0.0063 seconds
# svd_transpose took: 0.0061 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 200]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 200]

# Hilbert matrix creation took: 0.0005 seconds
# svd_direct took:     0.0064 seconds
# svd_transpose took: 0.0056 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 400]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 400]

# Hilbert matrix creation took: 0.0007 seconds
# svd_direct took:     0.0085 seconds
# svd_transpose took: 0.0116 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 800]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 800]

# Hilbert matrix creation took: 0.0025 seconds
# svd_direct took:     0.0137 seconds
# svd_transpose took: 0.0138 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 1600]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 1600]

# Hilbert matrix creation took: 0.0101 seconds
# svd_direct took:     0.0262 seconds
# svd_transpose took: 0.0215 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [100, 3200]
# Target SVD (U, S, Vh): [100, 100], [100], [100, 3200]

# Hilbert matrix creation took: 0.0471 seconds
# svd_direct took:     0.0357 seconds
# svd_transpose took: 0.0352 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 100]
# Target SVD (U, S, Vh): [200, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0001 seconds
# svd_direct took:     0.0038 seconds
# svd_transpose took: 0.0053 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 200]
# Target SVD (U, S, Vh): [200, 200], [200], [200, 200]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0081 seconds
# svd_transpose took: 0.0242 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 400]
# Target SVD (U, S, Vh): [200, 200], [200], [200, 400]

# Hilbert matrix creation took: 0.0002 seconds
# svd_direct took:     0.0396 seconds
# svd_transpose took: 0.0287 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 800]
# Target SVD (U, S, Vh): [200, 200], [200], [200, 800]

# Hilbert matrix creation took: 0.0024 seconds
# svd_direct took:     0.0489 seconds
# svd_transpose took: 0.0381 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 1600]
# Target SVD (U, S, Vh): [200, 200], [200], [200, 1600]

# Hilbert matrix creation took: 0.0096 seconds
# svd_direct took:     0.0508 seconds
# svd_transpose took: 0.0448 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [200, 3200]
# Target SVD (U, S, Vh): [200, 200], [200], [200, 3200]

# Hilbert matrix creation took: 0.0317 seconds
# svd_direct took:     0.0830 seconds
# svd_transpose took: 0.0636 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 100]
# Target SVD (U, S, Vh): [400, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0002 seconds
# svd_direct took:     0.0048 seconds
# svd_transpose took: 0.0050 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 200]
# Target SVD (U, S, Vh): [400, 200], [200], [200, 200]

# Hilbert matrix creation took: 0.0002 seconds
# svd_direct took:     0.0311 seconds
# svd_transpose took: 0.0304 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 400]
# Target SVD (U, S, Vh): [400, 400], [400], [400, 400]

# Hilbert matrix creation took: 0.0002 seconds
# svd_direct took:     0.0503 seconds
# svd_transpose took: 0.0365 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 800]
# Target SVD (U, S, Vh): [400, 400], [400], [400, 800]

# Hilbert matrix creation took: 0.0010 seconds
# svd_direct took:     0.0803 seconds
# svd_transpose took: 0.0746 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 1600]
# Target SVD (U, S, Vh): [400, 400], [400], [400, 1600]

# Hilbert matrix creation took: 0.0043 seconds
# svd_direct took:     0.1002 seconds
# svd_transpose took: 0.0975 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [400, 3200]
# Target SVD (U, S, Vh): [400, 400], [400], [400, 3200]

# Hilbert matrix creation took: 0.0555 seconds
# svd_direct took:     0.1476 seconds
# svd_transpose took: 0.1366 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 100]
# Target SVD (U, S, Vh): [800, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0010 seconds
# svd_direct took:     0.0066 seconds
# svd_transpose took: 0.0056 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 200]
# Target SVD (U, S, Vh): [800, 200], [200], [200, 200]

# Hilbert matrix creation took: 0.0010 seconds
# svd_direct took:     0.0321 seconds
# svd_transpose took: 0.0323 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 400]
# Target SVD (U, S, Vh): [800, 400], [400], [400, 400]

# Hilbert matrix creation took: 0.0010 seconds
# svd_direct took:     0.0544 seconds
# svd_transpose took: 0.0644 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 800]
# Target SVD (U, S, Vh): [800, 800], [800], [800, 800]

# Hilbert matrix creation took: 0.0010 seconds
# svd_direct took:     0.1344 seconds
# svd_transpose took: 0.3119 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 1600]
# Target SVD (U, S, Vh): [800, 800], [800], [800, 1600]

# Hilbert matrix creation took: 0.0042 seconds
# svd_direct took:     0.2747 seconds
# svd_transpose took: 0.3161 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [800, 3200]
# Target SVD (U, S, Vh): [800, 800], [800], [800, 3200]

# Hilbert matrix creation took: 0.0649 seconds
# svd_direct took:     0.3388 seconds
# svd_transpose took: 0.3455 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 100]
# Target SVD (U, S, Vh): [1600, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0095 seconds
# svd_direct took:     0.0069 seconds
# svd_transpose took: 0.0074 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 200]
# Target SVD (U, S, Vh): [1600, 200], [200], [200, 200]

# Hilbert matrix creation took: 0.0096 seconds
# svd_direct took:     0.0393 seconds
# svd_transpose took: 0.0318 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 400]
# Target SVD (U, S, Vh): [1600, 400], [400], [400, 400]

# Hilbert matrix creation took: 0.0094 seconds
# svd_direct took:     0.0580 seconds
# svd_transpose took: 0.0631 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 800]
# Target SVD (U, S, Vh): [1600, 800], [800], [800, 800]

# Hilbert matrix creation took: 0.0042 seconds
# svd_direct took:     0.1839 seconds
# svd_transpose took: 0.1784 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 1600]
# Target SVD (U, S, Vh): [1600, 1600], [1600], [1600, 1600]

# Hilbert matrix creation took: 0.0043 seconds
# svd_direct took:     0.5018 seconds
# svd_transpose took: 0.5065 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [1600, 3200]
# Target SVD (U, S, Vh): [1600, 1600], [1600], [1600, 3200]

# Hilbert matrix creation took: 0.0484 seconds
# svd_direct took:     0.9162 seconds
# svd_transpose took: 0.9187 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 100]
# Target SVD (U, S, Vh): [3200, 100], [100], [100, 100]

# Hilbert matrix creation took: 0.0636 seconds
# svd_direct took:     0.0091 seconds
# svd_transpose took: 0.0095 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 200]
# Target SVD (U, S, Vh): [3200, 200], [200], [200, 200]

# Hilbert matrix creation took: 0.0344 seconds
# svd_direct took:     0.0370 seconds
# svd_transpose took: 0.0325 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 400]
# Target SVD (U, S, Vh): [3200, 400], [400], [400, 400]

# Hilbert matrix creation took: 0.0198 seconds
# svd_direct took:     0.1007 seconds
# svd_transpose took: 0.0733 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 800]
# Target SVD (U, S, Vh): [3200, 800], [800], [800, 800]

# Hilbert matrix creation took: 0.0413 seconds
# svd_direct took:     0.2277 seconds
# svd_transpose took: 0.3807 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 1600]
# Target SVD (U, S, Vh): [3200, 1600], [1600], [1600, 1600]

# Hilbert matrix creation took: 0.0464 seconds
# svd_direct took:     0.7478 seconds
# svd_transpose took: 0.6932 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [3200, 3200]
# Target SVD (U, S, Vh): [3200, 3200], [3200], [3200, 3200]

# Hilbert matrix creation took: 0.0638 seconds
# svd_direct took:     3.4630 seconds
# svd_transpose took: 3.8927 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------
