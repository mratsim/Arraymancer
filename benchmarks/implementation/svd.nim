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

  var At = A.transpose().clone(colMajor) # gesdd destroys its input
  var scratchspace: seq[T]
  gesdd(At, result.Vh, result.S, result.U, scratchspace)

  result.Vh = result.Vh.transpose()
  result.U = result.U.transpose()

proc svd_smart[T](A: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  if A.shape[0] >= A.shape[1]:
    svd_direct(A)
  else:
    svd_transpose(A)

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

const Sizes = [2, 7, 50, 5000]
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

    profile:
      let svd_smart = svd_smart(H)
    echo &"svd_smart took: {stop-start:>4.4f} seconds"

    profile:
      let svd = svd(H)
    echo &"svd (production) took: {stop-start:>4.4f} seconds"


    echo &"\nChecking that we have the same results."
    checkError(svd_direct, svd_transpose)
    checkError(svd_direct, svd_smart)
    checkError(svd_direct, svd)

    when Display:
      echo "\n--- svd_direct -------------------"
      echo svd_direct
      echo "\n--- svd_transpose -------------------"
      echo svd_transpose
      echo "\n--- svd_smart -------------------"
      echo svd_smart
      echo "\n--- svd (production) -------------------"
      echo svd

# ###########################
# Starting a new experiment

# Matrix of shape:       [2, 2]
# Target SVD (U, S, Vh): [2, 2], [2], [2, 2]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0002 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [2, 7]
# Target SVD (U, S, Vh): [2, 2], [2], [2, 7]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0001 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.249000902703301e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 1.249000902703301e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 1.249000902703301e-16
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [2, 50]
# Target SVD (U, S, Vh): [2, 2], [2], [2, 50]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.387778780781446e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 1.387778780781446e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 1.387778780781446e-16
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [2, 5000]
# Target SVD (U, S, Vh): [2, 2], [2], [2, 5000]

# Hilbert matrix creation took: 0.1045 seconds
# svd_direct took:     0.0002 seconds
# svd_transpose took: 0.0002 seconds
# svd_smart took: 0.0001 seconds
# svd (production) took: 0.0001 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 8.049116928532385e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 8.049116928532385e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 8.049116928532385e-16
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [7, 2]
# Target SVD (U, S, Vh): [7, 2], [2], [2, 2]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.249000902703301e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [7, 7]
# Target SVD (U, S, Vh): [7, 7], [7], [7, 7]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [7, 50]
# Target SVD (U, S, Vh): [7, 7], [7], [7, 50]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.059702954533241e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 1.059702954533241e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 1.059702954533241e-16
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [7, 5000]
# Target SVD (U, S, Vh): [7, 7], [7], [7, 5000]

# Hilbert matrix creation took: 0.0861 seconds
# svd_direct took:     0.0014 seconds
# svd_transpose took: 0.0006 seconds
# svd_smart took: 0.0005 seconds
# svd (production) took: 0.0004 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 2.004632944713525e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 2.004632944713525e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 2.004632944713525e-16
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [50, 2]
# Target SVD (U, S, Vh): [50, 2], [2], [2, 2]

# Hilbert matrix creation took: 0.0001 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0000 seconds
# svd_smart took: 0.0000 seconds
# svd (production) took: 0.0000 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.387778780781446e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [50, 7]
# Target SVD (U, S, Vh): [50, 7], [7], [7, 7]

# Hilbert matrix creation took: 0.0001 seconds
# svd_direct took:     0.0000 seconds
# svd_transpose took: 0.0001 seconds
# svd_smart took: 0.0001 seconds
# svd (production) took: 0.0001 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 1.059702954533241e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [50, 50]
# Target SVD (U, S, Vh): [50, 50], [50], [50, 50]

# Hilbert matrix creation took: 0.0000 seconds
# svd_direct took:     0.0007 seconds
# svd_transpose took: 0.0005 seconds
# svd_smart took: 0.0006 seconds
# svd (production) took: 0.0004 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [50, 5000]
# Target SVD (U, S, Vh): [50, 50], [50], [50, 5000]

# Hilbert matrix creation took: 0.1050 seconds
# svd_direct took:     0.0326 seconds
# svd_transpose took: 0.0053 seconds
# svd_smart took: 0.0054 seconds
# svd (production) took: 0.0040 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 8.470677496361287e-17
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 8.470677496361287e-17
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 8.470677496361287e-17
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [5000, 2]
# Target SVD (U, S, Vh): [5000, 2], [2], [2, 2]

# Hilbert matrix creation took: 0.0514 seconds
# svd_direct took:     0.0004 seconds
# svd_transpose took: 0.0003 seconds
# svd_smart took: 0.0002 seconds
# svd (production) took: 0.0005 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 8.049116928532385e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [5000, 7]
# Target SVD (U, S, Vh): [5000, 7], [7], [7, 7]

# Hilbert matrix creation took: 0.0490 seconds
# svd_direct took:     0.0008 seconds
# svd_transpose took: 0.0015 seconds
# svd_smart took: 0.0007 seconds
# svd (production) took: 0.0007 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 2.004632944713525e-16
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [5000, 50]
# Target SVD (U, S, Vh): [5000, 50], [50], [50, 50]

# Hilbert matrix creation took: 0.0509 seconds
# svd_direct took:     0.0054 seconds
# svd_transpose took: 0.0308 seconds
# svd_smart took: 0.0050 seconds
# svd (production) took: 0.0050 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 8.470677496361287e-17
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape:       [5000, 5000]
# Target SVD (U, S, Vh): [5000, 5000], [5000], [5000, 5000]

# Hilbert matrix creation took: 0.0511 seconds
# svd_direct took:     13.9115 seconds
# svd_transpose took: 14.9179 seconds
# svd_smart took: 13.8181 seconds
# svd (production) took: 13.9073 seconds

# Checking that we have the same results.
# mean_absolute_error(svd_direct, svd_transpose) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd_smart) = 0.0
# ---------------------------------------------
# mean_absolute_error(svd_direct, svd) = 0.0
# ---------------------------------------------
