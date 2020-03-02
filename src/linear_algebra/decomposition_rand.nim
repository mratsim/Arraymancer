# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./helpers/[decomposition_lapack, auxiliary_lapack, triangular, init_colmajor],
  ../private/sequninit,
  ../tensor/tensor

from ./decomposition import svd

# LU Factorization - private implementations
# -------------------------------------------

proc lu_permuted_inplace(a: var Tensor) =
  ## Compute the pivoted LU decomposition of an input matrix ``a``.
  ## This overwrites the input matrix with the pivoted lower triangular matrix
  ##
  ## The decomposition solves the equation:
  ## A = P L U
  ##
  ## And the procedure returns PL
  ##
  ## where:
  ##   - P is a permutation matrix
  ##   - L is a lower-triangular matrix with unit diagonal
  ##   - U is an upper-triangular matrix
  ##
  ## Input:
  ##   - ``a``, a MxN matrix
  ##
  ## Output:
  ##   with K = min(M, N)
  ##   - PL, the product of P and L, of shape [M, K]

  assert a.rank == 2
  assert a.is_F_contiguous

  let k = min(a.shape[0], a.shape[1])
  var pivot_indices = newSeqUninit[int32](k)

  getrf(a, pivot_indices)

  a = a[_, 0..<k]
  tril_unit_diag_mut(a)
  laswp(a, pivot_indices, pivot_from = -1)

# Randomized truncated SVD
# -------------------------------------------

proc svd_randomized*[T](
        A: Tensor[T],
        n_components = 2,
        n_oversamples = 5,
        n_power_iters = 2
        # TODO: choose power iteration normalizer (QR, LU, none)
        # TODO: choose sample init between gaussian, uniform and Rademacher
        # TODO: seed and RNG overload
      ): tuple[U, S, Vh: Tensor[T]] =
  ## Compute approximate nearly optimal truncated Singular Value Decomposition
  ## of an input matrix ``a``.
  ##
  ## Decomposition is truncated to nb_components.
  ##
  ## Increasing nb_oversamples or nb_iter increases the accuracy of the approximation
  ##
  ## Input:
  ##   - A, a matrix of shape [M, N]
  ##   - nb_components: rank/dimension of the approximation
  ##     i.e. number of singular values and vectors to extract
  ##     Must be lower than min(M, N)
  ##     Default to 2 for 2D visualization
  ##   - nb_oversamples: Additional number of random projections in the sampling matrix
  ##       Recommended range 2 .. 10
  ##   - nb_power_iter: Number of power iterations
  ##     Power iterations enforce rapid decay of singular values and allow
  ##     the algorithm to sample dominant singular values and suppress
  ##     irrelevant information. Useful for noisy problems.
  ##
  ## Returns:
  ##   with K = nb_components
  ##   - U: Unitary matrix of shape [M, K] with rank-K approximation of left singular vectors as columns
  ##   - S: Rank-k approximation of singular values diagonal of length K in decreasing order
  ##   - Vh: Unitary matrix of shape [K, N] with rank-K approximation of right singular vectors as rows
  ##
  ## This is an approximate solution of the equation:
  ## A = U S V.h
  ##
  ## - with S being a diagonal matrix of singular values
  ## - with V being the right singular vectors and
  ##   V.h being the hermitian conjugate of V
  ##   for real matrices, this is equivalent to V.t (transpose)
  ##
  ## ⚠️: Input must not contain NaN
  ##
  ## Exception:
  ##   - This can throw if the algorithm did not converge.
  ##
  ## References:
  ##
  ## - A Randomized Algorithm for Principal Component Analysis
  ##   Rockhlin et al, 2009
  ##   https://epubs.siam.org/doi/10.1137/080736417
  ##
  ## - Finding structure with randomness, Probabilistic algorithms
  ##   for constructing approximate matrix decomposition
  ##   Halko et al, 2011
  ##   http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf
  ##
  ## - A randomized algorithm for the decomposition of matrices
  ##   Martinsson et al, 2011
  ##   https://www.sciencedirect.com/science/article/pii/S1063520310000242
  ##
  ## - Randomized Algorithms for Low-Rank Matrix Factorizations:
  ##   Sharp performance bounds
  ##   Witten et al, 2013
  ##   https://arxiv.org/abs/1308.5697
  ##
  ## - Subspace Iteration Randomization and Singular Value Problems
  ##   Gu, 2014
  ##   https://epubs.siam.org/doi/10.1137/130938700
  ##
  ## - An implementation of a randomized algorithm
  ##   for principal component analysis
  ##   Szlam et al, 2014
  ##   https://arxiv.org/abs/1412.3510
  ##
  ## - Randomized methods for matrix computations
  ##   Martinsson, 2016
  ##   https://arxiv.org/abs/1607.01649
  ##
  ## - RSVDPACK: An implementation of randomized algorithms for
  ##   computing the singular value, interpolative, and CUR decompositions
  ##   of matrices on multi-core and GPU architectures
  ##   Voronin, 2016
  ##   https://arxiv.org/abs/1502.05366
  ##
  ## - Randomized Matrix Decompositions using R
  ##   Erichson, 2016
  ##   https://arxiv.org/abs/1608.02148

  # Implementations:
  # - Scikit-learn:
  #     - https://github.com/scikit-learn/scikit-learn/blob/0.21.3/sklearn/utils/extmath.py#L230
  #     - Discussion: https://github.com/scikit-learn/scikit-learn/pull/5141
  # - fbpca (Tulloch, Tygert, Szlam paper):
  #     - https://github.com/facebook/fbpca
  #     - https://arxiv.org/abs/1412.3510
  # - rSVD (Erichson, Voronin):
  #     - https://github.com/erichson/rSVD
  #     - https://arxiv.org/abs/1608.02148
  # - RSVDPACK (Voronin, Martinsson):
  #     - https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes
  #     - https://arxiv.org/pdf/1502.05366.pdf

  # Checking correctness:
  # - TODO: fbpca Spectral Norm
  # - https://software.intel.com/en-us/articles/checking-correctness-of-lapack-svd-eigenvalue-and-one-sided-decomposition-routines

  # Defaults:
  # - nb_oversamples:
  #   - fbpca defaults to 2
  #   - sklearn defaults to 10
  #   - rSVD recommends 10 (page 14)
  #   - RSVDPACK recommends 5 (page 3) and range 5-10 (page 2)
  #
  # - nb_power_iter:
  #   - fbpca defaults to 2
  #   - sklearn uses 4 unless nb_components is less than 0.1 min(A.shape)
  #     in that case it uses 7
  #   - rSVD recommends 2 (page 14)
  #   - RSVDPACK recommends 2 (page 22)

  let k = n_components
  let m = A.shape[0]
  let n = A.shape[1]
  assert k <= min(m, n)

  # TODO: seed and RNG overload

  # The random test matrix Ω should have independent identically distributed
  # from some distribution which ensures that columns are linearly independent
  # with high-probability. For example: Gaussian, Uniform or Rademacher distribution.

  # The power iteration scheme can have its numerical stability improved
  # by orthogonalizing the sketch matrix Y between iteration.
  # QR is more accurate than LU but traditionally more costly
  # hence fbpca and sklearn defaults to LU (Facebook's scale of problems)
  #
  # According to Golub and Van Load, "Matrix Computations", 1996
  # and Higham "Functions of Matrices Theory and Computation", here are the following
  # decomposition costs
  #
  # | Decomposition / Factorization                     |     Flops (A n-by-n matrix)        |
  # | ------------------------------------------------- | -----------------------------------|
  # | LU factorization with partial pivoting (PA = LU)  |        2n³/3                       |
  # | Householder QR Factorization (A = QR)             | 2n²(m - n/3) for R                 |
  # |                                                   | 4(m²n - mn² + n³/3) for m x m Q    |
  # |                                                   | 2n²(m - n/3) for m x n Q           |
  # |                                                   | 2np(2m-n) for QB with m x p B      |
  # |                                                   | and Q in factored form             |
  #
  # In our case we are able to do QB with Q in factored form
  #
  # Memory efficiency can be determined:
  # A of shape [M, N]
  # L = k + nb_oversamples. Usually L << min(M, N)
  #
  # LU: N*L + M*L
  #     - A * Q    ([M,N] * [N,L])
  #     - A.T * Q' ([N,M] * [M,L])
  # QR: if we keep Q in factored form:
  #     - MxN (copy of A)
  #       but we do 2x copies of A per power iterations
  #       as input is mutated in-place
  #       and we need it in colMajor
  #     if we don't:
  #     - same as LU
  #
  # a scratchspace is needed anyway after QB decomposition so QR scratchspace is not an issue
  # A*Q or A.T*Q space is much lower than A
  var tau, scratchspace: seq[T]
  var Y, Z: Tensor[T]

  let L = k + n_oversamples                                    # Slight oversampling

  # SVD directly if nb_components within number of 25% of input dimension
  if L.float32 * 1.25 >= m.float32 or L.float32 * 1.25 >= n.float32:
    (result.U, result.S, result.Vh) = svd(A)
    result.U = result.U[_, 0..<k]
    result.S = result.S[0..<k]
    result.Vh = result.Vh[0..<k, _]
    return

  # We want to minimize the M or N dimension used in computation by transposing
  # There is a 2x-3x speed gap compared to not transposing appropriately
  # -----------------------------------------------------------------------------------------------------------
  # Ensure that A is contiguous
  let A = A.asContiguous(rowMajor, force=false)
  if m >= n:
    Y.newMatrixUninitColMajor(m, L)                                # Sketch Matrix ~ range samples
    Z.newMatrixUninitColMajor(n, L)
    tau.setLen(min(m, L))

    # QB decomposition ----------------------------------------------------------------------------------------
    var Q = randomTensor[T]([n, L], 1.T)                           # Sampling matrix Ω of shape [N, L]
    gemm(1.T, A, Q, 0.T, Y)                                        # Y = A*Q                  - [M, L]
    # -- Power Iterations (Optional) --------------------------------------------------------------------------
    for _ in 0 ..< n_power_iters:                                  # perform optional subspace iterations
      lu_permuted_inplace(Y)                                       # Y = lu(Y)
      gemm(1.T, A.transpose(), Y, 0.T, Z)                          # Z = A.T * Y              - [N, L]
      lu_permuted_inplace(Z)                                       # Z = lu(Z)
      gemm(1.T, A, Z, 0.T, Y)                                      # Y = A * Z                - [M, L]
    # -- Power Iterations -------------------------------------------------------------------------------------
    Q = Y                                                          #                          - [M, L]
    geqrf(Q, tau, scratchspace)                                    # Q = qr(Y) - orthonormal basis for samples
    orgqr(Q, tau, scratchspace)                                    # extract Q; next line project to low-dimensional space
    Z.newMatrixUninitColMajor(L, n)                                # Reuse Z buffer (shape NxL) to store B (shape LxN)
    gemm(1.T, Q.transpose(), A, 0.T, Z)                            # B = Q.T * A              - [L,M]*[M,N] -> [L, N]
    # QB decomposition ----------------------------------------------------------------------------------------
    gesdd(Z, result.U, result.S, result.Vh, scratchspace)          # U, S, Vh = svd(B)
    result.U = Q * result.U                                        # U = Q * Û - Recover left singular vectors

    # Extract k components from oversampled L
    result.U = result.U[_, 0 ..< k]
    result.S = result.S[0 ..< k]
    result.Vh = result.Vh[0 ..< k, _]
    return

  # -----------------------------------------------------------------------------------------------------------
  else:
    Y.newMatrixUninitColMajor(n, L)                                # Sketch Matrix ~ range samples
    Z.newMatrixUninitColMajor(m, L)                                # Temp space for final B matrix and normalized power iterations
    tau.setLen(min(L, n))

    # QB decomposition ----------------------------------------------------------------------------------------
    var Q = randomTensor[T]([m, L], 1.T)                           # Sampling matrix Ω  - [M, L]
    gemm(1.T, Q.transpose(), A, 0.T, Y)                            # Y = Q.T * A        - [N, L]
    # -- Power Iterations (Optional) --------------------------------------------------------------------------
    for _ in 0 ..< n_power_iters:                                  # perform optional subspace iterations
      lu_permuted_inplace(Y)                                       # Y = lu(Y)
      gemm(1.T, A, Y, 0.T, Z)                                      # Z = A.T * Y        - [M,N]*[N,L] -> [M, L]
      lu_permuted_inplace(Z)                                       # Z = lu(Z)
      gemm(1.T, A.transpose(), Z, 0.T, Y)                          # Y = A * Z          - [N,M]*[M,L] -> [N, L]
    # -- Power Iterations -------------------------------------------------------------------------------------
    Q = Y                                                          #                    - [N, L]
    geqrf(Q, tau, scratchspace)                                    # Q = qr(Y)  - orthonormal basis for samples
    orgqr(Q, tau, scratchspace)                                    # extract Q; next line project to low-dimensional space
    gemm(1.T, A, Q, 0.T, Z)                                        # B = A * Q          - [M,N]*[N,L] -> [M, L]
    # QB decomposition ----------------------------------------------------------------------------------------
    gesdd(Z, result.U, result.S, result.Vh, scratchspace)          # U, S, Vh = svd(B)
    result.Vh = result.Vh * Q.transpose()                          # Vh = V * Q.T - Recover right singular vectors

    # Extract k components from oversampled L
    result.U = result.U[_, 0 ..< k]
    result.S = result.S[0 ..< k]
    result.Vh = result.Vh[0 ..< k, _]
    return

# TODO: auto-rank / rank revealing SVD
# -------------------------------------------
## - A randomized blocked algorithm for efficiently
##   computing rank-revealing factorizations of matrices
##   Martinsson et al, 2015
##   https://arxiv.org/abs/1503.07157
##
## - Efficient randomized algorithms for
##   adaptive low-rank factorizations of large matrices
##   Gu et al, 2016
##   https://www.researchgate.net/publication/304641525_Efficient_randomized_algorithms_for_adaptive_low-rank_factorizations_of_large_matrices
