# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import  ../tensor,
        ./helpers/least_squares_lapack

proc least_squares_solver*[T: SomeFloat](a, b: Tensor[T],
                                         rcond = -1.T):
  tuple[
    solution: Tensor[T],
    residuals:  Tensor[T],
    matrix_rank: int,
    singular_values: Tensor[T]
  ] {.noinit.} =
  ## Solves the given linear least squares problem:
  ##
  ## `minimize | Ax - y |`
  ##
  ## where the matrix `A` is our input tensor `a` and the resulting vector `y` is
  ## given by our input tensor `b`.
  ##
  ## `a` needs to be of shape `N x M`. `b` may either be of shape `N` or `N x K`, where
  ## `K` represents the number of solutions to search for. One solution for each `k_i` is
  ## returned.
  ##
  ## `rcond` is the condition for singular values to be considered zero,
  ## `s(i) <= rcond * s(i)` are treated as zero.
  ##
  ## If `rcond = -1` is used, it determines the size automatically (to the machine precision).
  gelsd(a, b,
        result.solution,
        result.residuals,
        result.singular_values,
        result.matrix_rank)
