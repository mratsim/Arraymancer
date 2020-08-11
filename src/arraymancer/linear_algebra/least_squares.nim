# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import  ../tensor,
        ./helpers/least_squares_lapack

proc least_squares_solver*[T: SomeFloat](a, b: Tensor[T]):
  tuple[
    solution: Tensor[T],
    residuals:  Tensor[T],
    matrix_rank: int,
    singular_values: Tensor[T]
  ] {.noInit.}=

  gelsd(a, b,
        result.solution,
        result.residuals,
        result.singular_values,
        result.matrix_rank)
