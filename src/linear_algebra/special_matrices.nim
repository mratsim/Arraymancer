# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../tensor/tensor

proc hilbert*(n: int, T: typedesc[SomeFloat]): Tensor[T] =
  ## Generates an Hilbert matrix of shape [N, N]
  result = newTensorUninit[T]([n, n])
  let R = result.unsafe_raw_buf()

  # Reminder: The canonical Hilbert matrix
  #           assumes that i, j starts at 1 instead of 0

  # Base algorithm that exploits symmetry and avoid division half the time
  # for i in 0 ..< n:
  #   for j in i ..< n:
  #     R[i*n+j] = 1 / (i.T + j.T + 1)
  #     R[j*n+i] = R[i*n+j]

  const Tile = 32
  for i in countup(0, n-1, Tile):
    for j in countup(0, n-1, Tile):
      for ii in i ..< min(i+Tile, n):
        for jj in j ..< min(j+Tile, n):
          # Result is triangular and we go through rows first
          if jj >= ii:
            R[ii*n+jj] = 1 / (ii.T + jj.T + 1)
          else:
            R[ii*n+jj] = R[jj*n+ii]

  # TODO: scipy has a very fast hilbert matrix generation
  #       via Hankel Matrix and fancy 2D indexing
