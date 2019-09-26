# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack,
  ./overload,
  ../../tensor/tensor

# Auxiliary functions from Lapack
# ----------------------------------

overload(laswp, slaswp)
overload(laswp, dlaswp)

proc laswp*(a: var Tensor, pivot_indices: openarray[int32], pivot_from: static int32) =
  ## Apply A = P * A
  ## where P is a permutation matrix, represented pivot_indices of rows.
  ##
  ## A is a matrix of shape MxN.
  ## A is permuted in-place

  # http://www.netlib.org/lapack/explore-html/d1/db5/group__real_o_t_h_e_rauxiliary_gacb14404955e1b301d7877892a3c83f3d.html#gacb14404955e1b301d7877892a3c83f3d

  assert a.rank == 2
  assert a.is_F_contiguous()

  let
    k = min(a.shape[0], a.shape[1]).int32
    n = k
    lda = a.shape[0].int32  # Assumes colMajor
    k1 = 1'i32              # First element of pivot_indices that will be used for row interchange
    k2 = k                  # Last element of pivot_indices that will be used for row interchanged
    incx = pivot_from.int32 # 1: multiply by Permutation on the right
                            # -1: multiply by permutation on the left

  assert k == pivot_indices.len

  laswp(n.unsafeAddr, a.get_data_ptr, lda.unsafeAddr, k1.unsafeAddr, k2.unsafeAddr,
        pivot_indices[0].unsafeAddr, incx.unsafeAddr)
