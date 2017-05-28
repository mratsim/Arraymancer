# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

proc gemm_macro_kernel[T](mc, nc, kc: int,
                        alpha: T,
                        beta: T,
                        C: var seq[T], offC: int,
                        incRowC, incColC: int,
                        ibuf_A: var array[MCKC, T],
                        ibuf_B: var array[KCNC, T],
                        ibuf_C: var array[MRNR, T]) =
  let mp = (mc+MR-1) div MR
  let np = (nc+NR-1) div NR

  let mod_mr = mc mod MR
  let mod_nr = nc mod NR

  var mr: int
  var nr: int

  for j in 0 ..< np:
    nr = if (j != np-1 or mod_nr == 0): NR
         else: mod_nr
    for i in 0 ..< mp:
      mr = if (i != mp-1 or mod_mr == 0): MR
           else: mod_mr

      if (mr==MR and nr==NR):
        gemm_micro_kernel(kc, alpha,
                          ibuf_A, i*kc*MR,
                          ibuf_B, j*kc*NR,
                          beta,
                          C, i*MR*incRowC+j*NR*incColC + offC,
                          incRowC, incColC)
      else:
        gemm_micro_kernel(kc, alpha,
                          ibuf_A, i*kc*MR,
                          ibuf_B, j*kc*NR,
                          0.T,
                          ibuf_C, 0,
                          1, MR)
        gescal(mr, nr, beta,
               C, i*MR*incRowC+j*NR*incColC + offC,
               incRowC, incColC)
        geaxpy(mr, nr,
               1.T,
               ibuf_C,
               1, MR,
               C, i*MR*incRowC+j*NR*incColC + offC,
               incRowC, incColC)
