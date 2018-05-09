# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest

suite "Input/Output":
  test "[IO] Reading numpy files with numeric ndarrays":

    # Note: whatever the read type we convert it to int in the end.
    let expected_1d = [1, 2, 3, 4].toTensor
    let expected_2d = [[1, 2, 3, 4], [5, 6, 7, 8]].toTensor
    const folder = "./tests/io/npy_files/"

    # Reading from a native int - native endianness
    block:
      let a = read_npy[int](folder & "int.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int](folder & "int_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int](folder & "int_2D_f.npy")
      check: a == expected_2d

    # Reading from a float32 - little endian (converted to int)
    block:
      let a = read_npy[int](folder & "f32LE.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int](folder & "f32LE_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int](folder & "f32LE_2D_f.npy")
      check: a == expected_2d

    # Reading from an uint64 - big endian (converted to int)
    block:
      let a = read_npy[int](folder & "u64BE.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int](folder & "u64BE_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int](folder & "u64BE_2D_f.npy")
      check: a == expected_2d
