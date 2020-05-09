# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer, ../testutils
import unittest, os

suite "[IO] Numpy .npy file support":

  const folder = "./tests/io/npy_files/"
  const test_write_file = "./build/test_numpy_write.npy"

  # Note: whatever the read type we convert it to int in the end.
  let expected_1d = [1'i64, 2, 3, 4].toTensor
  let expected_2d = [[1'i64, 2, 3, 4], [5'i64, 6, 7, 8]].toTensor

  test "[IO] Reading numpy files with numeric ndarrays":

    # Reading from an int64 - little endian
    block:
      let a = read_npy[int64](folder & "int.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int64](folder & "int_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int64](folder & "int_2D_f.npy")
      check: a == expected_2d

    # Reading from a float32 - little endian (converted to int)
    block:
      let a = read_npy[int64](folder & "f32LE.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int64](folder & "f32LE_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int64](folder & "f32LE_2D_f.npy")
      check: a == expected_2d

    # Reading from an uint64 - big endian (converted to int)
    block:
      let a = read_npy[int64](folder & "u64BE.npy")
      check: a == expected_1d

    block:
      let a = read_npy[int64](folder & "u64BE_2D_c.npy")
      check: a == expected_2d

    block:
      let a = read_npy[int64](folder & "u64BE_2D_f.npy")
      check: a == expected_2d

  test "[IO] Arraymancer produces the same .npy files as Numpy":

    when system.cpuEndian == littleEndian:
      # int64 - littleEndian
      block:
        expected_1d.write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "int.npy")

      block:
        expected_2d.write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "int_2D_c.npy")

      block:
        expected_2d.asContiguous(colMajor, force = true).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "int_2D_f.npy")

      # float32 - littleEndian
      block:
        expected_1d.astype(float32).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "f32LE.npy")

      block:
        expected_2d.astype(float32).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "f32LE_2D_c.npy")

      block:
        expected_2d.astype(float32).asContiguous(colMajor, force = true).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "f32LE_2D_f.npy")

    else:
      # uint64 - bigEndian
      block:
        expected_1d.astype(uint64).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "u64BE.npy")

      block:
        expected_2d.astype(uint64).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "u64BE_2D_c.npy")

      block:
        expected_2d.astype(uint64).asContiguous(colMajor, force = true).write_npy(test_write_file)
        check: sameFileContent(test_write_file, folder & "u64BE_2D_f.npy")
