# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import nimhdf5
import unittest, os, options

import sequtils


template withFile(filename: string, actions: untyped): untyped =
  block:
    actions
    removeFile(filename)


suite "[IO] HDF5 .h5 file support":

  const test_write_file = "./build/test_hdf5_write.h5"
  const tensorName = "TestTensor"
  const groupName = "subgroup"
  const nestedGroups = "subgroup/nested"
  const name4D = "happy4DTensor"

  # Note: whatever the read type we convert it to int in the end.
  let expected_1d = [1'i64, 2, 3, 4].toTensor
  let expected_1d_float = [1'f64, 2, 3, 4].toTensor
  let expected_2d = [[1'i64, 2, 3, 4], [5'i64, 6, 7, 8]].toTensor
  let expected_2d_float = [[1'f64, 2, 3, 4], [5'f64, 6, 7, 8]].toTensor
  let expected_4d = [[[[1'i64, 2, 3, 4],
                       [5'i64, 6, 7, 8]],
                      [[1'i64, 2, 3, 4],
                       [5'i64, 6, 7, 8]]],
                     [[[1'i64, 2, 3, 4],
                       [5'i64, 6, 7, 8]],
                      [[1'i64, 2, 3, 4],
                       [5'i64, 6, 7, 8]]]].toTensor
  echo expected_4d.shape


  # test the HDF5 interface via the following:
  # - write tensor to file
  # - read tensor back, compare with input tensor
  # - read tensor back as different type
  # - write tensor to specific name and group, read back
  # - write several tensors to one file, read them back
  # - ?

  test "[IO] Writing Arraymancer tensor to HDF5 file":

    withFile(test_write_file):
      expected_1d.write_hdf5(test_write_file)

      # read back
      let a = read_hdf5[int64](test_write_file)
      check a == expected_1d

    withFile(test_write_file):
      # write an integer dataset
      expected_1d.write_hdf5(test_write_file)

      # read back as float dataset
      let a = read_hdf5[float64](test_write_file)
      check a == expected_1d_float

    withFile(test_write_file):
      # write an integer dataset
      expected_2d.write_hdf5(test_write_file, name = tensorName)

      # read back as float dataset
      let a = read_hdf5[float64](test_write_file, name = tensorName)
      check a == expected_2d_float

    withFile(test_write_file):
      expected_2d.write_hdf5(test_write_file)


      # read back
      let a = read_hdf5[int64](test_write_file)
      check a == expected_2d

    withFile(test_write_file):
      expected_4d.write_hdf5(test_write_file, name = name4D)

      # read back
      let a = read_hdf5[int64](test_write_file, name = name4D)
      check a == expected_4d

    withFile(test_write_file):
      # write several tensors one after another, read them back
      echo "Now more one"
      expected_1d.write_hdf5(test_write_file)
      expected_1d_float.write_hdf5(test_write_file)
      expected_2d.write_hdf5(test_write_file)
      expected_2d_float.write_hdf5(test_write_file)
      expected_4d.write_hdf5(test_write_file)

      let a = read_hdf5[int64](test_write_file, number = 0)
      check a == expected_1d
      let b = read_hdf5[float64](test_write_file, number = 1)
      check b == expected_1d_float
      let c = read_hdf5[int64](test_write_file, number = 2)
      check c == expected_2d
      let d = read_hdf5[float64](test_write_file, number = 3)
      check d == expected_2d_float
      let e = read_hdf5[int64](test_write_file, number = 4)
      check e == expected_4d
