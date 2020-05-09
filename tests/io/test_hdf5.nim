# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer, ../testutils
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
  var expected_fortran = [[1'i64, 2, 3],
                          [4'i64, 5, 6],
                          [7'i64, 8, 9]].toTensor
  expected_fortran = expected_fortran.asContiguous(colMajor, force = true)

  test "[IO] Writing Arraymancer tensors to HDF5 file and reading them back":

    withFile(test_write_file):
      # write 1D tensor and read back
      expected_1d.write_hdf5(test_write_file)

      let a = read_hdf5[int64](test_write_file)
      check a == expected_1d

    withFile(test_write_file):
      # write an 1D integer tensor and read back as float
      expected_1d.write_hdf5(test_write_file)

      let a = read_hdf5[float64](test_write_file)
      check a == expected_1d_float

    withFile(test_write_file):
      # write 2d tensor and read back
      expected_2d.write_hdf5(test_write_file)

      let a = read_hdf5[int64](test_write_file)
      check a == expected_2d

    withFile(test_write_file):
      # write a 2D integer tensor and read back as float
      expected_2d.write_hdf5(test_write_file, name = tensorName)

      let a = read_hdf5[float64](test_write_file, name = tensorName)
      check a == expected_2d_float

    withFile(test_write_file):
      # write 4D tensor and read back
      expected_4d.write_hdf5(test_write_file, name = name4D)

      let a = read_hdf5[int64](test_write_file, name = name4D)
      check a == expected_4d

    withFile(test_write_file):
      # write tensor to subgroup and read back
      expected_2d.write_hdf5(test_write_file, group = groupName)

      let a = read_hdf5[int64](test_write_file, group = groupName)
      check a == expected_2d

    withFile(test_write_file):
      # write several tensors one after another, read them back
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

    withFile(test_write_file):
      # write several tensors without closing the file in between
      var h5f = H5file(test_write_file, "rw")

      h5f.write_hdf5(expected_1d)
      h5f.write_hdf5(expected_1d_float)
      h5f.write_hdf5(expected_2d)
      h5f.write_hdf5(expected_2d_float)
      h5f.write_hdf5(expected_4d)

      let a = read_hdf5[int64](h5f, number = 0)
      check a == expected_1d
      let b = read_hdf5[float64](h5f, number = 1)
      check b == expected_1d_float
      let c = read_hdf5[int64](h5f, number = 2)
      check c == expected_2d
      let d = read_hdf5[float64](h5f, number = 3)
      check d == expected_2d_float
      let e = read_hdf5[int64](h5f, number = 4)
      check e == expected_4d

      let err = h5f.close()
      if err != 0:
        assert false, "Could not close H5 file correctly!"

    withFile(test_write_file):
      # check whether Fortran order is recovered
      expected_fortran.write_hdf5(test_write_file)

      let a = read_hdf5[int64](test_write_file)
      check a == expected_fortran
      check a.is_F_contiguous

    withFile(test_write_file):
      # write named tensor into nested group, read back
      expected_2d.write_hdf5(test_write_file,
                             name = tensorName,
                             group = nestedGroups)

      let a = read_hdf5[int64](test_write_file,
                               name = tensorName,
                               group = nestedGroups)
      check a == expected_2d
