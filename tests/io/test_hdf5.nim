# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import nimhdf5
import unittest, os

suite "[IO] HDF5 .h5 file support":

  const test_write_file = "./build/test_hdf5_write.h5"

  # Note: whatever the read type we convert it to int in the end.
  let expected_1d = [1'i64, 2, 3, 4].toTensor
  let expected_2d = [[1'i64, 2, 3, 4], [5'i64, 6, 7, 8]].toTensor

  # test the HDF5 interface via the following:
  # - write tensor to file
  # - read tensor back, compare with input tensor
  # - read tensor back as different type
  # - write tensor to specific name and group, read back
  # - write several tensors to one file, read them back
  # - ?

  test "[IO] Writing Arraymancer tensor to HDF5 file":

    block:
      expected_1d.write_hdf5(test_write_file)

    block:
      expected_2d.write_hdf5(test_write_file)
