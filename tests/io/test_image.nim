# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, os

suite "[IO] Numpy .npy file support":
  const imgpath = "./tests/io/images/nim_in_action_cover.jpg"

  test "[IO] Reading numpy files with numeric ndarrays":
    let img = read_image(imgpath)

    let channels = 3
    let width = 767
    let height = 964

    check: img.shape == [channels, height, width].toMetadataArray
