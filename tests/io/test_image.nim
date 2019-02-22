# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, os

suite "[IO] Reading and writing images":
  const
    imgpath = "./tests/io/images/nim_in_action_cover.jpg"
    outpng = "./build/test_img_out.png"
    outjpg = "./build/test_img_out.jpg"
    rwpng = "./build/test_img_read_write.png"

  test "[IO] Reading images":
    let img = read_image(imgpath)

    let channels = 3
    let width = 767
    let height = 964

    check: img.shape == [channels, height, width].toMetadataArray

  test "[IO] Writing images":
    let img = read_image(imgpath)

    img.write_png(outpng)
    img.write_jpg(outjpg, quality = 70)

  test "[IO] Reading and Wriging images are equal":
    let
        gray_1_6_8 = @[
                  [
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [255,255,255,255,255,255,255,255],
                    [255,255,255,255,255,255,255,255],
                    [255,255,255,255,255,255,255,255],
                  ],
                ].toTensor.astype(uint8)
    check: gray_1_6_8.shape == @[1,6,8]

    gray_1_6_8.write_png(rwpng)
    let rwimg = read_image(rwpng)

    check: gray_1_6_8 == rwimg
