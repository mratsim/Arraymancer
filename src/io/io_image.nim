# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  stb_image/read,
  stb_image/write,
  ../tensor/tensor

func whc_to_chw[T](img: Tensor[T]): Tensor[T] {.inline.}=
  ## Convert image from Width x Height x Channel convention
  ## to the Channel x Height x Width convention.
  img.permute(2, 1, 0)

func chw_to_whc[T](img: Tensor[T]): Tensor[T] {.inline.}=
  ## Convert image from Channel x Height x Width convention
  ## to the Width x Height x Channel convention.
  img.permute(2, 1, 0)

proc read_image*(filepath: string): Tensor[uint8] =
  ## Read an image file and loads it into a Tensor[uint8] of shape
  ## Channel x Height x Width. Channel is 1 for greyscale, 3 for RGB.
  ##
  ## Supports JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, PNM
  ## See stb_image https://github.com/nothings/stb/blob/master/stb_image.h
  ##
  ## Usage example with conversion to [0..1] float:
  ## .. code:: nim
  ##   let raw_img = read_image('path/to/image.png')
  ##   let img = raw_img.map_inline:
  ##     x.float32 / 255.0

  var width, height, channels: int
  let desired_channels = 0 # Channel autodetection

  let raw_pixels = load(filepath, width, height, channels, desired_channels)
  result = raw_pixels.toTensor.reshape(width, height, channels).whc_to_chw

# TODO should we add:
#   - Normalization:
#     - [0, 1] range  - img / 255.0
#     - [-1, 1] range - img * 2 / 255.0 - 1
#   - mean centering: (img - mean(dataset)) / stddev(dataset)
#     - do we substract a global mean
#     - or a mean per color channel
