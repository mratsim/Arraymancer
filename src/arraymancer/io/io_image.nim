# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  stb_image/read,
  stb_image/write,
  ../tensor

func hwc_to_chw[T](img: Tensor[T]): Tensor[T] {.inline.}=
  ## Convert image from Width x Height x Channel convention
  ## to the Channel x Height x Width convention.
  img.permute(2, 0, 1)

func chw_to_hwc[T](img: Tensor[T]): Tensor[T] {.inline.}=
  ## Convert image from Channel x Height x Width convention
  ## to the Width x Height x Channel convention.
  img.permute(1, 2, 0)

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
  ##   let img = forEach x in raw_img:
  ##     x.float32 / 255.0

  var width, height, channels: int
  let desired_channels = 0 # Channel autodetection

  let raw_pixels = load(filepath, width, height, channels, desired_channels)
  result = raw_pixels.toTensor.reshape(height, width, channels).hwc_to_chw

proc read_image*(buffer: seq[byte]): Tensor[uint8] =
  ## Read an image from a buffer and loads it into a Tensor[uint8] of shape
  ## Channel x Height x Width. Channel is 1 for greyscale, 3 for RGB.
  ##
  ## Supports JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, PNM
  ## See stb_image https://github.com/nothings/stb/blob/master/stb_image.h
  ##

  # TODO: ideally this should also accept pointer + length
  # but nim-stb_image only accept seq[bytes] (and convert it to pointer + length internally)

  var width, height, channels: int
  let desired_channels = 0 # Channel autodetection

  let raw_pixels = load_from_memory(buffer, width, height, channels, desired_channels)
  result = raw_pixels.toTensor.reshape(width, height, channels).hwc_to_chw


template gen_write_image(proc_name: untyped): untyped {.dirty.}=

  proc proc_name*(img: Tensor[uint8], filepath: string) =
    ## Create an image file from a tensor

    var success = false
    let
      img = img.chw_to_hwc.asContiguous(rowMajor, force = true)
      h = img.shape[0]
      w = img.shape[1]
      c = img.shape[2]
    success = proc_name(filepath, w, h, c, toOpenArray(img.storage.raw_buffer, 0, img.size-1))

    doAssert success

gen_write_image(write_png)
gen_write_image(write_bmp)
gen_write_image(write_tga)

proc write_jpg*(img: Tensor[uint8], filepath: string, quality = 100) =
  ## Create a jpeg image file from a tensor

  var success = false
  let
    img = img.chw_to_hwc.asContiguous(rowMajor, force = true)
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
  success = write_jpg(filepath, w, h, c, toOpenArray(img.storage.raw_buffer, 0, img.size-1), quality)

  doAssert success
