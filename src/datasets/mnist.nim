# Copyright 2017 the Arraymancer contributors
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


# ##########################################################################
# Download and read images and labels from http://yann.lecun.com/exdb/mnist/


# Format:
#
# ⚠ 32 bit uint - Big Endian!
# Row-Major
# unsigned byte = uint8 = char

# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).


# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.


import  streams, endians, os,
        ../tensor/tensor

proc readInt32BE(stream: FileStream): int32 =
  var little_endian = stream.readInt32
  bigEndian32(addr result, addr little_endian)

proc read_mnist_images*(imgsPath: string): Tensor[uint8] {.noInit.}=
  ## Load MNIST images into a Tensor[uint8]
  ## Input:
  ##   - A path to a MNIST images file
  ##
  ## Returns:
  ##   - A tensor of images with shape (N, H, W)
  ##     - N, number of images
  ##     - H, height
  ##     - W, width
  ##
  ## MNIST data can be downloaded here: http://yann.lecun.com/exdb/mnist/
  ## It must be uncompressed before use. Download and decompression will be automated in the future.
  # Compressed reader pending https://github.com/nim-lang/zip/pull/20
  if not existsFile(imgsPath):
    raise newException(IOError, "MNIST images file \"" & imgsPath & "\" does not exist")

  let stream = newFileStream(imgsPath, mode = fmRead)

  let magic_number = stream.readInt32BE
  doAssert magic_number == 2051'i32, "This file is not a MNIST images file, did you forget to decompress it?"

  let
    n_imgs = stream.readInt32BE.int
    n_rows = stream.readInt32BE.int
    n_cols = stream.readInt32BE.int

  result = newTensorUninit[uint8](n_imgs, n_rows, n_cols)
  discard stream.readData(result.get_data_ptr, result.size)

proc read_mnist_labels*(labelsPath: string): Tensor[uint8] {.noInit.}=
  ## Load MNIST labels into a Tensor[uint8]
  ## Input:
  ##   - A path to a MNIST labels file
  ##
  ## Returns:
  ##   - A tensor of images with shape (N, H, W)
  ##     - N, number of images
  ##     - H, height
  ##     - W, width
  ##
  ## MNIST data can be downloaded here: http://yann.lecun.com/exdb/mnist/
  ## It must be uncompressed before use.  Download and decompression will be automated in the future.
  # Compressed reader pending https://github.com/nim-lang/zip/pull/20
  if not existsFile(labelsPath):
    raise newException(IOError, "MNIST labels file \"" & labelsPath & "\" does not exist")

  let stream = newFileStream(labelsPath, mode = fmRead)

  let magic_number = stream.readInt32BE
  doAssert magic_number == 2049'i32, "This file is not a MNIST labels file, did you forget to decompress it?"

  let
    n_labels = stream.readInt32BE.int

  result = newTensorUninit[uint8](n_labels)
  discard stream.readData(result.get_data_ptr, result.size)