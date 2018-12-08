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


import  streams, endians, os, httpClient, strformat, future, sequtils, ospaths,
        ../tensor/tensor, ../io/io_stream_readers, ./util,
        zip/gzipfiles

type Mnist = tuple[
  train_images: Tensor[uint8],
  test_images: Tensor[uint8],
  train_labels: Tensor[uint8],
  test_labels: Tensor[uint8],
]
const MNISTFilenames = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
  ]

proc read_mnist_images(stream: Stream): Tensor[uint8] {.noInit.}=
  ## Load MNIST images into a Tensor[uint8] from a stream
  ## Input:
  ##   - A stream of MNIST image data
  ##
  ## Returns:
  ##   - A tensor of images with shape (N, H, W)
  ##     - N, number of images
  ##     - H, height
  ##     - W, width
  defer: stream.close

  let magic_number = stream.readInt32BE
  doAssert magic_number == 2051'i32, "This file is not a MNIST images file."

  let
    n_imgs = stream.readInt32BE.int
    n_rows = stream.readInt32BE.int
    n_cols = stream.readInt32BE.int

  result = newTensorUninit[uint8](n_imgs, n_rows, n_cols)
  discard stream.readData(result.get_data_ptr, result.size)

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
  if not existsFile(imgsPath):
    raise newException(IOError, "MNIST images file \"" & imgsPath & "\" does not exist")

  let stream = newGzFileStream(imgsPath, mode = fmRead)
  return read_mnist_images(stream)

proc read_mnist_labels*(stream: Stream): Tensor[uint8] {.noInit.}=
  ## Load MNIST labels into a Tensor[uint8] from a file
  ## Input:
  ##   - A stream of MNIST labels data
  ##
  ## Returns:
  ##   - A tensor of labels with shape (N)
  ##     - N, number of images
  defer: stream.close

  let magic_number = stream.readInt32BE
  doAssert magic_number == 2049'i32, "This file is not a MNIST labels file."

  let
    n_labels = stream.readInt32BE.int

  result = newTensorUninit[uint8](n_labels)
  discard stream.readData(result.get_data_ptr, result.size)

proc read_mnist_labels*(labelsPath: string): Tensor[uint8] {.noInit.}=
  ## Load MNIST labels into a Tensor[uint8] from a file
  ## Input:
  ##   - A path to a MNIST labels file
  ##
  ## Returns:
  ##   - A tensor of labels with shape (N)
  ##     - N, number of images
  ## MNIST data can be downloaded here: http://yann.lecun.com/exdb/mnist/
  if not existsFile(labelsPath):
    raise newException(IOError, "MNIST labels file \"" & labelsPath & "\" does not exist")

  let stream = newGzFileStream(labelsPath, mode = fmRead)
  return read_mnist_labels(stream)

func mnistFilesPath(cache_dir: string): array[4, string] =
  for idx, val in result.mpairs:
    val = cache_dir / MNISTFileNames[idx]

proc download_mnist_files(files: array[4, string]) =
  ## Download the MNIST files from http://yann.lecun.com/exdb/mnist/
  ## It will download the files to the current directory.
  const mnist_domain = "http://yann.lecun.com/exdb/mnist/"
  let client = newHttpClient()

  for idx, f in files:
    let url = fmt"{mnist_domain}{MNISTFilenames[idx]}"
    client.downloadFile(url, f)

proc delete_mnist_files(files: array[4, string]) =
  ## Deletes the downloaded MNIST files.
  for f in files:
    discard tryRemoveFile(f)

proc load_mnist*(cache = true): Mnist =
  ## Loads the MNIST dataset into a tuple with fields:
  ## - train_images
  ## - train_labels
  ## - test_images
  ## - test_labels
  ##
  ## Use the cache argument (bool) as false to cleanup the files each time.
  ##
  ## The cache by default will be in "~/.cache/arraymancer" on Unix
  ## and "%USERNAME%/.cache/arraymancer" on Windows, yhis can be changed with
  ## the XDG_CACHE_HOME environment variable.
  ##
  ## This proc will:
  ## - download the files if necessary
  ## - unzip them
  ## - load into a tuple
  ## - delete the downloaded files if cache is false

  let
    cache_dir = get_cache_dir()
    files = cache_dir.mnistFilesPath

  if not files.all(x => x.existsFile):
    create_cache_dirs_if_necessary()
    download_mnist_files(files)

  # Training
  result.train_images = read_mnist_images(files[0])
  result.train_labels = read_mnist_labels(files[1])

  # Testing
  result.test_images = read_mnist_images(files[2])
  result.test_labels = read_mnist_labels(files[3])

  if not cache:
    delete_mnist_files(files)
