# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.


import
  ../tensor/tensor, ./io_stream_readers,
  ../tensor/private/p_init_cpu,
  ../tensor/backend/memory_optimization_hints,
  os, streams, strscans, strformat, parseutils

func get_parser_metadata[T](header_raw: string):
  tuple[parser: proc(stream: FileStream): T {.nimcall.}, shape: MetadataArray, layout: OrderType] =

  var
    npy_type: string
    npy_fortran: string
    npy_shape: string

  let matched = scanf(header_raw, "{'descr': '$+', 'fortran_order': $+, 'shape': $+, }", npy_type, npy_fortran, npy_shape)
  doAssert matched, &"Error while parsing .npy header {header_raw}"

  block: # Check type and convert
    case npy_type
    of "<f4": result.parser = proc(x: FileStream): T = x.readFloat32LE.T
    of ">f4": result.parser = proc(x: FileStream): T = x.readFloat32BE.T
    of "<f8": result.parser = proc(x: FileStream): T = x.readFloat64LE.T
    of ">f8": result.parser = proc(x: FileStream): T = x.readFloat64BE.T
    of "<i4": result.parser = proc(x: FileStream): T = x.readInt32LE.T
    of ">i4": result.parser = proc(x: FileStream): T = x.readInt32BE.T
    of "<i8": result.parser = proc(x: FileStream): T = x.readInt64LE.T
    of ">i8": result.parser = proc(x: FileStream): T = x.readInt64BE.T
    of "<u4": result.parser = proc(x: FileStream): T = x.readUInt32LE.T
    of ">u4": result.parser = proc(x: FileStream): T = x.readUInt32BE.T
    of "<u8": result.parser = proc(x: FileStream): T = x.readUInt64LE.T
    of ">u8": result.parser = proc(x: FileStream): T = x.readUInt64BE.T
    else:
      raise newException(ValueError, &"Numpy type not supported: {npy_type} in file.")

  block: # Get the shape
    var
      char_idx = 0
      shape_idx = 0
    doAssert npy_shape[char_idx] == '('
    inc char_idx

    while true:
      case npy_shape[char_idx]
      of {'0'..'9'}:
        char_idx += npy_shape.parseInt(result.shape[shape_idx], char_idx)
        inc shape_idx
        inc result.shape.len
      of ' ', ',':
        inc char_idx
      of ')':
        break
      else:
        raise newException(ValueError, &"Invalid token '{npy_shape[char_idx]}' in numpy shape {npy_shape}")

  block: # Get C (row-major) or Fortran (col-major) order type
    if npy_fortran == "False":  result.layout = rowMajor
    elif npy_fortran == "True": result.layout = colMajor
    else:
      raise newException(ValueError, &"Invalid token '{npy_fortran}' in numpy description {header_raw}")

proc read_npy*[T](npyPath: string): Tensor[T] {.noInit.} =
  ## Reads a .npy file and returns a Tensor of the specified type.
  ## If the ndarray is stored in a different type inside the file, it will be converted.
  ##
  ## Input:
  ##   - The path to a numpy file as string
  ## Output:
  ##   - A tensor
  ##
  ## Only integer, unsigned integer and float ndarrays are supported at the moment.

  if unlikely(not existsFile(npyPath)):
    raise newException(IOError, &".npy file \"{npyPath}\" does not exist")

  let stream = newFileStream(npyPath, mode = fmRead)

  # Check magic string
  var magic_string: array[6, char]
  discard stream.readData(magic_string.addr, 6)

  doAssert magic_string == ['\x93', 'N', 'U', 'M', 'P', 'Y'],
    "This file is not a Numpy file"

  # Check version
  let version: tuple[major, minor: uint8] = (stream.readUint8, stream.readUint8)
  doAssert (version == (1'u8, 0'u8)) or (version == (2'u8, 0'u8)),
    "Unsupported .npy format. Arraymancer can only load 1.0 or 2.0 files " &
    &"but the file {npyPath} is at version {version}."

  # Get header
  let header_len =  if version.major == 1: stream.readUInt16LE.int
                    else:                  stream.readUInt32LE.int

  var header_raw = newString(header_len)
  discard stream.readData(header_raw[0].addr, header_len)

  # Get the raw data parser and metadata.
  let (parser, shape, layout) = get_parser_metadata[T](header_raw)

  # Read the data
  tensorCpu(shape, result, layout)
  result.storage.Fdata = newSeqUninitialized[T](result.size)

  withMemoryOptimHints()
  let r_ptr {.restrict.}= result.dataArray

  for i in 0..<result.size:
    r_ptr[i] = stream.parser
