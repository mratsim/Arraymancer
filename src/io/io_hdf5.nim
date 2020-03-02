# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor,
  ../laser/tensor/[initialization, allocator],
  strutils, strformat, os, options,
  nimhdf5

# constant which we use to keep track of how many tensors are stored
# in a given H5 file without having to visit the whole H5 file.
# We create an attribute on the root group with this name and increment
# for each tensor.
const NumTensorStored = "numTensorsStored"

func parseNameAndGroup(h5f: var H5FileObj,
                       toWrite: static[bool],
                       name, group: Option[string],
                       number: Option[int] = none(int)):
                      tuple[dsetName, grpName: string, numTensors: uint32] =
  ## helper proc to parse the name and group arguments
  ## of write/read_hdf5
  ## Input:
  ##   - name: name of the tensor dataset in the file
  ##   - group: location to store the tensor in
  ##   - number = if no name and group given, the number of the
  ##     generic tensor to read
  ## Output:
  ##   - (string, string): tuple of dataset name, group name
  ##     correctly parsed according to user desired name /
  ##     generic name

  if group.isSome:
    result.grpName = group.get
  if name.isSome:
    result.dsetName = name.get
  else:
    # create a generic name based on tensor rank and number of
    # tensors stored in file
    # get num tensors in file (uint32 should be plenty I guess...)
    result.numTensors = 0'u32

    if NumTensorStored in h5f.attrs:
      result.numTensors = h5f.attrs[NumTensorStored, uint32]

    when toWrite:
      if number.isSome and number.get < result.numTensors.int:
        let num = number.get
        result.dsetName = &"Tensor_{num}"
      else:
        result.dsetName = &"Tensor_{result.numTensors}"
    else:
      if number.isSome and number.get < result.numTensors.int:
        let num = number.get
        result.dsetName = &"Tensor_{num}"
      elif result.numTensors.int > 0:
        # if we read without specific tensor, read the last written
        result.dsetName = &"Tensor_{result.numTensors - 1}"

proc read_hdf5*[T: SomeNumber](h5f: var H5FileObj,
                               name, group: Option[string],
                               number: Option[int]): Tensor[T] {.noInit.} =
  ## Reads a .h5 file (written by arraymancer) and returns a tensor of the
  ## specified type.
  ## If the tensor is stored in a different type in the file, it will be
  ## converted.
  ##
  ## Input:
  ##   - The H5 file object we read from
  ##   - A non-generic name of the tensor to read
  ##   - A group different from the root group in which tensor is stored
  ##   - if generic names are used the `number`-th tensor to read
  ## Output:
  ##   - A tensor
  # get name of the correct tensor
  let (dsetName, grpName, _) = h5f.parseNameAndGroup(false, name, group, number)

  # check whether dataset exists in file
  if not h5f.isDataset(grpName / dsetName):
    raise newException(ValueError, "Given name `" & grpName / dsetName & "` does not " &
      "correspond to an existing tensor in the file: " & h5f.name)

  var h5dset = h5f[(grpName / dsetName).dset_str]

  # get the meta data from the attributes
  let shape = h5dset.attrs["shape", seq[int]]
  let rank  = h5dset.attrs["rank", int]
  let size  = h5dset.attrs["size", int]

  # since the datatype of the resulting tensor is not necessarily
  # the same as the actual data in the H5 file (we may want to convert
  # it), get a converter proc for it
  let convertTo = h5dset.convertType(T)
  # finally convert seq to tensor and reshape

  # Read the data
  block:
    var size: int
    if h5dset.attrs["is_C_contiguous", string] == "true":
      result.initTensorMetadata(shape, size, rowMajor)
    else:
      result.initTensorMetadata(shape, size, colMajor)
    result.storage.allocCpuStorage(size)

  let tmp = convertTo(h5dset)
  result.copyFromRaw(tmp[0].addr, tmp.len)

  assert shape == h5dset.shape
  assert shape == @(result.shape)
  assert rank == result.rank
  assert size == result.size

proc read_hdf5*[T: SomeNumber](h5f: var H5FileObj,
                              name, group = "",
                              number = -1): Tensor[T] {.noInit, inline.} =
  ## wrapper around the real `read_hdf5` to provide a nicer interface
  ## without having to worry about `some` and `none`
  let
    nameOpt = if name.len > 0: some(name) else: none(string)
    groupOpt = if group.len > 0: some(group) else: none(string)
    numberOpt = if number >= 0: some(number) else: none(int)
  result = read_hdf5[T](h5f, nameOpt, groupOpt, numberOpt)

proc read_hdf5*[T: SomeNumber](hdf5Path: string,
                               name, group = "",
                               number = -1): Tensor[T] {.noInit, inline.} =
  ## convenience wrapper around `read_hdf5` with `var H5DataSet` argument.
  ## opens the given H5 file for reading and then calls the read proc
  withH5(hdf5Path, "r"):
    # template opens file with read access as injected `h5f` and closes
    # file after actions
    result = read_hdf5[T](h5f, name, group, number)

proc write_hdf5*[T: SomeNumber](h5f: var H5FileObj,
                                t: Tensor[T],
                                name, group: Option[string]) =
  ## Exports a tensor to a hdf5 file
  ## To keep this a simple convenience proc, the tensor is stored
  ## in the root group of the hdf5 file under a generic name.
  ## If the `name` argument is given, the tensor is stored under this
  ## name instead. If the `group` argument is given the tensor is
  ## stored in that group instead of the root group.
  ## Note: if no `name` is given, we need to visit the whole file
  ## to check for existing tensors. This will introduce a small
  ## overhead!
  ##
  ## Input:
  ##   - The tensor to write
  ##   - The H5 file object we write to
  ##   - An optional name for the dataset of the tensor in the file
  ##     Useful to store multiple tensors in a single HDF5 file
  ##   - An optional name for a `group` to store the tensor in

  let (dsetName, grpName, numTensors) = h5f.parseNameAndGroup(true, name, group)

  var dset = h5f.create_dataset(grpName / dsetName,
                                @(t.shape),
                                dtype = T)

  # make sure the tensor is contiguous
  let tCont = t.asContiguous
  # write tensor data
  dset.unsafeWrite(tCont.get_data_ptr, tCont.size)

  # now write attributes of tensor
  dset.attrs["rank"] = tCont.rank
  dset.attrs["shape"] = @(tCont.shape)
  dset.attrs["size"] = tCont.size
  # workaround since we can't write bool attributes
  dset.attrs["is_C_contiguous"] = if tCont.is_C_contiguous: "true" else: "false"

  # write new number of tensors stored
  h5f.attrs[NumTensorStored] = numTensors + 1

proc write_hdf5*[T: SomeNumber](h5f: var H5FileObj,
                                t: Tensor[T],
                                name, group = "") {.inline.} =
  ## wrapper around the real `write_hdf5` to provide a nicer interface
  ## without having to worry about `some` and `none`
  let
    nameOpt = if name.len > 0: some(name) else: none(string)
    groupOpt = if group.len > 0: some(group) else: none(string)
  h5f.write_hdf5(t, nameOpt, groupOpt)

proc write_hdf5*[T: SomeNumber](t: Tensor[T],
                                hdf5Path: string,
                                name, group = "") {.inline.} =
  ## convenience wrapper around `write_hdf5` with `var H5DataSet` argument.
  ## opens the given H5 file for writing and then calls the write proc
  withH5(hdf5Path, "rw"):
    # template opens file with write access as injected `h5f` and closes
    # file after actions
    h5f.write_hdf5(t, name, group)
