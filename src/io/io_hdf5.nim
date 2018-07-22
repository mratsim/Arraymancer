# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor,
  strutils, strformat, ospaths, options,
  nimhdf5

# constant which we use to keep track of how many tensors are stored
# in a given H5 file without having to visit the whole H5 file.
# We create an attribute on the root group with this name and increment
# for each tensor.
const NumTensorStored = "numTensorsStored"

proc parseNameAndGroup(h5f: var H5FileObj,
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

  # TODO: clean up this proc!
  # maybe disallow no args at all?
  # use Option[T] for number arg

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

proc read_hdf5*[T: SomeNumber](hdf5Path: string,
                               name, group: Option[string],
                               number: Option[int]): Tensor[T] {.noInit.} =
  ## Reads a .h5 file (written by arraymancer) and returns a tensor of the
  ## specified type.
  ## If the tensor is stored in a different type in the file, it will be
  ## converted.
  ##
  ## Input:
  ##   - The path to a HDF5 file as a string
  ##   - A non-generic name of the tensor to read
  ##   - A group different from the root group in which tensor is stored
  ##   - if generic names are used the `number`-th tensor to read
  ## Output:
  ##   - A tensor
  # open hdf5 file
  var h5f = H5File(hdf5Path, "r")
  # get name of the correct tensor
  let (dsetName, grpName, _) = h5f.parseNameAndGroup(false, name, group, number)

  var h5dset = h5f[(grpName / dsetName).dset_str]

  # get the meta data from the attributes
  let shape = h5dset.attrs["shape", seq[int]]
  let rank  = h5dset.attrs["rank", int]
  let size  = h5dset.attrs["size", int]
  let is_C_contiguous =
    if h5dset.attrs["is_C_contiguous", string] == "true":
      true
    else:
      false

  # since the datatype of the resulting tensor is not necessarily
  # the same as the actual data in the H5 file (we may want to convert
  # it), get a converter proc for it
  let convertTo = h5dset.convertType(T)
  # finally convert seq to tensor and reshape
  result = convertTo(h5dset).toTensor.reshape(shape)

  # TODO: take these out...
  assert shape == h5dset.shape
  assert shape == @(result.shape)
  assert rank == result.rank
  assert size == result.size

  # close file
  let err = h5f.close()
  if err != 0:
    # TODO: raise? echo?
    echo "WARNING: could not properly close H5 file. Error code: ", err

proc read_hdf5*[T: SomeNumber](hdf5Path: string,
                               name, group = "",
                               number = -1): Tensor[T] {.noInit, inline.} =
  ## wrapper around the real `read_hdf5` to provide a nicer interface
  ## without having to worry about `some` and `none`
  let
    nameOpt = if name.len > 0: some(name) else: none(string)
    groupOpt = if group.len > 0: some(group) else: none(string)
    numberOpt = if number >= 0: some(number) else: none(int)
  result = read_hdf5[T](hdf5Path, nameOpt, groupOpt, numberOpt)

proc write_hdf5*[T: SomeNumber](t: Tensor[T],
                                hdf5Path: string,
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
  ## TODO: introduce a "num tensors stored" like attribute in the
  ## h5 file, which we write. Instead of visiting the whole file
  ## just access that single attribute.
  ##
  ## Input:
  ##   - The tensor to write
  ##   - The path to a HDF5 file as a string (will be created)
  ##   - An optional name for the dataset of the tensor in the file
  ##     Useful to store multiple tensors in a single HDF5 file
  ##   - An optional name for a `group` to store the tensor in

  # create hdf5 file
  var h5f = H5File(hdf5Path, "rw")

  let (dsetName, grpName, numTensors) = h5f.parseNameAndGroup(true, name, group)

  var dset = h5f.create_dataset(grpName / dsetName,
                                @(t.shape),
                                dtype = T)
  # write tensor data
  # TODO: for efficiency we'd want to hand the whole tensor and in
  # nimhdf5 access its data as the raw data ptr. Modify tensor
  # functions in nimhdf5!
  dset.unsafeWrite(t.get_data_ptr, t.size)

  # now write attributes of tensor
  dset.attrs["rank"] = t.rank
  dset.attrs["shape"] = @(t.shape)
  dset.attrs["size"] = t.size
  # workaround since we can't write bool attributes
  dset.attrs["is_C_contiguous"] = if t.is_C_contiguous: "true" else: "false"

  # write new number of tensors stored
  h5f.attrs[NumTensorStored] = numTensors + 1

  # close file
  let err = h5f.close()
  if err != 0:
    # TODO: raise? echo?
    echo "WARNING: could not properly close H5 file. Error code: ", err
    #raise newException(HDF5LibraryError,  "WARNING: could not properly close " &
    #  "H5 file. Error code: ", err)

proc write_hdf5*[T: SomeNumber](t: Tensor[T],
                                hdf5Path: string,
                                name, group = "") {.inline.} =
  ## wrapper around the real `write_hdf5` to provide a nicer interface
  ## without having to worry about `some` and `none`
  let
    nameOpt = if name.len > 0: some(name) else: none(string)
    groupOpt = if group.len > 0: some(group) else: none(string)
  write_hdf5[T](t, hdf5Path, nameOpt, groupOpt)
