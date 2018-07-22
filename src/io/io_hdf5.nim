# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor,
  strutils, strformat, ospaths,
  nimhdf5

# constant which we use to keep track of how many tensors are stored
# in a given H5 file without having to visit the whole H5 file.
# We create an attribute on the root group with this name and increment
# for each tensor.
const NumTensorStored = "numTensorsStored"

proc read_hdf5*[T: SomeNumber](hdf5Path: string): Tensor[T] {.noInit.} =
  ## Reads a .h5 file (written by arraymancer) and returns a tensor of the
  ## specified type.
  ## If the tensor is stored in a different type in the file, it will be
  ## converted.
  ##
  ## Input:
  ##   - The path to a HDF5 file as a string
  ## Output:
  ##   - A tensor
  discard

proc write_hdf5*[T: SomeNumber](t: Tensor[T],
                                hdf5Path: string,
                                name = "",
                                group = "") =
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
  var
    h5f = H5File(hdf5Path, "rw")
    grpName = ""
    dsetName = ""
  if group.len > 0:
    grpName = group
  if name.len > 0:
    dsetName = name
  else:
    # create a generic name based on tensor rank and number of
    # tensors stored in file
    # get num tensors in file (int32 should be plenty I guess...)
    var numTensors = 0'u32
    if NumTensorStored in h5f.attrs:
      numTensors = h5f.attrs[NumTensorStored, uint32]
    dsetName = &"Tensor_{t.rank}_{numTensors}"

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

  # close file
  let err = h5f.close()
  if err != 0:
    # TODO: raise? echo?
    echo "WARNING: could not properly close H5 file. Error code: ", err
    #raise newException(HDF5LibraryError,  "WARNING: could not properly close " &
    #  "H5 file. Error code: ", err)
