# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../tensor
import ./helpers/triangular
import std / [sequtils, bitops, strformat]

proc hilbert*(n: int, T: typedesc[SomeFloat]): Tensor[T] =
  ## Generates an Hilbert matrix of shape [N, N]
  result = newTensorUninit[T]([n, n])
  let R = result.unsafe_raw_buf()

  # Reminder: The canonical Hilbert matrix
  #           assumes that i, j starts at 1 instead of 0

  # Base algorithm that exploits symmetry and avoid division half the time
  # for i in 0 ..< n:
  #   for j in i ..< n:
  #     R[i*n+j] = 1 / (i.T + j.T + 1)
  #     R[j*n+i] = R[i*n+j]

  const Tile = 32
  for i in countup(0, n-1, Tile):
    for j in countup(0, n-1, Tile):
      for ii in i ..< min(i+Tile, n):
        for jj in j ..< min(j+Tile, n):
          # Result is triangular and we go through rows first
          if jj >= ii:
            R[ii*n+jj] = 1 / (ii.T + jj.T + 1)
          else:
            R[ii*n+jj] = R[jj*n+ii]

  # TODO: scipy has a very fast hilbert matrix generation
  #       via Hankel Matrix and fancy 2D indexing

proc vandermonde*[T](x: Tensor[T], orders: Tensor[SomeNumber]): Tensor[float] =
  ## Returns a "Generalized" Vandermonde matrix of the input `x` over the given `orders`
  ##
  ## A "generalized" Vandermonde matrix consists of the input `x` split into
  ## multiple rows where each row contains the powers of `x_i` elevated to each
  ## of the elements of the `orders` tensor.
  ##
  ## `V_ij = x_i ^ order_j`
  ##
  ## Inputs:
  ##   - x: The input tensor `x` (which must be a rank-1 tensor)
  ##   - orders: The "exponents" tensor (which must also be a rank-1 tensor)
  ## Result:
  ##   - The constructed Vandermonde matrix
  assert x.squeeze.rank == 1
  let x = x.squeeze.asType(float)
  result = newTensorUninit[float]([x.size.int, orders.size])
  for i, ax in enumerateAxis(result, axis = 1):
    result[_, i] = (x ^. orders[i].float).unsqueeze(axis = 1)

proc vandermonde*[T](x: Tensor[T], order: int = -1, increasing = true): Tensor[float] =
  ## Returns a Vandermonde matrix of the input `x` up to the given `order`
  ##
  ## A Vandermonde matrix consists of the input `x` split into multiple
  ## rows where each row contains all powers of `x_i` from 0 to `order-1`
  ## (by default) or from `order-1` down to 0 (if `increasing` is set to false).
  ##
  ## `V_ij = x_i ^ order_j`
  ##
  ## where `order_j` runs from 0 to `order-1` or from `order-1` down to 0.
  ##
  ## Inputs:
  ##   - x: The input tensor `x` (which must be a rank-1 tensor)
  ##   - order: the order of the Vandermonde matrix. If not provided,
  ##     (or non positive) the order is set to the size of `x`.
  ##   - increasing: If true, the powers of `x_i` will run from 0 to `order-1`,
  ##                 otherwise they will run from `order-1` down to 0.
  ## Result:
  ##   - The constructed Vandermonde matrix
  let order = if order > 0: order else: x.size.int
  let orders = if increasing:
    arange(order.float)
  else:
    arange((order - 1).float, -1.0, -1.0)
  vandermonde(x, orders)

proc vandermonde*(order: int, increasing = true): Tensor[float] {.inline.} =
  ## Returns a "square" Vandermonde matrix of the given `order`
  ##
  ## A square Vandermonde matrix is a Vandermonde matrix of the given order
  ## whose input tensor is `arange(order)`.
  ##
  ## `V_ij = x_i ^ order_j`
  ##
  ## where `order_j` runs from 0 to `order-1` or from `order-1` down to 0.
  ##
  ## Inputs:
  ##   - order: the order of the Vandermonde matrix.
  ##   - increasing: If true, the powers of `x_i` will run from 0 to `order-1`,
  ##                 otherwise they will run from `order-1` down to 0.
  ## Result:
  ##   - The constructed Vandermonde matrix
  let x = arange(order.float)
  vandermonde(x, order, increasing = increasing)

proc vander*[T](x: Tensor[T], order: int = -1, increasing = false): Tensor[float] {.inline.} =
  ## Same as `vandermonde` but with `increasing` set to false by default
  ##
  ## This procedure is meant for compatibility with numpy, whose `vander()`
  ## function defaults to increasing = false (as opposed to Arraymancer's
  ## `vandermonde`, which defaults to increasing = true).
  ##
  ## See also: `vandermonde`
  vandermonde(x, order, increasing = increasing)

proc vander*(order: int = -1, increasing = false): Tensor[float] {.inline.} =
  ## Same as the square `vandermonde` but with `increasing` set to false by default
  ##
  ## This procedure is meant for compatibility with numpy, whose `vander()`
  ## function defaults to increasing = false (as opposed to Arraymancer's
  ## `vandermonde`, which defaults to increasing = true).
  ##
  ## See also: `vandermonde`
  vandermonde(order, increasing = increasing)

proc diagonal*[T](a: Tensor[T], k = 0, anti = false): Tensor[T] {.noInit.} =
  ## Gets the k-th diagonal (or anti-diagonal) of a matrix
  ##
  ## Input:
  ##      - A matrix (which can be rectangular)
  ##      - k: The index k of the diagonal that will be extracted. The default is 0 (i.e. the main diagonal).
  ##        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
  ##      - anti: If true, get the k-th "anti-diagonal" instead of the k-th regular diagonal.
  ## Result:
  ##      - A copy of the diagonal elements as a rank-1 tensor
  bind `&`
  assert a.rank == 2, "diagonal() only works on matrices"
  assert k < a.shape[0], &"Diagonal index ({k=}) exceeds the output matrix height ({a.shape[0]})"
  assert k < a.shape[1], &"Diagonal index ({k=}) exceeds the output matrix width ({a.shape[1]})"
  let size = min(a.shape[0], a.shape[1]) - abs(k)
  result = newTensor[T](size)

  if anti:
    if k >= 0:
      let size = min(a.shape[0] - abs(k), a.shape[1])
      result = newTensor[T](size)
      for i in 0 ..< size:
        result[i] = a[a.shape[0]-1-(i+k), i]
    else:
      let size = min(a.shape[0], a.shape[1] - abs(k))
      result = newTensor[T](size)
      for i in 0 ..< size:
        result[i] = a[a.shape[0]-1-i, i-k]
  else:
    if k >= 0:
      let size = min(a.shape[0], a.shape[1] - abs(k))
      result = newTensor[T](size)
      for i in 0 ..< size:
        result[i] = a[i, i+k]
    else:
      let size = min(a.shape[0]-abs(k), a.shape[1])
      result = newTensor[T](size)
      for i in 0 ..< size:
        result[i] = a[i-k, i]

proc set_diagonal*[T](a: var Tensor[T], d: Tensor[T], k = 0, anti = false) =
  ## Sets a diagonal of a matrix (in place)
  ##
  ## Input:
  ##      - The matrix that will be changed in place.
  ##      - Rank-1 tensor containg the elements that will be copied into the selected diagonal.
  ##      - k: The index k of the diagonal that will be changed. The default is 0 (i.e. the main diagonal).
  ##        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
  ##      - anti: If true, set the k-th "anti-diagonal" instead of the k-th regular diagonal.
  bind `&`
  assert a.rank == 2, "set_diagonal() only works on matrices"
  assert d.rank == 1, "The diagonal passed to set_diagonal() must be a rank-1 tensor"
  assert k < a.shape[0], &"Diagonal index ({k=}) exceeds input matrix height ({a.shape[0]})"
  assert k < a.shape[1], &"Diagonal index ({k=}) exceeds input matrix width ({a.shape[1]})"
  if anti:
    if k >= 0:
      when compileOption("boundChecks"):
        let size = min(a.shape[0] - abs(k), a.shape[1])
        doAssert size == d.size, &"Diagonal input size ({d.size}) does not match the {k}-th upper anti-diagonal size ({size})"
      for i in 0 ..< d.size:
        a[a.shape[0]-1-(i+k), i] = d[i]
    else:
      when compileOption("boundChecks"):
        let size = min(a.shape[0], a.shape[1] - abs(k))
        doAssert size == d.size, &"Diagonal input size ({d.size}) does not match the {-k}-th lower anti-diagonal size ({size})"
      for i in 0 ..< d.size:
        a[a.shape[0]-1-i, i-k] = d[i]
  else:
    if k >= 0:
      when compileOption("boundChecks"):
        let size = min(a.shape[0], a.shape[1] - abs(k))
        doAssert size == d.size, &"Diagonal input size ({d.size}) does not match the {k}-th upper diagonal size ({size})"
      for i in 0 ..< d.size:
        a[i, i+k] = d[i]
    else:
      when compileOption("boundChecks"):
        let size = min(a.shape[0] - abs(k), a.shape[1])
        doAssert size == d.size, &"Diagonal input size ({d.size}) does not match the {-k}-th lower diagonal size ({size})"
      for i in 0 ..< d.size:
        a[i-k, i] = d[i]

proc with_diagonal*[T](a: Tensor[T], d: Tensor[T], k = 0, anti = false): Tensor[T] {.noInit.} =
  ## Copy the input matrix, changing one of its diagonals into the elements of the rank-1 input tensor d
  ##
  ## Input:
  ##      - The matrix that will copied into the output.
  ##      - Rank-1 tensor containg the elements that will be copied into the selected diagonal.
  ##      - k: The index k of the diagonal that will be changed. The default is 0 (i.e. the main diagonal).
  ##        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
  ##      - anti: If true, set the k-th "anti-diagonal" instead of the k-th regular diagonal.
  result = a
  set_diagonal(result, d, k, anti=anti)

proc diag*[T](d: Tensor[T], k = 0, anti = false): Tensor[T] {.noInit.} =
  ## Creates a new square diagonal matrix from an rank-1 input tensor
  ##
  ## Input:
  ##      - Rank-1 tensor containing the elements of the diagonal
  ##      - k: The index of the diagonal that will be set. The default is 0.
  ##        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
  ##      - anti: If true, set the k-th "anti-diagonal" instead of the k-th regular diagonal.
  ## Result:
  ##      - The constructed, square diagonal matrix
  doAssert d.rank == 1, "Diagonal must be a rank-1 tensor"
  let size = d.size + abs(k)
  result = zeros[T](size, size)
  result.set_diagonal(d, k=k, anti=anti)

proc identity*[T](n: int): Tensor[T] {.noInit.} =
  ## Return an identity matrix (i.e. 2-D tensor) of size `n`
  ##
  ## The identity matrix is a square 2-D tensor with ones on the main diagonal and zeros elsewhere.
  ## This is basically the same as calling `eye(n, n)`.
  ##
  ## Input:
  ##      - Number of rows / columns in the output.
  ## Result:
  ##      - The constructed indentity 2-D tensor
  result = diag(ones[T](n))

proc eye*[T](shape: varargs[int]): Tensor[T] {.noInit.} =
  ## Return a 2-D tensor with ones on the diagonal and zeros elsewhere
  ##
  ## Input:
  ##      - The shape of the output matrix
  ## Result:
  ##      - The constructed, rank-2 diagonal tensor
  doAssert shape.len == 2, "eye() takes exactly two arguments"
  result = zeros[T](shape)
  result.set_diagonal(ones[T](min(shape)))

proc tri*[T](shape: Metadata, k: static int = 0, upper: static bool = false): Tensor[T] {.noInit.} =
  ## Return a 2-D tensor with ones at and below the given diagonal and zeros elsewhere
  ##
  ## Inputs:
  ##      - The (rank-2) shape of the output matrix.
  ##      - k: The sub-diagonal at and below which the tensor will be filled with ones.
  ##           The default is 0.
  ##      - upper: If true, the tensor will be filled with ones at and above the given
  ##               diagonal. The default is false.
  ## Result:
  ##      - The constructed, rank-2 triangular tensor.
  bind `&`
  assert shape.len == 2, &"tri() requires a rank-2 shape as it's input but a shape of rank {shape.len} was passed"
  assert k < shape[0], &"tri() received a diagonal index ({k=}) which exceeds the output matrix height ({shape[0]})"
  assert k < shape[1], &"tri() received a diagonal index ({k=}) which exceeds the output matrix width ({shape[1]})"

  result = ones[T](shape)
  when upper:
    result = triu(result, k = k)
  else:
    result = tril(result, k = k)

proc tri*[T](shape_ax1, shape_ax0: int, k: static int = 0, upper: static bool = false): Tensor[T] {.noInit, inline.} =
  ## Return a 2-D tensor with ones at and below the given diagonal and zeros elsewhere
  ##
  ## Inputs:
  ##      - The shape of the output matrix.
  ##      - k: The sub-diagonal at and below which the tensor will be filled with ones.
  ##           The default is 0.
  ##      - upper: If true, the tensor will be filled with ones at and above the given
  ##               diagonal. The default is false.
  ## Result:
  ##      - The constructed, rank-2 triangular tensor
  tri[T](toMetadata(shape_ax1, shape_ax0), k = k, upper = upper)

# Also export the tril and triu functions which are also part of numpy's API
# and which are implemented in helpers/triangular.nim
export tril, triu

proc circulant*[T](t: Tensor[T], axis = -1, step = 1): Tensor[T] {.noInit.} =
  ## Construct a circulant matrix from a rank-1 tensor
  ##
  ## A circulant matrix is a square matrix in which each column (or row) is
  ## a cyclic shift of the previous column (or row).
  ##
  ## By default this function cirulates over the columns of the output which
  ## are rotated down by 1 element over the previous column, but this behavior
  ## can be changed by using the `axis` and `step` arguments.
  ##
  ## Inputs:
  ## - A rank-1 Tensor
  ## - axis: The axis along which the circulant matrix will be constructed.
  ##         Defaults to -1 (i.e. the columns, which are the last axis).
  ## - step: The number of elements by which the input tensor will be shifted
  ##         each time. Defaults to 1.
  ## Result:
  ## - The constructed circulant matrix
  ##
  ## Example:
  ## ```nim
  ## echo circulant([1, 3, 6])
  ## Tensor[system.int] of shape "[3, 3]" on backend "Cpu"
  ## # |1      6     3|
  ## # |3      1     6|
  ## # |6      3     1|
  ## ```
  result = newTensor[T](t.len, t.len)
  if axis == 0:
    for n in 0 ..< t.len:
      result[n, _] = t.roll(step * n)
  else:
    for n in 0 ..< t.len:
      result[_, n] = t.roll(step * n)

proc toeplitz*[T](c, r: Tensor[T]): Tensor[T] {.noInit.} =
  ## Construct a Toeplitz matrix
  ##
  ## A Toeplitz matrix has constant diagonals, with c as its first column
  ## and r as its last row (note that `r[0]` is ignored but _should_ be the
  ## same as `c[^1])`. This is similar to (but different than) a Hankel matrix,
  ## which has constant anti-diagonals instead.
  ##
  ## Inputs:
  ## - c: The first column of the Toeplitz matrix
  ## - r: The last row of the Toeplitz matrix (note that `r[0]` is ignored)
  ## Result:
  ## - The constructed Toeplitz matrix
  ##
  ## Notes:
  ## - There is a version of this procedure that takes a single argument `c`,
  ##   in which case `r` is set to `c.conjugate`.
  ## - Use the `hankel` procedure to generate a Hankel matrix instead.
  ##
  ## Example:
  ## ```nim
  ## echo toeplitz([1, 3, 6], [9, 10, 11, 12])
  ## # Tensor[system.int] of shape "[3, 4]" on backend "Cpu"
  ## # |1     10    11    12|
  ## # |3      1    10    11|
  ## # |6      3     1    10|
  ## ```
  result = newTensor[T](c.len, r.len)
  let t = c.flatten[_|-1].append(r[1.._])
  for n in 0 ..< c.len:
    let idx_start = c.len - 1 - n
    result[n, _] = t[(idx_start)..<(idx_start + r.len)]

proc toeplitz*[T](c: Tensor[T]): Tensor[T] {.noInit, inline.} =
  ## Construct a square Toeplitz matrix from a single tensor
  ##
  ## A Toeplitz matrix has constant diagonals. This version of this procedure
  ## gets a single tensor as its input. The input tensor is used as-is to set
  ## the first column of the Toeplitz matrix, and is conjugated to set the
  ## first row of the Toeplitz matrix.
  ##
  ## Inputs:
  ## - c: The first column of the Toeplitz matrix. It is also the complex
  ##      conjugate of the first row of the Toeplitz matrix.
  ## Result:
  ## - The constructed square Toeplitz matrix
  ##
  ## Notes:
  ## - There is a version of this procedure that takes two arguments
  ##   (`c` and `r`),
  ## - While there is also a single input `hankel` procedure, its behavior
  ##   is quite different, since it sets `r` to an all zeros tensor instead.
  ##
  ## Examples:
  ## ```nim
  ## echo toeplitz([1, 3, 6])
  ## # Tensor[system.int] of shape "[3, 3]" on backend "Cpu"
  ## # |1      3     6|
  ## # |3      1     3|
  ## # |6      3     1|
  ##
  ## echo toeplitz([1.0+2.0.im, 3.0+4.0.im, 6.0+7.0.im].toTensor)
  ## # Tensor[complex.Complex64] of shape "[3, 3]" on backend "Cpu"
  ## # |(1.0, 2.0)     (3.0, -4.0)    (6.0, -7.0)|
  ## # |(3.0, 4.0)      (1.0, 2.0)    (3.0, -4.0)|
  ## # |(6.0, 7.0)      (3.0, 4.0)     (1.0, 2.0)|
  ## ```
  when T is Complex:
    toeplitz(c, c.conjugate)
  else:
    toeplitz(c, c)

proc hankel*[T](c, r: Tensor[T]): Tensor[T] {.noInit.} =
  ## Construct a Hankel matrix.
  ##
  ## A Hankel matrix has constant anti-diagonals, with c as its first column
  ## and r as its last row (note that `r[0]` is ignored but _should_ be the same
  ## as `c[^1])`. This is similar to a Toeplitz matrix, which has constant
  ## diagonals instead.
  ##
  ## Inputs:
  ## - c: The first column of the Hankel matrix
  ## - r: The last row of the Hankel matrix (note that `r[0]` is ignored)
  ## Result:
  ## - The constructed Hankel matrix
  ##
  ## Notes:
  ## - There is a version of this procedure that takes a single argument `c`,
  ##   in which case `r` is set to all zeroes, resulting in a "triangular"
  ##   Hankel matrix.
  ## - Use the `toeplitz` procedure to generate a Toeplitz matrix instead.
  ##
  ## Example:
  ## ```nim
  ## echo hankel([1, 3, 6], [9, 10, 11, 12])
  ## # Tensor[system.int] of shape "[3, 4]" on backend "Cpu"
  ## # |1      3     6    10|
  ## # |3      6    10    11|
  ## # |6     10    11    12|
  ## ```
  result = newTensor[T](c.len, r.len)
  let t = c.flatten.append(r[1.._])
  for n in 0 ..< r.len:
    result[_, n] = t[n..<(n+c.len)]

proc hankel*[T](c: Tensor[T]): Tensor[T] {.noInit, inline.} =
  ## Construct a "triangular" Hankel matrix (i.e. with zeros on its last row)
  ##
  ## The "triangular" Hankel matrix is a Hankel matrix in which all the items
  ## below the main anti-diagonal are set to 0. This is equivalent to creating
  ## a regular Hankel metrix in which the `c` argument is set to all zeros.
  ##
  ## Inputs:
  ## - c: The first column of the Hankel matrix
  ## Result:
  ## - The constructed "triangular" Hankel matrix
  ##
  ## Notes:
  ## - There is a version of this procedure that takes two arguments
  ##   (`c` and `r`),
  ## - While there is also a single input `toeplitz` procedure, its behavior
  ##   is quite different, since it sets `r` to the complex conjugate of `c`.
  ##
  ## Example:
  ## ```nim
  ## echo hankel([1, 3, 6], [9, 10, 11, 12])
  ## # Tensor[system.int] of shape "[3, 3]" on backend "Cpu"
  ## # |1      3     6|
  ## # |3      6     0|
  ## # |6      0     0|
  ## ```
  hankel(c, zeros_like(c))

type MeshGridIndexing* = enum xygrid, ijgrid

proc meshgrid*[T](t_list: varargs[Tensor[T]], indexing = MeshGridIndexing.xygrid):
    seq[Tensor[T]] {.noinit.} =
  ## Return a sequence of coordinate matrices from coordinate vectors.
  ##
  ## Make N-D coordinate tensors for vectorized evaluations of N-D scalar/vector
  ## fields over N-D grids, given one-dimensional coordinate tensors x1, x2,..., xn.
  ##
  ## Inputs:
  ## - xi: The coordinate tensors. Each vector must be a rank-1 tensor.
  ## - indexing: Cartesian (`xygrid`, default) or matrix (`ijgrid`) indexing of the output.
  ##             The indexing mode only affects the first 2 output Tensors.
  ##             In the 2-D case with inputs of length M and N, the outputs are of shape
  ##             (N, M) for `xygrid` indexing and (M, N) for `ijgrid` indexing.
  ##             In the 3-D case with inputs of length M, N and P, outputs are of shape
  ##             (N, M, P) for `xygrid` indexing and (M, N, P) for `ijgrid` indexing.
  ## Result:
  ## - List of N meshgrid N-dimensional Tensors
  ##   For tensors x1, x2,..., xn with lengths Ni=len(xi), returns (N1, N2, N3,..., Nn)
  ##   shaped tensors if indexing=`ijgrid` or (N2, N1, N3,..., Nn) shaped tensors if
  ##   indexing=`xygrid` with the elements of xi repeated to fill the matrix along
  ##   the first dimension for x1, the second for x2 and so on.
  ##
  ## Notes:
  ## - This function follows and implements the `numpy.meshgrid` API.

  let t_list = if indexing == MeshGridIndexing.xygrid:
      # In xy mode, the first two dimensions are swapped before broadcasting
      @[t_list[1], t_list[0]] & t_list[2..^1]
    else:
      t_list.toSeq
  result = newSeq[Tensor[T]](t_list.len)
  var out_shape = t_list.mapIt(it.size.int)
  let dims = repeat(1, t_list.len)
  var n = 0
  for t in t_list:
    var item_shape = dims
    item_shape[n] = t.size.int
    result[n] = broadcast(t.reshape(item_shape), out_shape)
    inc n
  if indexing == MeshGridIndexing.xygrid:
    # In xy mode, we must swap back the first two dimensions after broadcast
    result = @[result[1], result[0]] & result[2..^1]

func int2bit_impl(value: int, n: int, msbfirst: bool): seq[bool] {.noinit, inline.} =
  ## Convert an integer into binary sequence of size `n`
  let largest_index = min(n-1, sizeof(value) * 8)
  result = newSeq[bool](n)
  if msbfirst:
    for k in countdown(largest_index, 0):
      result[n-1-k] = testBit(value, k)
  else:
    for k in countup(0, largest_index):
      result[k] = testBit(value, k)

proc int2bit*(value: int, n: int, msbfirst = true): Tensor[bool] =
  ## Convert an integer into a "bit" tensor of size `n`
  ##
  ## A "binary" tensor is a tensor containing the bits that represent
  ## the input integer.
  ##
  ## Inputs:
  ##   - value: The input integer
  ##   - n: The size of the output tensor. No check is done to ensure that `n`
  ##        is large enough to represent `value`.
  ##   - msbfirst: If `true` (the default), the first element will be the most
  ##               significant bit (i.e. the msb will be first). Otherwise the
  ##               least significant bit will be first.
  ## Result:
  ##   - The constructed "bit" tensor
  ## Notes:
  ##   This is similar to Matlab's `int2bit` (except that Matlab's version
  ##   fills the bit tensors column-wise, while this fills them row-wise).
  ##   It is also similar to (but more flexible than) numpy's `unpackbits`.
  ## Examples:
  ## .. code:: nim
  ##   echo int2bit(12, 5)
  ##   # Tensor[system.bool] of shape "[5]" on backend "Cpu"
  ##   #    false     true     true    false    false
  ##   echo int2bit(12, 5, msbfirst = false)
  ##   # Tensor[system.bool] of shape "[5]" on backend "Cpu"
  ##   #    false    false     true     true    false

  int2bit_impl(value, n, msbfirst = msbfirst).toTensor

proc int2bit*[T: SomeInteger](t: Tensor[T], n: int, msbfirst = true): Tensor[bool] {.noinit.} =
  ## Convert an integer tensor of rank-X into a "bit" tensor of rank X+1
  ##
  ## The "bit" tensor corresponding to an integer tensor is a tensor in
  ## which each integer element is replaced with its binary representation
  ## (of size `n`). This requires increasing the rank of the output tensor
  ## by 1, making the size of its last axis equal to `n`.
  ##
  ## Inputs:
  ##   - value: The input tensor (of rank X)
  ##   - n: The size of the output tensor. No check is done to ensure that `n`
  ##        is large enough to represent `value`.
  ##   - msbfirst: If `true` (the default), elements of the input tensor will
  ##               be converted to bits by placing the most significant bit
  ##               first. Otherwise the least significant bit will be first.
  ## Result:
  ##   - The constructed "bit" tensor (of rank X+1 and shape `t.shape` + [n])
  ## Notes:
  ##   This is similar to Matlab's `int2bit` (except that Matlab's version
  ##   fills the bit tensors column-wise, while this fills them row-wise).
  ##   It is also similar to (but more flexible than) numpy's `unpackbits`.
  ## Examples:
  ## .. code:: nim
  ##   echo int2bit([12, 6], 5)
  ##   # Tensor[system.bool] of shape "[2, 5]" on backend "Cpu"
  ##   # |false     true     true    false    false|
  ##   # |false    false     true     true    false|
  ##   echo int2bit([12, 6], msbfirst = false)
  ##   # Tensor[system.bool] of shape "[2, 5]" on backend "Cpu"
  ##   # |false    false     true     true    false|
  ##   # |false     true     true    false    false|
  var shape = t.shape
  shape.add(n)
  result = newTensor[bool](t.size * n)
  for idx in 0 ..< t.size:
    result[n * idx ..< n * (idx + 1)] = int2bit(t[idx], n, msbfirst = msbfirst)
  result = result.reshape(shape)
