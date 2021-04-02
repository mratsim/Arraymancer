# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ./dsl_types, ./dsl_utils,
  ../tensor

proc out_shape_conv2d(in_shape: array[3, int], kernel: array[4, int], padding, strides: tuple[h, w: int]): array[3, int] {.noInit.}=
  ## Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
  ##   outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*upscaleA)+1) )/ convolutionStride;

  ## Input and result are of shape [C, H, W]
  ## Kernel of shape [C_out, C_in, h, w]

  # Reminder we don't consider N (bath size) in the topology
  template kH: int = kernel[2]
  template kW: int = kernel[3]
  template pH: int = padding.h
  template pW: int = padding.w
  template sH: int = strides.h
  template sW: int = strides.w

  template iH: int = in_shape[1]
  template iW: int = in_shape[2]
  template dH: int = 1 # dilation # TODO
  template dW: int = 1 # dilation

  result[0] = kernel[0]                                     # C
  result[1] = 1 + (iH + 2*pH - (((kH-1) * dH) + 1)) div sH  # H
  result[2] = 1 + (iW + 2*pW - (((kW-1) * dW) + 1)) div sW  # W

proc out_shape_maxpool2d(in_shape: array[3, int], kernel, padding, strides: tuple[h, w: int]): array[3, int] {.noInit.}=

  # Reminder we don't consider N (bath size) in the topology
  template C: int = in_shape[0]
  template H: int = in_shape[1]
  template W: int = in_shape[2]

  template kH: int = kernel.h
  template kW: int = kernel.w
  template pH: int = padding.h
  template pW: int = padding.w
  template sH: int = strides.h
  template sW: int = strides.w

  result[0] = C
  result[1] = (H + (2 * pH) - kH) div sH + 1
  result[2] = (W + (2 * pW) - kW) div sW + 1

proc out_shape_flatten*(in_shape: openarray[int]): int {.inline.}=
  ## Flatten a tensor shape (i.e. returns the product)
  ## A tensor of shape [1, 2, 3] will have a shape [1*2*3]
  ## when flattened
  # TODO: make that work only at compile-time on a custom TopoShape type
  #       to avoid conflicts with other libraries.
  assert in_shape.len != 0
  result = 1
  for val in in_shape:
    result *= val

proc isValidLayerSection(section: NimNode): bool =

  # Expected AST - cv1: Conv2D(20, 5, 5)
  # Call
  #   Ident "cv1"
  #   StmtList
  #     Call
  #       Ident "Conv2D"
  #       IntLit 1
  #       IntLit 20
  #       IntLit 5
  #       IntLit 5
  (section.kind == nnkCall) and
    (section[0].kind == nnkIdent) and
    (section[1].kind == nnkStmtlist) and
    (section[1].len == 1) and
    (section[1][0].kind == nnkCall)

template unknown(section: Nimnode) =
  error:
    lineInfo(section) &
      ": unknown neural network configuration section \"" &
      $section[0] & "\""

template incorrect(section: Nimnode) =
  error:
    lineInfo(section) &
      ": incorrect neural network configuration section \"" &
      $section[0] & "\""

proc topoFromInput(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # Initializes the ident --> (input, output) topology table with the input shapes

  # Call
  #   Ident "Input"
  #   Bracket
  #     IntLit 1
  #     IntLit 28
  #     IntLit 28

  if desc.len != 2:
    incorrect(desc) ## Placeholder to specify padding stride in the future

  self[ident] = LayerTopology(kind: lkInput,
                                in_shape: desc[1],
                                out_shape: desc[1])

proc topoFromConv2D(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # Call
  #   Ident "Conv2D"
  #   # Kernel (C_in,C_out, kH, kW)
  #   DotExpr
  #     Ident "x"
  #     Ident "out_shape"
  #   IntLit 20
  #   IntLit 5
  #   IntLit 5
  #   # Kernel strides & padding

  var padding, strides: NimNode

  if desc.len > 5:
    incorrect(desc) ## Placeholder to specify padding stride in the future
  else:
    padding = quote do: (0, 0)
    strides = quote do: (1, 1)

  var in_shape = self.replaceInputNodes(desc[1])
  in_shape = quote do: `in_shape`

  let
    c_out = desc[2]
    kH = desc[3]
    kW = desc[4]

  let kernel = quote do:
    # C_out, C_in, kH, kW
    [`c_out`, `in_shape`[0], `kH`, `kW`]

  let out_shape = quote do:
    out_shape_conv2d(`in_shape`, `kernel`, `padding`, `strides`)

  self[ident] = LayerTopology(kind: lkConv2D,
                                in_shape: in_shape,
                                out_shape: out_shape,
                                c2d_kernel_shape: kernel,
                                c2d_padding: padding,
                                c2d_strides: strides)

proc topoFromMaxPool2D(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # Call
  #   Ident "MaxPool2D"
  #   DotExpr
  #     Ident "cv1"
  #     Ident "out_shape"
  #   Par
  #     IntLit 2
  #     IntLit 2
  #   Par
  #     IntLit 0
  #     IntLit 0
  #   Par
  #     IntLit 2
  #     IntLit 2

  if desc.len != 5:
    incorrect(desc) ## Placeholder to specify padding stride in the future

  var in_shape = self.replaceInputNodes(desc[1])
  in_shape = quote do: `in_shape`

  let
    kernel = desc[2]
    padding = desc[3]
    strides = desc[4]

  let out_shape = quote do:
    out_shape_maxpool2d(`in_shape`, `kernel`, `padding`, `strides`)

  self[ident] = LayerTopology(kind: lkMaxPool2D,
                                in_shape: in_shape,
                                out_shape: out_shape,
                                m2d_kernel: kernel,
                                m2d_padding: padding,
                                m2d_strides: strides)

proc topoFromLinear(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # Call
  #   Ident "Linear"
  #   IntLit 500
  #   IntLit 10

  if desc.len != 3:
    incorrect(desc) ## Placeholder to specify padding stride in the future

  var in_shape = self.replaceInputNodes(desc[1])
  in_shape = quote do: `in_shape`

  self[ident] = LayerTopology(kind: lkLinear,
                                in_shape: in_shape,
                                out_shape: desc[2])

proc topoFromGCN(self: var TopoTable, ident: NimNode, desc: NimNode) =


  if desc.len != 3:
    incorrect(desc) ## Placeholder to specify padding stride in the future

  var in_shape = self.replaceInputNodes(desc[1])
  in_shape = quote do: `in_shape`

  self[ident] = LayerTopology(kind: lkGCN,
                                in_shape: in_shape,
                                out_shape: desc[2])

proc topoFromFlatten(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # Call
  #   Ident "Flatten"
  #   DotExpr
  #     Ident "mp2.out_shape"
  #     Ident "out_shape"

  if desc.len != 2:
    incorrect(desc)

  var in_shape = self.replaceInputNodes(desc[1])
  in_shape = quote do: `in_shape`

  let out_shape = quote do:
    out_shape_flatten(`in_shape`)

  self[ident] = LayerTopology(kind: lkFlatten,
                                in_shape: in_shape,
                                out_shape: out_shape)

proc topoFromGRU(self: var TopoTable, ident: NimNode, desc: NimNode) =

  # TODO: check that input

  # Call
  #   Ident "GRU"
  #   DotExpr             # [seq_len/time, batch_size, features]
  #     Ident "foo"
  #     Ident "out_shape"
  #   IntLit 4            # Hidden size
  #   IntLit 5            # Nb layers

  if desc.len != 4:
    incorrect(desc)

  # TODO: ensure single computation `in_shape`
  let in_shape = self.replaceInputNodes(desc[1])

  # TODO: support bidirectional
  let hidden_size = desc[2]
  let out_shape = nnkBracket.newTree(
      nnkBracketExpr.newTree(in_shape, newLit 0),
      nnkBracketExpr.newTree(in_shape, newLit 1),
      hidden_size
    )

  self[ident] = LayerTopology(kind: lkGRU,
                                in_shape: in_shape,
                                out_shape: out_shape,
                                gru_hidden_size: hidden_size,
                                gru_nb_layers: desc[3]
                                )

proc topoFromLayer(self: var TopoTable, ident: NimNode, desc: NimNode) =

  if eqIdent(desc[0], "Conv2D"):
    self.topoFromConv2D(ident, desc)
  elif eqIdent(desc[0], "MaxPool2D"):
    self.topoFromMaxPool2D(ident, desc)
  elif eqIdent(desc[0], "Linear"):
    self.topoFromLinear(ident, desc)
  elif eqIdent(desc[0], "GCN"):
    self.topoFromGCN(ident, desc)
  elif eqIdent(desc[0], "Input"):
    self.topoFromInput(ident, desc)
  elif eqIdent(desc[0], "Flatten"):
    self.topoFromFlatten(ident, desc)
  elif eqIdent(desc[0], "GRU"):
    self.topoFromGRU(ident, desc)
  else:
    unknown(desc)

proc topoFromLayers*(self: var TopoTable, layers: NimNode) =

  ## Add all layers and their known parameters to the table
  #

  for section in layers:
    if section.isValidLayerSection:
      assert section[0] notin self
      self.topoFromLayer(
        ident($section[0]),
        section[1][0]
        )
    else:
      unknown(section)
