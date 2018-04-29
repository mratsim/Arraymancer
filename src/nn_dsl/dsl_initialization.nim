# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ./dsl_types

proc trainParamsConv2D(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  # 1. Create the object field
  var convConfig: ModelField

  convConfig.field_name = field_name
  convConfig.field_type = nnkBracketExpr.newTree(
      ident("Conv2DLayer"), self.subtype
    )

  # 2. Configure weight and bias
  let
    topo = self.topoTable.getOrDefault(field_name)
    sst = self.subtype[1] # we need to get the subsubtype float32/float64

    kshape = topo.c2d_kernel_shape

    w_shape = kshape
    b_shape = quote do: [`kshape`[0], 1, 1]

    w = quote do: randomTensor(`w_shape`, `sst`(1)) .- `sst`(0.5)
    b = quote do: randomTensor(`b_shape`, `sst`(1)) .- `sst`(0.5)

  convConfig.init_call = newStmtList()

  let ctx = self.context

  convConfig.init_call.add quote do:
    result.`field_name`.weight = `ctx`.variable(
      `w`, requires_grad = true # TODO allow freezing
    )

    result.`field_name`.bias = `ctx`.variable(
      `b`, requires_grad = true # TODO allow freezing
    )

  self.trainparams.add convConfig

proc trainParamsLinear(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  # 1. Create the object field
  var linearConfig: ModelField

  linearConfig.field_name = field_name
  linearConfig.field_type = nnkBracketExpr.newTree(
      ident("LinearLayer"), self.subtype
    )

  # 2. Configure weight and bias
  let
    topo = self.topoTable.getOrDefault(field_name)
    sst = self.subtype[1] # we need to get the subsubtype float32/float64

    in_shape = topo.in_shape
    out_shape = topo.out_shape

    w_shape = quote do: [`out_shape`, `in_shape`]
    b_shape = quote do: [1, `out_shape`]

    w = quote do: randomTensor(`w_shape`, `sst`(1)) .- `sst`(0.5)
    b = quote do: randomTensor(`b_shape`, `sst`(1)) .- `sst`(0.5)

  linearConfig.init_call = newStmtList()

  let ctx = self.context

  linearConfig.init_call.add quote do:
    result.`field_name`.weight = `ctx`.variable(
      `w`, requires_grad = true # TODO allow freezing
    )

    result.`field_name`.bias = `ctx`.variable(
      `b`, requires_grad = true # TODO allow freezing
    )

  self.trainparams.add linearConfig

proc genModelFieldInit*(self: Neuromancer) =

  self.trainparams = @[]

  for k, v in pairs(self.topoTable):
    case v.kind:
    of lkConv2D: self.trainParamsConv2D(k, v)
    of lkLinear: self.trainParamsLinear(k, v)
    else:
      discard
