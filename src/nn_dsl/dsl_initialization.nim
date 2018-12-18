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

    w = quote do: kaiming_normal(`w_shape`, `sst`)
    b = quote do: zeros[`sst`](`b_shape`)

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

    w = quote do: kaiming_normal(`w_shape`, `sst`)
    b = quote do: zeros[`sst`](`b_shape`)
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

proc trainParamsGRU(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  # 1. Create the object field
  var GRUConfig: ModelField

  GRUConfig.field_name = field_name
  GRUConfig.field_type = nnkBracketExpr.newTree(
      ident("GRULayer"), self.subtype
    )

  # 2. Configure weights and biases
  let
    topo = self.topoTable.getOrDefault(field_name)
    sst = self.subtype[1] # we need to get the subsubtype float32/float64

    in_shape = topo.in_shape
    nb_features = quote do: `in_shape`[2]
    hidden_size = topo.gru_hidden_size
    nb_layers = topo.gru_nb_layers

    W3s0_shape = quote do: [3 * `hidden_size`, `nb_features`]
    W3sN_shape = quote do: [`nb_layers` - 1, 3 * `hidden_size`, `hidden_size`] # TODO bidir support
    U3s_shape = quote do: [`nb_layers`, 3 * `hidden_size`, `hidden_size`]
    biases_shape = quote do: [`nb_layers`, 1, 3 * `hidden_size`]

    W3s0 = quote do: xavier_uniform(`W3s0_shape`, `sst`)
    W3sN = quote do: xavier_uniform(`W3sN_shape`, `sst`)
    # TODO U3s, hidden state weight: orthogonal initialization
    # https://github.com/mratsim/Arraymancer/issues/339
    U3s = quote do: yann_normal(`U3s_shape`, `sst`)
    bW3s = quote do: zeros[`sst`](`biases_shape`)
    bU3s = quote do: zeros[`sst`](`biases_shape`)

  GRUConfig.init_call = newStmtList()

  let ctx = self.context

  GRUConfig.init_call.add quote do:
    result.`field_name`.W3s0 = `ctx`.variable(
      `W3s0`, requires_grad = true # TODO allow freezing
    )

    result.`field_name`.W3sN = block:
      if `nb_layers` > 1:
        `ctx`.variable(
            `W3sN`, requires_grad = true # TODO allow freezing
          )
      else:
        # Empty variable, we still need it initialised to allow `requires_grad`
        Variable[Tensor[`sst`]](context: `ctx`)

    result.`field_name`.U3s = `ctx`.variable(
      `U3s`, requires_grad = true # TODO allow freezing
    )

    result.`field_name`.bW3s = `ctx`.variable(
      `bW3s`, requires_grad = true # TODO allow freezing
    )

    result.`field_name`.bU3s = `ctx`.variable(
      `bU3s`, requires_grad = true # TODO allow freezing
    )

  self.trainparams.add GRUConfig

proc genModelFieldInit*(self: Neuromancer) =

  self.trainparams = @[]

  for k, v in pairs(self.topoTable):
    case v.kind:
    of lkConv2D: self.trainParamsConv2D(k, v)
    of lkLinear: self.trainParamsLinear(k, v)
    of lkGRU: self.trainParamsGRU(k, v)
    else:
      discard
