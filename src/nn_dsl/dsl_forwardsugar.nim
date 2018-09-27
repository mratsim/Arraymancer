# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ./dsl_types

proc shortcutConv2D(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  # TODO: Add padding and strides

  let shortcut = quote do:
    template `field_name`(x: Variable): Variable =
      x.conv2d(self.`field_name`.weight, self.`field_name`.bias)

  self.forward_templates.add shortcut

proc shortcutMaxPool2D(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  let
    topo = self.topoTable.getOrDefault(field_name)
    kernel = topo.m2d_kernel
    padding = topo.m2d_padding
    strides = topo.m2d_strides

  let shortcut = quote do:
    template `field_name`(x: Variable): Variable =
      x.maxpool2D(`kernel`, `padding`, `strides`)

  self.forward_templates.add shortcut

proc shortcutLinear(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  let shortcut = quote do:
    template `field_name`(x: Variable): Variable =
      x.linear(self.`field_name`.weight, self.`field_name`.bias)

  self.forward_templates.add shortcut

proc shortcutFlatten(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  let shortcut = quote do:
    template `field_name`(x: Variable): Variable =
      x.flatten

  self.forward_templates.add shortcut

proc shortcutGRU(self: Neuromancer, field_name: NimNode, topo: LayerTopology) =

  let
    topo = self.topoTable.getOrDefault(field_name)
    seq_len = topo.gru_seq_len
    layers = topo.gru_nb_layers

  let shortcut = quote do:
    template `field_name`(input, hidden0: Variable): tuple[output, hiddenN: Variable] =
      input.gru(
        hidden0, `seq_len`, `layers`,
        self.`field_name`.W3s0, self.`field_name`.W3sN,
        self.`field_name`.U3s,
        self.`field_name`.bW3s, self.`field_name`.bU3s
      )

  self.forward_templates.add shortcut

proc genTemplateShortcuts*(self: Neuromancer) =

  self.forward_templates = @[]

  for k, v in pairs(self.topoTable):
    case v.kind:
    of lkConv2D: self.shortcutConv2D(k, v)
    of lkLinear: self.shortcutLinear(k, v)
    of lkMaxPool2D: self.shortcutMaxPool2D(k, v)
    of lkFlatten: self.shortcutFlatten(k, v)
    of lkGRU: self.shortcutGRU(k, v)
    else:
      discard
