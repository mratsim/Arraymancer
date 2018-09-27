# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables, hashes,
  ../autograd/autograd

type
  #################################################

  NetworkSections* = tuple[layers, forward: NimNode]

  ModelField* = tuple[field_name: NimNode, field_type: NimNode, init_call: NimNode]
    ## Field name, type and initialization proc

  Neuromancer* = ref object
    ## Pathfinder that will go through the network topology
    ## and create the corresponding type, initialization and forward proc

    # Running state
    topoTable*: TopoTable          # Retrieve the layer properties.
                                  # Maps a NimNode of nnkIdent to the corresponding input and output shape
    subtype*: NimNode

    # Outputs
    context*: NimNode                # copy-paste of the context
    type_section*: NimNode           # The type section to declare the new Model type
    trainparams*: seq[ModelField]    # stores the weight/bias and initialization proc
    init_proc*: NimNode              # proc init(T: typedesc[Model]): Model
    forward_proc*: NimNode           # proc forward(self: Model, x, y: Variable)
    forward_templates*: seq[NimNode] # This adds templates synctactic sugar, so that the field name is substituted
                                    # template cv1(x: Variable[T]): Variable[T] =
                                    #   x.conv2d(self.cv1.weight, self.cv1.bias)
    forward_asserts*: NimNode

  #################################################

  LayerKind* = enum
    lkInput, lkConv2D, lkLinear, lkMaxPool2D, lkFlatten,
    lkGRU

  LayerTopology* = object
    ## Describe a layer topology
    in_shape*, out_shape*: NimNode # Input and output shape
    case kind*: LayerKind
    of lkConv2D:
      c2d_kernel_shape*: NimNode
      c2d_padding*, c2d_strides*: NimNode
    of lkMaxPool2D:
      m2d_kernel*, m2d_padding*, m2d_strides*: NimNode
    of lkGRU:
      gru_hidden_size*: NimNode
      gru_nb_layers*: NimNode
    else:
      discard

  TopoTable* = Table[NimNode, LayerTopology]

  #####################################################
  # Todo: move that in NN part
  TrainableLayer*[TT] = concept layer
    block:
      var trainable: false
      for field in fields(layer):
        trainable = trainable or (field is Variable[TT])
      trainable

  Conv2DLayer*[TT] = object
    weight*: Variable[TT]
    bias*: Variable[TT]
  LinearLayer*[TT] = object
    weight*: Variable[TT]
    bias*: Variable[TT]
  GRULayer*[TT] = object
    W3s0*, W3sN*: Variable[TT]
    U3s*: Variable[TT]
    bW3s*, bU3s*: Variable[TT]

proc hash*(x: NimNode): Hash =
  assert x.kind == nnkIdent
  result = hash($x)
