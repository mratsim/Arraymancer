# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  autograd


type
  SectionInfo = object
    idents: seq[NimNode]
    body: NimNode

proc splitSections(config: NimNode): tuple[layers, forward: SectionInfo] =
  template unknown =
    error:
      lineInfo(section) &
        ": unknown neural network configuration section \"" &
        $section[0] & "\""

  for section in config:
    if section.kind == nnkCall or section.kind == nnkCommand:
      # We have to deal with layer with multiple inputs like "layer x, y, z:"
      # so we will sort the different parts of each section beforehand.
      proc getSectionInfo(nodes: seq[NimNode]): SectionInfo =
        for i, node in nodes.pairs:
          if node.kind == nnkIdent:
            result.idents.add node
          else:
            doAssert node.kind == nnkStmtList
            #if i != nodes.len - 2:
            #  #echo treeRep nodes
            doAssert i == nodes.len - 1
            result.body = node 

      if eqIdent(section[0], "layers"):
        result.layers = section[1..^1].getSectionInfo()
      elif eqIdent(section[0], "forward"):
        result.forward = section[1..^1].getSectionInfo()
      else:
        unknown()
    else:
        unknown()

type
  LayerInfo = object
    name: NimNode
    typeName: NimNode
    arguments: seq[NimNode]

func createLayerInfo(layers: SectionInfo): seq[LayerInfo] =
  doAssert layers.body.kind == nnkStmtList

  for layer in layers.body:

    doAssert layer.kind == nnkCall
    doAssert layer.len == 2
    doAssert layer[0].kind == nnkIdent
    doAssert layer[1].kind == nnkStmtList
    doAssert layer[1].len == 1
    doAssert layer[1][0].kind == nnkCall
    doAssert layer[1][0].len >= 1
    doAssert layer[1][0][0].kind == nnkIdent
    result.add LayerInfo(
      name: layer[0],
      typeName: layer[1][0][0]
    )
    if layer[1][0].len >= 2:
      result[^1].arguments = layer[1][0][1..^1]

func createModelType(layerInfos: seq[LayerInfo], modelName: NimNode): NimNode =
  var recList = newNimNode(nnkRecList)
  for layerInfo in layerInfos:
    doAssert layerInfo.name.kind == nnkIdent
    doAssert layerInfo.typeName.kind == nnkIdent
    recList.add newIdentDefs(
      layerInfo.name,
      newNimNode(nnkBracketExpr).add(
        layerInfo.typeName,
        ident"T"
      )
    )
  
  doAssert modelName.kind == nnkIdent
  result = newNimNode(nnkTypeSection).add(
    newNimNode(nnkTypeDef).add(
      modelName,
      newNimNode(nnkGenericParams).add(
        newIdentDefs(
          ident"T",
          newEmptyNode()
        )
      ),
      newNimNode(nnkObjectTy).add(
        newEmptyNode(),
        newEmptyNode(),
        recList
      )
    )
  )
  
func createInitProc(layerInfos: seq[LayerInfo], layers: SectionInfo, modelName: NimNode): NimNode =
  doAssert modelName.kind == nnkIdent

  var body = newNimNode(nnkStmtList)
  for layerInfo in layerInfos:
    body.add(
      newNimNode(nnkTemplateDef).add(
        layerInfo.name,
        newEmptyNode(),
        newEmptyNode(),
        newNimNode(nnkFormalParams).add ident"auto",
        newEmptyNode(),
        newEmptyNode(),
        newStmtList(
          newDotExpr(
            ident"result",
            layerInfo.name
          )
        )
      )
    )
  for layerInfo in layerInfos:
    body.add(
      newAssignment(
        layerInfo.name,
        newCall(
          ident"init",
          ident"ctx",
          newNimNode(nnkBracketExpr).add(
            layerInfo.typeName,
            ident"T"
          )
        ).add(layerInfo.arguments)
      )
    )

  var params = @[
    newNimNode(nnkBracketExpr).add(
      modelName,
      ident"T"
    ),
    newIdentDefs(
      ident"ctx",
      newNimNode(nnkBracketExpr).add(
        ident"Context",
        newNimNode(nnkBracketExpr).add(
          ident"AnyTensor",
          ident"T"
        )
      )
    ),
    newIdentDefs(
      ident"model_type",
      newNimNode(nnkBracketExpr).add(
        ident"typedesc",
        newNimNode(nnkBracketExpr).add(
          modelName,
          ident"T"
        )
      )
    )
  ]

  for inputIdent in layers.idents:
    params.add(
      newIdentDefs(
        inputIdent,
        ident"auto"
      )
    )
    
  result = newProc(
    name = ident"init",
    params = params,
    body = body
  )
  # GenericParams
  result[2] = newNimNode(nnkGenericParams).add(
    newIdentDefs(
      ident"T",
      newEmptyNode()
    )
  )

func createForwardProc(layerInfos: seq[LayerInfo], forward: SectionInfo, modelName: NimNode): NimNode =

  var body = newNimNode(nnkStmtList)

  for layerInfo in layerInfos:
    body.add(
      newNimNode(nnkTemplateDef).add(
        layerInfo.name,
        newEmptyNode(),
        newEmptyNode(),
        newNimNode(nnkFormalParams).add(
          ident"auto",
          newIdentDefs(
            ident"x",
            newNimNode(nnkBracketExpr).add(
              ident"varargs",
              ident"untyped"
            )
          )
        ),
        newEmptyNode(),
        newEmptyNode(),
        newStmtList(
          newCall(
            ident"forward",
            newDotExpr(
              ident"self",
              layerInfo.name,
            ),
            ident"x"
          )
        )
      )
    )
  body.add forward.body

  var params = @[
    ident"auto",
    newIdentDefs(
      ident"self",
      newNimNode(nnkBracketExpr).add(
        modelName,
        ident"T"
      )
    )
  ]

  for inputIdent in forward.idents:
    params.add(
      newIdentDefs(
        inputIdent,
        ident"auto"
      )
    )

  result = newProc(
    name = ident"forward",
    params = params,
    body = body
  )
  # GenericParams
  result[2] = newNimNode(nnkGenericParams).add(
    newIdentDefs(
      ident"T",
      newEmptyNode()
    )
  )



macro network*(model_name: untyped, config: untyped): untyped =
  ## Declare a neural network.
  ##
  ## Example usage:
  ##    .. code:: nim
  ##          network DemoNet:
  ##            layers h, w:
  ##              cv1:        Conv2D(@[1, h, w], 20, (5, 5))
  ##              mp1:        Maxpool2D(cv1.out_shape, (2,2), (0,0), (2,2))
  ##              cv2:        Conv2D(mp1.out_shape, 50, (5, 5))
  ##              mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
  ##              fl:         Flatten(mp2.out_shape)
  ##              hidden:     Linear(fl.out_shape[0], 500)
  ##              classifier: Linear(500, 10)
  ##            forward x:
  ##              x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

  # TODO better doc

  # 0. separate the configuration into layers and forward part
  let sections = config.splitSections()

  # 1. create layer info
  let layerInfos = sections.layers.createLayerInfo()

  # 2. create model type
  let modelType = createModelType(layerInfos, model_name)

  # 3. create init proc
  let initProc = createInitProc(layerInfos, sections.layers, model_name)

  # 4. create forward proc

  let forwardProc = createForwardProc(layerInfos, sections.forward, model_name)

  # 5. combine everything into a statement

  result = newStmtList(
    modelType,
    initProc,
    forwardProc
  )

  echo toStrLit(result)