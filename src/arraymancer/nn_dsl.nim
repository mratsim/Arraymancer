# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  private/ast_utils,
  autograd

macro ctxSubtype*(context: Context): untyped =
  ## Extract the subtype from a Context
  result = replaceSymsByIdents(context.getTypeInst[1])

proc splitSections(config: NimNode): tuple[layers, forward: NimNode] =
  template unknown =
    error:
      lineInfo(section) &
        ": unknown neural network configuration section \"" &
        $section[0] & "\""

  for section in config:
    if section.kind == nnkCall:
      if eqIdent(section[0], "layers"):
        result.layers = section[1]
      else:
        unknown()
    elif section.kind == nnkCommand:
      if eqIdent(section[0], "forward"):
        # For forward we copy everything.
        # We have to deal with forward with multiple inputs like "forward x, y, z:"
        # and we will do that later.
        result.forward = section
      else:
        unknown()
    else:
        unknown()

type
  LayerInfo = object
    name: NimNode
    typeName: NimNode
    arguments: seq[NimNode]

func createLayerInfo(layers: NimNode): seq[LayerInfo] =
  for layer in layers:

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

func createModelType(layerInfos: seq[LayerInfo], modelName: NimNode, ctxSubtype: NimNode): NimNode =
  var recList = newNimNode(nnkRecList)
  for layerInfo in layerInfos:
    doAssert layerInfo.name.kind == nnkIdent
    doAssert layerInfo.typeName.kind == nnkIdent
    recList.add newIdentDefs(
      layerInfo.name,
      newNimNode(nnkBracketExpr).add(
        layerInfo.typeName,
        ctxSubtype[1]
      )
    )
  
  doAssert modelName.kind == nnkIdent
  doAssert ctxSubtype.kind == nnkBracketExpr
  result = newNimNode(nnkTypeSection).add(
    newNimNode(nnkTypeDef).add(
      modelName,
      newEmptyNode(),
      newNimNode(nnkObjectTy).add(
        newEmptyNode(),
        newEmptyNode(),
        recList
      )
    )
  )
  
func createInitProc(layerInfos: seq[LayerInfo], modelName, ctxSubtype: NimNode): NimNode =
  doAssert modelName.kind == nnkIdent
  doAssert ctxSubtype.kind == nnkBracketExpr

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
            ctxSubtype[1]
          )
        ).add(layerInfo.arguments)
      )
    )
    
  result = newProc(
    name = ident"init",
    params = @[
      modelName,
      newIdentDefs(
        ident"ctx",
        newNimNode(nnkBracketExpr).add(
          ident"Context",
          ctxSubtype
        )
      ),
      newIdentDefs(
        ident"model_type",
        newNimNode(nnkBracketExpr).add(
          ident"typedesc",
          modelName
        )
      )
    ],
    body = body
  )

func createForwardProc(layerInfos: seq[LayerInfo], forward, modelName, ctxSubtype: NimNode): NimNode =

  doAssert forward.kind == nnkCommand
  doAssert forward.len == 3
  doAssert forward[0].strVal == "forward"
  
  let
    inputIdent = forward[1]
    forwardCall = forward[2]
  

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
            ident"auto"
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
  body.add forwardCall

  let inOutType = newNimNode(nnkBracketExpr).add(
    ident"Variable",
    ctxSubtype
  )

  result = newProc(
    name = ident"forward",
    params = @[
      inOutType,
      newIdentDefs(
        ident"self",
        modelName
      ),
      newIdentDefs(
        inputIdent,
        inOutType
      )
    ],
    body = body
  )


macro network*(ctx: Context, model_name: untyped, config: untyped): untyped =
  ## Declare a neural network.
  ##
  ## Example usage:
  ##    .. code:: nim
  ##         network ctx, DemoNet:
  ##           layers:
  ##             cv1:        Conv2D(@[1, 28, 28], 20, (5, 5))
  ##             mp1:        Maxpool2D(cv1.out_shape, (2,2), (0,0), (2,2))
  ##             cv2:        Conv2D(mp1.out_shape, 50, (5, 5))
  ##             mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
  ##             fl:         Flatten(mp2.out_shape)
  ##             hidden:     Linear(fl.out_shape[0], 500)
  ##             classifier: Linear(500, 10)
  ##           forward x:
  ##             x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

  # TODO fix doc
  # TODO better doc

  # 0. separate the configuration into layers and forward part
  let sections = config.splitSections()

  # 1. create layer info
  let layerInfos = sections.layers.createLayerInfo()

  # 2. create model type
  let ctxSubtype = getAST(ctxSubtype(ctx))
  let modelType = createModelType(layerInfos, model_name, ctxSubtype)

  # 3. create init proc
  let initProc = createInitProc(layerInfos, model_name, ctxSubtype)

  # 4. create forward proc

  let forwardProc = createForwardProc(layerInfos, sections.forward, model_name, ctxSubtype)

  # 5. combine everything into a statement

  result = newStmtList(
    modelType,
    initProc,
    forwardProc
  )

  echo toStrLit(result)

# TODO document layers
# TODO write layer types for embedding, gcn, gru
# TODO try to fix identifier ambiguities inside init and forward function of neural model
# TODO maybe add error handling to this macro
# TODO make sure no performance regressions
# TODO what about layers based on other tensor types?
