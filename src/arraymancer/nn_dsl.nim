# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros


type
  SectionInfo = object
    idents: seq[NimNode]
    body: NimNode

func splitSections(config: NimNode): tuple[layers, forward: SectionInfo] =
  template unknown =
    error:
      lineInfo(section) &
        ": unknown neural network configuration section \"" &
        $section[0] & "\""

  for section in config:
    if section.kind == nnkCall or section.kind == nnkCommand:
      # We have to deal with forward/init with multiple inputs like "forward x, y, z:"
      # so we will handle these now.
      proc getSectionInfo(nodes: seq[NimNode]): SectionInfo =
        for i, node in nodes.pairs:
          if node.kind == nnkIdent:
            result.idents.add node
          elif node.kind == nnkStmtList and i == nodes.len - 1:
            result.body = node
          else:
            unknown()

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

func createLayerInfo(sectionInfo: SectionInfo): seq[LayerInfo] =
  
  # sectionInfo.idents contains constains a list of identifiers that will be used
  # as function parameters for the init funxtion
  # sectionInfo.body contains the description of layers, e.g.:
  #    cv:        Conv2D(mp1.outShape, 50, (5, 5))
  #    mp:        MaxPool2D(cv2.outShape, (2,2), (0,0), (2,2))
  #    fl:         Flatten(mp2.outShape)
  #    hidden:     Linear(fl.outShape[0], 500)

  if sectionInfo.body.kind != nnkStmtList:
    error("Layer body must be a statement list: \"" & $toStrLit(sectionInfo.body) & "\"", sectionInfo.body)

  for layer in sectionInfo.body:

    if layer.kind != nnkCall or
    layer.len != 2 or
    layer[0].kind != nnkIdent or
    layer[1].kind != nnkStmtList or
    layer[1].len != 1 or
    layer[1][0].kind != nnkCall or
    layer[1][0].len < 1 or
    layer[1][0][0].kind != nnkIdent:
      error("Unknown configuration of layer section: \"" & $toStrLit(layer) & "\"", layer)

    result.add LayerInfo(
      name: layer[0],
      typeName: layer[1][0][0]
    )
    if layer[1][0].len >= 2:
      result[^1].arguments = layer[1][0][1..^1]

func createModelType(layerInfos: seq[LayerInfo], modelName: NimNode): NimNode =

  # creates the type defintion of the model, e.g.:
  #  type
  #    SomeConvNet[T] = object
  #      cv: Conv2D[T]
  #      mp: Maxpool2D[T]
  #      fl: Flatten[T]

  let underlyingTypeSymbol = genSym(nskType, "T")
  var recList = newNimNode(nnkRecList)
  for layerInfo in layerInfos:
    doAssert layerInfo.name.kind == nnkIdent
    doAssert layerInfo.typeName.kind == nnkIdent
    recList.add newIdentDefs(
      layerInfo.name,
      newNimNode(nnkBracketExpr).add(
        layerInfo.typeName,
        underlyingTypeSymbol
      )
    )
  
  doAssert modelName.kind == nnkIdent

  result = newNimNode(nnkTypeSection).add(
    newNimNode(nnkTypeDef).add(
      modelName,
      newNimNode(nnkGenericParams).add(
        newIdentDefs(
          underlyingTypeSymbol,
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
  
func createInitProc(layerInfos: seq[LayerInfo], sectionInfo: SectionInfo, modelName: NimNode): NimNode =

  # creates init function of the model, e.g.:
  #   proc init[T](ctx: Context[AnyTensor[T]], modelType: typedesc[SomeConvNet[T]], h: auto; w: auto): SomeConvNet[T] =
  #     template cv(): auto =
  #       result.cv
  #     template mp(): auto =
  #       result.mp
  #     template fl(): auto =
  #       result.fl
  #     cv = init(ctx, Conv2D[T], @[1, h, w], 20, (5, 5))
  #     mp = init(ctx, Maxpool2D[T], cv1.outShape, (2, 2), (0, 0), (2, 2))
  #     fl = init(ctx, Flatten[T], mp.outShape)

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

  let
    ctxSymbol = genSym(nskParam, "ctx")
    underlyingTypeSymbol = ident($toStrLit(genSym(nskGenericParam, "T")))# TODO fix this
  for layerInfo in layerInfos:
    body.add(
      newAssignment(
        layerInfo.name,
        newCall(
          ident"init",
          ctxSymbol,
          newNimNode(nnkBracketExpr).add(
            layerInfo.typeName,
            underlyingTypeSymbol
          )
        ).add(layerInfo.arguments)
      )
    )

  var params = @[
    newNimNode(nnkBracketExpr).add(
      modelName,
      underlyingTypeSymbol
    ),
    newIdentDefs(
      ctxSymbol,
      newNimNode(nnkBracketExpr).add(
        ident"Context",
        newNimNode(nnkBracketExpr).add(
          ident"AnyTensor",
          copy underlyingTypeSymbol # needs to be copied for workaround for https://github.com/nim-lang/Nim/issues/19542
        )
      )
    ),
    newIdentDefs(
      genSym(nskParam, "modelType"),
      newNimNode(nnkBracketExpr).add(
        ident"typedesc",
        newNimNode(nnkBracketExpr).add(
          modelName,
          underlyingTypeSymbol
        )
      )
    )
  ]

  for inputIdent in sectionInfo.idents:
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
      underlyingTypeSymbol,
      newEmptyNode()
    )
  )

func createForwardProc(layerInfos: seq[LayerInfo], forward: SectionInfo, modelName: NimNode): NimNode =

  # create the forward function, e.g.:
  # proc forward[T](self: SomeConvNet[T]; x: auto): auto =
  #   template cv1(x: varargs[untyped]): auto =
  #     forward(self.cv1, x)
  # 
  #   template mp1(x: varargs[untyped]): auto =
  #     forward(self.mp1, x)
  # 
  #   template fl(x: varargs[untyped]): auto =
  #     forward(self.fl, x)
  # 
  #   x.cv1.relu.mp1.cv2.relu.mp2.fl

  let
    selfSymbol = genSym(nskParam, "self")
    underlyingTypeSymbol = genSym(nskGenericParam, "T")

  var body = newNimNode(nnkStmtList)

  for layerInfo in layerInfos:
    let xSymbol = genSym(nskParam, "input")
    body.add(
      newNimNode(nnkTemplateDef).add(
        layerInfo.name,
        newEmptyNode(),
        newEmptyNode(),
        newNimNode(nnkFormalParams).add(
          ident"auto",
          newIdentDefs(
            xSymbol,
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
              selfSymbol,
              layerInfo.name,
            ),
            xSymbol
          )
        )
      )
    )
  body.add forward.body

  var params = @[
    ident"auto",
    newIdentDefs(
      selfSymbol,
      newNimNode(nnkBracketExpr).add(
        modelName,
        underlyingTypeSymbol
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
      underlyingTypeSymbol,
      newEmptyNode()
    )
  )



macro network*(modelName: untyped, config: untyped): untyped =
  ## Declare a neural network.
  ##
  ## Example usage:
  ## 
  ## .. code:: nim
  ##   network DemoNet:
  ##     layers h, w:
  ##       cv1:        Conv2D(@[1, h, w], 20, (5, 5))
  ##       mp1:        Maxpool2D(cv1.outShape, (2,2), (0,0), (2,2))
  ##       cv2:        Conv2D(mp1.outShape, 50, (5, 5))
  ##       mp2:        MaxPool2D(cv2.outShape, (2,2), (0,0), (2,2))
  ##       fl:         Flatten(mp2.outShape)
  ##       hidden:     Linear(fl.outShape[0], 500)
  ##       classifier: Linear(500, 10)
  ##     forward x:
  ##       x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier
  ## 
  ##   let
  ##     ctx = newContext Tensor[float32]
  ##     model = ctx.init(DemoNet, 28, 28)
  ##
  ## 
  ## Custom layers can be created by providing a type, an init-function, and a forward-function.
  ## The type could look like this:
  ## 
  ## .. code:: nim
  ##   type
  ##     MyLayer*[T] = object
  ##       someWeights*: Variable[Tensor[T]]
  ##       importantInfo*: seq[int]
  ## 
  ## It is important that the type has exactly one generic parameter which corresponds to the
  ## underlying type (e.g., ``float32`` or ``int8``).
  ## The init-function is required to adhere to the following structure:
  ## 
  ## .. code:: nim
  ##   proc init*[T](
  ##     ctx: Context[Tensor[T]], # could also be Context[AnyTensor[T]] for example
  ##     layerType: typedesc[MyLayer[T]],
  ##     myInitParam: string
  ##     # ... here you can add all the necessary init parameters, like shapes and number of output features
  ##   ): MyLayer[T] =
  ##     discard # your init stuff
  ## 
  ## The only requirement for the forward function is that the first parameter must be of your layer type like this:
  ## 
  ## .. code:: nim
  ##   proc forward*[T](self: MyLayer[T], myInput: SpecialInputType, doNothing: bool): Variable[Tensor[T]] =
  ##     if not doNothing:
  ##       result = myInput.yourComputations(self.importantInfo, self.someWeights)
  ##     
  ## 
  ## Often it is also useful to provide ``proc outShape(m: MyLayer): seq[int]`` and possibly
  ## ``proc inShape(m: MyLayer): seq[int]`` functions.
  ## 
  ## Your custom layer can then be used for example like this:
  ## 
  ## .. code:: nim
  ##   network DemoNet2:
  ##     layers:
  ##       myLayer:    MyLayer(myInitParam = "hello!")
  ##       fl:         Flatten(myLayer.outShape)
  ##       hidden:     Linear(fl.outShape[0], 500)
  ##       classifier: Linear(500, 10)
  ##     forward x:
  ##       x.myLayer(doNothing = false).fl.hidden.relu.classifier

  # TODO better doc
  
  if modelName.kind != nnkIdent:
    error("Name of model must be an identifier: \"" & $toStrLit(modelName) & "\"", modelName)

  # 0. separate the configuration into layers and forward part
  let sections = config.splitSections()

  # 1. create layer info
  let layerInfos = sections.layers.createLayerInfo()

  # 2. create model type
  let modelType = createModelType(layerInfos, modelName)

  # 3. create init proc
  let initProc = createInitProc(layerInfos, sections.layers, modelName)

  # 4. create forward proc

  let forwardProc = createForwardProc(layerInfos, sections.forward, modelName)

  # 5. combine everything into a statement
  result = newStmtList(
    modelType,
    initProc,
    forwardProc
  )

  # echo toStrLit(result)