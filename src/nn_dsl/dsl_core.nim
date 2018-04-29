# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ./dsl_types, ./dsl_initialization, ./dsl_utils, ./dsl_topology, ./dsl_forwardsugar,
  ../autograd/autograd

proc splitSections(config: NimNode): NetworkSections =
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

proc genModelType(self: Neuromancer, model_name: string) =

  var records = nnkRecList.newTree

  for record in self.trainparams:
    let (field_name, field_type, _) = record
    records.add nnkIdentDefs.newTree(
      newIdentNode($field_name),
      field_type,
      newEmptyNode()
    )

  self.type_section = nnkStmtList.newTree(
    nnkTypeSection.newTree(
      nnkTypeDef.newTree(
        newIdentNode(model_name),
        newEmptyNode(), # Generic params get here
        nnkObjectTy.newTree(
          newEmptyNode(),
          newEmptyNode(),
          records
        )
      )
    )
  )

proc genInitProc(self: Neuromancer, model_name: string) =

  self.init_proc = newStmtList()

  let
    subtype = self.subtype
    modelType = newIdentNode(model_name)
    procBody = newStmtList()

  for record in self.trainparams:
    let (_, _, initStmt) = record
    procBody.add initStmt

  self.init_proc.add quote do:
    proc init(ctx: Context[`subtype`], model_type: typedesc[`modelType`]): `modelType` =
      `procBody`

proc genForwardProc(self: Neuromancer, model_name: string, forward: NimNode) =

  forward.expectKind(nnkCommand)
  assert eqIdent(forward[0], "forward")
  forward[^1].expectKind(nnkStmtList)
  # forward x:
  #   x.cv1.relu.mp1.flatten.classifier
  # -----------------------------------
  # Command
  #   Ident "forward"
  #   Ident "x"
  #   StmtList
  #     DotExpr
  #       DotExpr
  #         DotExpr
  #           DotExpr
  #             DotExpr
  #               Ident "x"
  #               Ident "cv1"
  #             Ident "relu"
  #           Ident "mp1"
  #         Ident "flatten"
  #       Ident "classifier"

  # 0. Prepare type information and the raw proc body
  let
    ModelType = newIdentNode(model_name)
    InOutType = nnkBracketExpr.newTree(
      newIdentNode("Variable"), self.subtype
    )
    procBody = forward[^1]

  # 1. Create the input variables with their type
  var inputVars = nnkIdentDefs.newTree()

  for varIndex in 1..forward.len-2:
    inputVars.add newIdentNode($forward[varIndex])

  inputVars.add InOutType
  inputVars.add newEmptyNode() # Default Value

  # 2. Add the shortut syntax templates
  var shortcutTemplates = newStmtList()

  for shortcut in self.forward_templates:
    shortcutTemplates.add shortcut

  # 3. Create the forward proc
  self.forward_proc = nnkProcDef.newTree(
    newIdentNode("forward"), newEmptyNode(), newEmptyNode(),
    nnkFormalParams.newTree(
      # Result type
      InOutType,
      # Model
      nnkIdentDefs.newTree(newIdentNode("self"), ModelType, newEmptyNode()),
      # Variables
      inputVars
    ),
    newEmptyNode(), newEmptyNode(),
    nnkStmtlist.newTree(
      # TODO asserts
      shortcutTemplates,
      procBody,
    )
  )

macro network*(ctx: Context, model_name: untyped, config: untyped): untyped =

  # 0. - Separate the configuration into layers and forward part
  #    - get the subtype of the model (Tensor[float32], CudaTensor[float64], ...)
  let sections = config.splitSections

  # 1. Initialize the VM to analyse the neural network Graph.
  #    - Get the input shapes
  #    - Get the layers
  let vm = new Neuromancer
  vm.context = ctx
  vm.subtype = getAST(ctxSubtype(ctx))
  vm.topoTable = initTable[NimNode, LayerTopology]()
  vm.topoTable.topoFromLayers(sections.layers)

  # 2. Generate the model fields, initialization and template synctactic sugar
  vm.genModelFieldInit()
  vm.genTemplateShortcuts()

  # 3. Generate the type section
  vm.genModelType($model_name)
  vm.genInitProc($model_name)
  vm.genForwardProc($model_name, sections.forward)

  # 4. Output the result: type + init proc + forward proc
  result = newStmtList()
  result.add vm.type_section
  result.add vm.init_proc
  result.add vm.forward_proc



