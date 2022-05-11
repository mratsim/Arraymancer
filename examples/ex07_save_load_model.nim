import ../src/arraymancer, strformat, os

#[
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
*** this example is modified from the simple2layers example to show how to
save/load models. We can do this by defining our own model class
and forward procedure, then defining procedures to save/load the model. ***
]#

# ##################################################################
# Environment variables

# INPUTD is input dimension;
# HIDDEND is hidden dimension; OUTPUTD is output dimension.
let (BATCHSIZE, INPUTD, HIDDEND, OUTPUTD) = (32, 1000, 100, 10)

# Create the autograd context that will hold the computational graph
let ctx = newContext Tensor[float32]

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
let
  x = ctx.variable(randomTensor[float32](BATCHSIZE, INPUTD, 1'f32))
  y = randomTensor[float32](BATCHSIZE, OUTPUTD, 1'f32)

# ##################################################################
# Define the model.

type
  LinearLayer = object
    weight: Variable[Tensor[float32]]
    bias: Variable[Tensor[float32]]
  ExampleNetwork = object
    hidden: LinearLayer
    output: LinearLayer

template weightInit(shape: varargs[int], init_kind: untyped): Variable =
  ctx.variable(
    init_kind(shape, float32),
    requires_grad = true)

proc newExampleNetwork(ctx: Context[Tensor[float32]]): ExampleNetwork =
  result.hidden.weight = weightInit(HIDDEND, INPUTD, kaiming_normal)
  result.hidden.bias   = ctx.variable(zeros[float32](1, HIDDEND), requires_grad = true)
  result.output.weight = weightInit(OUTPUTD, HIDDEND, yann_normal)
  result.output.bias   = ctx.variable(zeros[float32](1, OUTPUTD), requires_grad = true)

proc forward(network: ExampleNetwork, x: Variable): Variable =
  result =  x.linear(
    network.hidden.weight, network.hidden.bias).relu.linear(
      network.output.weight, network.output.bias)

proc save(network: ExampleNetwork) =
  # this is a quick prototype, but you get the idea.
  # perhaps a better way to do this would be to save all weights/biases of
  # the model into a single file.
  network.hidden.weight.value.write_npy("hiddenweight.npy")
  network.hidden.bias.value.write_npy("hiddenbias.npy")
  network.output.weight.value.write_npy("outputweight.npy")
  network.output.bias.value.write_npy("outputbias.npy")

proc load(ctx: Context[Tensor[float32]]): ExampleNetwork =
  result.hidden.weight = ctx.variable(read_npy[float32]("hiddenweight.npy"), requires_grad = true)
  result.hidden.bias   = ctx.variable(read_npy[float32]("hiddenbias.npy"), requires_grad = true)
  result.output.weight = ctx.variable(read_npy[float32]("outputweight.npy"), requires_grad = true)
  result.output.bias   = ctx.variable(read_npy[float32]("outputbias.npy"), requires_grad = true)


var
  model = if fileExists("hiddenweight.npy"): ctx.load() else: ctx.newExampleNetwork()
  # quick prototype for model variable assignment.
  # if a numpy file exists in the currentDir you will load the model, otherwise create a new model
  optim = model.optimize(SGD, learning_rate = 1e-4'f32)

# ##################################################################
# Training

for t in 0 ..< 250:
  let
    y_pred = model.forward(x)
    loss = y_pred.mse_loss(y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  loss.backprop()
  optim.update()

# save model
model.save()

# simple sanity check for loading model (validates that the model is saved correctly)
var hidden_weights = model.hidden.weight.value
let newModel = ctx.load()
doAssert newModel.hidden.weight.value == hidden_weights, "loaded model weights do not match with original model"
