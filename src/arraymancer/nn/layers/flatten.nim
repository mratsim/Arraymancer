import
  ../../tensor,
  ../../autograd

type
  Flatten*[T] = object
    inShape: seq[int]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[Flatten],
  inShape: seq[int]
): Flatten[T] =

  ## Creates a flattening layer, which "flattens" its input by reshaping it into a one-dimensional tensor.
  ## ``inShape`` describes the expected shape of the input.
  result.inShape = inShape

proc forward*[T](self: Flatten[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  assert (@(input.value.shape))[1..^1] == self.inShape
  input.flatten()

func outShape*[T](self: Flatten[T]): seq[int] =
  result = @[1]
  for i in self.inShape:
      result[0] *= i

func inShape*[T](self: Flatten[T]): seq[int] =
  self.inShape
