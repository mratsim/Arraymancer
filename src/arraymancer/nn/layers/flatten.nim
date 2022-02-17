import
  ../../tensor,
  ../../autograd

type
  Flatten*[T] = object
    in_shape: seq[int]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[Flatten[T]],
  inShape: seq[int]
): Flatten[T] =
  result.inShape = inShape

proc forward*[T](self: Flatten[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.flatten()

proc outShape*[T](self: Flatten[T]): seq[int] =
  result = @[1]
  for i in self.inShape:
      result[0] *= i

proc inShape*[T](self: Flatten[T]): seq[int] =
  self.inShape