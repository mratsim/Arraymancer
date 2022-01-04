import
  ../../tensor,
  ../../autograd

type
  Flatten*[T] = object
    in_shape: seq[int]

proc init*[T](
  ctx: Context[Tensor[T]],
  layer_type: typedesc[Flatten[T]],
  in_shape: seq[int]
): Flatten[T] =
  result.in_shape = in_shape

proc forward*[T](self: Flatten[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.flatten()

proc out_shape*[T](self: Flatten[T]): seq[int] =    
  result = @[1]
  for i in self.in_shape:
      result[0] *= i

proc in_shape*[T](self: Flatten[T]): seq[int] =
  self.in_shape