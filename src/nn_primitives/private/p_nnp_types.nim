import ../../tensor/tensor

type
  Size2D* = tuple
    ## Tuple of height and width
    ## This is notably used to specify padding and stride parameters for Convolution2D.
    height: int
    width: int

proc nchw_channels*[T](input: Tensor[T]): int {.inline.}  =
  ## Return number of channels of a Tensor in NCWH layout
  input.shape[^3]

proc nchw_height*[T](input: Tensor[T]): int {.inline.} =
  ## Return height of a Tensor in NCWH layout
  input.shape[^2]

proc nchw_width*[T](input: Tensor[T]): int {.inline.} =
  ## Return width of a Tensor in NCWH layout
  input.shape[^1]
