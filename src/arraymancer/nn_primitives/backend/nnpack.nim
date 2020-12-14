# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## *
##  @brief Status code for any NNPACK function call.
##

import ../../std_version_types

{.passl: "-lnnpack -lpthreadpool -lpthread".}

type
  nnp_status* {.size: sizeof(cint).} = enum ## * The call succeeded, and all output arguments now contain valid data.
    nnp_status_success = 0,     ## * NNPACK function was called with batch_size == 0.
    nnp_status_invalid_batch_size = 2, ## * NNPACK function was called with channels == 0.
    nnp_status_invalid_channels = 3, ## * NNPACK function was called with input_channels == 0.
    nnp_status_invalid_input_channels = 4, ## * NNPACK function was called with output_channels == 0.
    nnp_status_invalid_output_channels = 5, ## * NNPACK function was called with input_size.height == 0 or input_size.width == 0
    nnp_status_invalid_input_size = 10, ## * NNPACK function was called with input_stride.height == 0 or input_stride.width == 0
    nnp_status_invalid_input_stride = 11, ## * NNPACK function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
                                       ##
                                       ##   - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
                                       ##   - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
                                       ##   - input_padding.top    >= kernel_size.height (>= pooling_size.height)
                                       ##   - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
                                       ##
    nnp_status_invalid_input_padding = 12, ## * NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0
    nnp_status_invalid_kernel_size = 13, ## * NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0
    nnp_status_invalid_pooling_size = 14, ## * NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0
    nnp_status_invalid_pooling_stride = 15, ## * NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration
    nnp_status_invalid_algorithm = 16, ## * NNPACK function was called with convolution transform strategy not in nnp_convolution_transform_strategy enum
    nnp_status_invalid_transform_strategy = 17, ## * NNPACK function was called with output_subsampling.height == 0 or output_subsampling.width == 0
    nnp_status_unsupported_input_size = 20, ## * NNPACK does not support the particular input stride for the function
    nnp_status_unsupported_input_stride = 21, ## * NNPACK does not support the particular input padding for the function
    nnp_status_unsupported_input_padding = 22, ## * NNPACK does not support the particular kernel size for the function
    nnp_status_unsupported_kernel_size = 23, ## * NNPACK does not support the particular pooling size for the function
    nnp_status_unsupported_pooling_size = 24, ## * NNPACK does not support the particular pooling stride for the function
    nnp_status_unsupported_pooling_stride = 25, ## * NNPACK does not support the particular convolution algorithm for the function
    nnp_status_unsupported_algorithm = 26, ## * NNPACK does not support the particular convolution transform strategy for the algorithm
    nnp_status_unsupported_transform_strategy = 27, ## * NNPACK does not support the particular activation function for the function
    nnp_status_unsupported_activation = 28, ## * NNPACK does not support the particular activation function parameters for the function
    nnp_status_unsupported_activation_parameters = 29, ## * NNPACK function was called before the library was initialized
    nnp_status_uninitialized = 50, ## * NNPACK does not implement this function for the host CPU
    nnp_status_unsupported_hardware = 51, ## * NNPACK failed to allocate memory for temporary buffers
    nnp_status_out_of_memory = 52, ## * Scratch space buffer is too small
    nnp_status_insufficient_buffer = 53, ## * Scratch space buffer is not properly aligned
    nnp_status_misaligned_buffer = 54

  pthreadpool_t* = pointer

const
  nnp_status_invalid_output_subsampling* = nnp_status_invalid_kernel_size
  nnp_status_invalid_activation* = nnp_status_invalid_pooling_size
  nnp_status_invalid_activation_parameters* = nnp_status_invalid_pooling_stride

## *
##  @brief Activation applied applied after a convolutional or fully-connected layer.
##

type
  nnp_activation* {.size: sizeof(cint).} = enum ## * Identity activation f(x) := x, i.e. no transformation
    nnp_activation_identity = 0, ## * ReLU activation f(x) := max(0, x)
    nnp_activation_relu = 1


## *
##  @brief Algorithm for computing convolutional layers.
##

type
  nnp_convolution_algorithm* {.size: sizeof(cint).} = enum ## * Let NNPACK choose the algorithm depending on layer parameters
    nnp_convolution_algorithm_auto = 0, ## * Tiled convolution based on 2D Fourier transform with 8x8 blocks. Supports kernels up to 8x8.
    nnp_convolution_algorithm_ft8x8 = 1, ## * Tiled convolution based on 2D Fourier transform with 16x16 blocks. Supports kernels up to 16x16.
    nnp_convolution_algorithm_ft16x16 = 2, ## * Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks. Supports only 3x3 kernels.
    nnp_convolution_algorithm_wt8x8 = 3, ## * Direct convolution via implicit GEMM.
    nnp_convolution_algorithm_implicit_gemm = 4, ## * Direct convolution implementation.
    nnp_convolution_algorithm_direct = 5, ## *
                                       ##  Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks in FP16.
                                       ##  Supports only 3x3 kernels. Implemented only for new ARM processors (with NEON-HP),
                                       ##  on non-supported processors falls back to nnp_convolution_algorithm_wt8x8.
                                       ##
    nnp_convolution_algorithm_wt8x8_fp16 = 6


type
  nnp_convolution_transform_strategy* {.size: sizeof(cint).} = enum
    nnp_convolution_transform_strategy_compute = 1,
    nnp_convolution_transform_strategy_precompute = 2,
    nnp_convolution_transform_strategy_reuse = 3


##  For backward compatibility

const
  nnp_convolution_transform_strategy_block_based* = nnp_convolution_transform_strategy_compute
  nnp_convolution_transform_strategy_tuple_based* = nnp_convolution_transform_strategy_compute

## *
##  @brief Size of images, kernels, and pooling filters in NNPACK.
##

type
  nnp_size* {.bycopy.} = object
    width*: csize_t              ## * Width (horizontal size) of an image, kernel, or pooling filter.
    ## * Height (vertical size) of an image, kernel, or pooling filter.
    height*: csize_t

## *
##  @brief Padding of images in NNPACK.
##

type
  nnp_padding* {.bycopy.} = object
    top*: csize_t                ## * Padding above the image data
    ## * Padding on the right of image data
    right*: csize_t              ## * Padding below the image data
    bottom*: csize_t             ## * Padding on the left of image data
    left*: csize_t

## *
##  @brief Profiling information about time spent in different phases of a function call.
##

type
  nnp_profile* {.bycopy.} = object
    total*: cdouble            ## * Time spent inside the function call, in seconds.
    ## * Time spend on transformation of the input or input gradient tensor, in seconds.
    input_transform*: cdouble  ## * Time spend on transformation of the kernel or kernel gradient tensor, in seconds.
    kernel_transform*: cdouble ## * Time spend on transformation of the output or output gradient tensor, in seconds.
    output_transform*: cdouble ## * Time spend on multiplication-accumulation of transformed coefficients, in seconds.
    block_multiplication*: cdouble


proc nnp_initialize*(): nnp_status {.cdecl, importc: "nnp_initialize".}
proc nnp_deinitialize*(): nnp_status {.cdecl, importc: "nnp_deinitialize".}
## *
##  @brief Computes output of a 2D convolutional layer from input and kernel tensors.
##  @details This function targets training of convolutional neural networks and performs forward propagation.
##           It is optimized for moderate minibatch sizes (64-128) and can be inefficient on a small minibatch.
##           For minibatch size 1, use nnp_convolution_inference for optimal performance.
##  @param algorithm The type of algorithm to use for convolution. Possible values are:
##
##     - nnp_convolution_algorithm_auto    -- let the function choose the algorithm.
##     - nnp_convolution_algorithm_ft8x8   -- tiled convolution based on 2D Fourier transform with 8x8 blocks.
##                                            Supports kernels up to 8x8.
##     - nnp_convolution_algorithm_ft16x16 -- tiled convolution based on 2D Fourier transform with 16x16 blocks.
##                                            Supports kernels up to 16x16.
##     - nnp_convolution_algorithm_wt8x8   -- tiled convolution based on 2D Winograd transform F(3x3, 6x6).
##                                            Supports only 3x3 kernels.
##
##  @param batch_size The number of images on the input and output of the convolutional layer.
##  @param input_channels The number of channels (AKA features, dimensions) in the input images.
##  @param output_channels The number of channels (AKA features, dimensions) in the output images.
##  @param input_size Size of input images, excluding implicit zero-padding.
##  @param input_padding Implicit zero-padding of input images.
##  @param kernel_size Kernel size.
##  @param[in]  input  A 4D tensor input[batch_size][input_channels][input_size.height][input_size.width].
##  @param[in]  kernel A 4D tensor kernel[output_channels][input_channels][kernel_size.height][kernel_size.width].
##  @param[in]  bias   A 1D array bias[output_channels].
##  @param[out] output A 4D tensor output[batch_size][output_channels][output_size.height][output_size.width] where
##                         output_size.height = (input_padding.top + input_size.height + input_padding.bottom) -
##                                              (kernel_size.height - 1)
##                         output_size.width  = (input_padding.left + input_size.width + input_padding.right) -
##                                              (kernel_size.width - 1)
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##  @param[out] profile An optional pointer to profiling structure.
##                      If provided, the structure would record time spent in different phases of the computation.
##

proc nnp_convolution_output*(algorithm: nnp_convolution_algorithm;
                            batch_size: csize_t; input_channels: csize_t;
                            output_channels: csize_t; input_size: nnp_size;
                            input_padding: nnp_padding; kernel_size: nnp_size;
                            input: ptr cfloat; kernel: ptr cfloat; bias: ptr cfloat;
                            output: ptr cfloat; workspace_buffer: pointer=nil;
                            workspace_size: ptr csize_t=nil; activation: nnp_activation=nnp_activation_identity;
                            activation_parameters: pointer=nil;
                            threadpool: pthreadpool_t=nil; profile: ptr nnp_profile=nil): nnp_status {.
    cdecl, importc: "nnp_convolution_output".}
## *
##  @brief Computes gradient of input of a 2D convolutional layer from gradient of output and kernel tensors.
##  @details This function targets training of convolutional neural networks and performs backward propagation.
##           It is optimized for moderate minibatch sizes (64-128) and can be inefficient on a small minibatch.
##  @param algorithm The type of algorithm to use for convolution. Possible values are:
##
##     - nnp_convolution_algorithm_auto    -- let the function choose the algorithm.
##     - nnp_convolution_algorithm_ft8x8   -- tiled convolution based on 2D Fourier transform with 8x8 blocks.
##                                            Supports kernels up to 8x8.
##     - nnp_convolution_algorithm_ft16x16 -- tiled convolution based on 2D Fourier transform with 16x16 blocks.
##                                            Supports kernels up to 16x16.
##     - nnp_convolution_algorithm_wt8x8   -- tiled convolution based on 2D Winograd transform F(3x3, 6x6).
##                                            Supports only 3x3 kernels.
##
##  @param batch_size The number of images (and their gradients) on the input and output of the convolutional layer.
##  @param input_channels The number of channels (AKA features, dimensions) in the input images (and gradients).
##  @param output_channels The number of channels (AKA features, dimensions) in the output images (and gradients).
##  @param input_size Size of input images and their gradients, excluding implicit zero-padding.
##  @param input_padding Implicit zero-padding of input images.
##  @param kernel_size Kernel size.
##  @param[in]  grad_output A 4D tensor grad_output[batch_size][output_channels][output_size.height][output_size.width]
##                          where
##                            output_size.height = (input_padding.top + input_size.height + input_padding.bottom) -
##                                                 (kernel_size.height - 1)
##                            output_size.width  = (input_padding.left + input_size.width + input_padding.right) -
##                                                 (kernel_size.width - 1)
##  @param[in]  kernel      A 4D tensor kernel[output_channels][input_channels][kernel_size.height][kernel_size.width].
##  @param[out] grad_input  A 4D tensor grad_input[batch_size][input_channels][input_size.height][input_size.width].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##  @param[out] profile An optional pointer to profiling structure.
##                      If provided, the structure would record time spent in different phases of the computation.
##

proc nnp_convolution_input_gradient*(algorithm: nnp_convolution_algorithm;
                                    batch_size: csize_t; input_channels: csize_t;
                                    output_channels: csize_t; input_size: nnp_size;
                                    input_padding: nnp_padding;
                                    kernel_size: nnp_size;
                                    grad_output: ptr cfloat; kernel: ptr cfloat;
                                    grad_input: ptr cfloat;
                                    workspace_buffer: pointer = nil;
                                    workspace_size: ptr csize_t = nil;
                                    activation: nnp_activation = nnp_activation_identity;
                                    activation_parameters: pointer = nil;
                                    threadpool: pthreadpool_t = nil;
                                    profile: ptr nnp_profile = nil): nnp_status {.cdecl,
    importc: "nnp_convolution_input_gradient".}
## *
##  @brief Computes gradient of kernel of a 2D convolutional layer from gradient of output and input tensors.
##  @details This function targets training of convolutional neural networks and performs backward propagation.
##           It is optimized for moderate minibatch sizes (64-128) and can be inefficient on a small minibatch.
##  @param algorithm The type of algorithm to use for convolution. Possible values are:
##
##     - nnp_convolution_algorithm_auto    -- let the function choose the algorithm.
##     - nnp_convolution_algorithm_ft8x8   -- tiled convolution based on 2D Fourier transform with 8x8 blocks.
##                                            Supports kernels up to 8x8.
##     - nnp_convolution_algorithm_ft16x16 -- tiled convolution based on 2D Fourier transform with 16x16 blocks.
##                                            Supports kernels up to 16x16.
##
##  @param batch_size The number of images (and their gradients) on the input and output of the convolutional layer.
##  @param input_channels The number of channels (AKA features, dimensions) in the input images.
##  @param output_channels The number of channels (AKA features, dimensions) in the output images (and gradients).
##  @param input_size Size of input images and their gradients, excluding implicit zero-padding.
##  @param input_padding Implicit zero-padding of input images.
##  @param kernel_size Kernel size.
##  @param[in]  input       A 4D tensor input[batch_size][input_channels][input_size.height][input_size.width].
##  @param[in]  grad_output A 4D tensor grad_output[batch_size][output_channels][output_size.height][output_size.width]
##                          where
##                            output_size.height = (input_padding.top + input_size.height + input_padding.bottom) -
##                                                 (kernel_size.height - 1)
##                            output_size.width  = (input_padding.left + input_size.width + input_padding.right) -
##                                                 (kernel_size.width - 1)
##  @param[out] grad_kernel A 4D tensor
##                          grad_kernel[output_channels][input_channels][kernel_size.height][kernel_size.width].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##  @param[out] profile An optional pointer to profiling structure.
##                      If provided, the structure would record time spent in different phases of the computation.
##

proc nnp_convolution_kernel_gradient*(algorithm: nnp_convolution_algorithm;
                                     batch_size: csize_t; input_channels: csize_t;
                                     output_channels: csize_t; input_size: nnp_size;
                                     input_padding: nnp_padding;
                                     kernel_size: nnp_size; input: ptr cfloat;
                                     grad_output: ptr cfloat;
                                     grad_kernel: ptr cfloat;
                                     workspace_buffer: pointer = nil;
                                     workspace_size: ptr csize_t = nil;
                                     activation: nnp_activation = nnp_activation_identity;
                                     activation_parameters: pointer = nil;
                                     threadpool: pthreadpool_t = nil;
                                     profile: ptr nnp_profile = nil): nnp_status {.cdecl,
    importc: "nnp_convolution_kernel_gradient".}
## *
##  @brief Computes output of a 2D convolutional layer for a single input image and a kernel tensor.
##  @details This function targets prediction with convolutional neural networks and performs forward propagation.
##  @param algorithm The type of algorithm to use for convolution. Possible values are:
##
##     - nnp_convolution_algorithm_auto    -- let the function choose the algorithm.
##     - nnp_convolution_algorithm_ft8x8   -- tiled convolution based on 2D Fourier transform with 8x8 blocks.
##                                            Supports kernels up to 8x8.
##     - nnp_convolution_algorithm_ft16x16 -- tiled convolution based on 2D Fourier transform with 16x16 blocks.
##                                            Supports kernels up to 16x16.
##     - nnp_convolution_algorithm_wt8x8   -- tiled convolution based on 2D Winograd transform F(3x3, 6x6).
##                                            Supports only 3x3 kernels.
##
##  @param transform_strategy A strategy that guides computation of kernel transforms coefficients.
##                            Possible values are:
##
##     - nnp_convolution_transform_strategy_block_based -- do multiplication-accumulations on blocks of transformed
##                                                         coefficients.
##     - nnp_convolution_transform_strategy_tuple_based -- do multiplication-accumulations on tuples of transformed
##                                                         coefficients.
##
##  @param input_channels The number of channels (AKA features, dimensions) in the input image.
##  @param output_channels The number of channels (AKA features, dimensions) in the output image.
##  @param input_size Size of input image, excluding implicit zero-padding.
##  @param input_padding Implicit zero-padding of input image.
##  @param kernel_size Kernel size.
##  @param output_subsampling Subsample region for output, also known as convolution stride.
##  @param[in]  input  A 3D tensor input[input_channels][input_size.height][input_size.width].
##  @param[in]  kernel A 4D tensor kernel[output_channels][input_channels][kernel_size.height][kernel_size.width].
##  @param[in]  bias   A 1D array bias[output_channels].
##  @param[out] output A 3D tensor output[output_channels][output_size.height][output_size.width] where
##                         output_size.height = (input_padding.top + input_size.height + input_padding.bottom) -
##                                              (kernel_size.height - 1)
##                         output_size.width  = (input_padding.left + input_size.width + input_padding.right) -
##                                              (kernel_size.width - 1)
##  @param[in] workspace_buffer Buffer for scratch memory used during computation. Buffer must be aligned on 64 bytes.
##                              If workspace_buffer is NULL and workspace_size is non-NULL, NNPACK would store the size
##                              of required workspace memory at the workspace_size location, and exit without
##                              computations.
##                              If workspace_buffer is NULL and workspace_size is NULL, NNPACK would allocate memory
##                              before and deallocate after this computation, potentially at significant runtime cost.
##  @param[in,out] workspace_size Pointer to the size of workspace buffer.
##                                If workspace_buffer is NULL, NNPACK will write the size of required scratch memory to
##                                the location specified by this pointer.
##                                If workspace_buffer is non-NULL, NNPACK expects workspace_size to specify the size of
##                                the buffer, in bytes.
##                                If workspace_size is NULL, workspace_buffer must be NULL as well. In this case NNPACK
##                                would allocate memory before and deallocate after this computation, potentially at
##                                significant runtime cost.
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##  @param[out] profile An optional pointer to profiling structure.
##                      If provided, the structure would record time spent in different phases of the computation.
##

proc nnp_convolution_inference*(algorithm: nnp_convolution_algorithm;
    transform_strategy: nnp_convolution_transform_strategy; input_channels: csize_t;
                               output_channels: csize_t; input_size: nnp_size;
                               input_padding: nnp_padding; kernel_size: nnp_size;
                               output_subsampling: nnp_size; input: ptr cfloat;
                               kernel: ptr cfloat; bias: ptr cfloat;
                               output: ptr cfloat; workspace_buffer: pointer;
                               workspace_size: ptr csize_t;
                               activation: nnp_activation;
                               activation_parameters: pointer;
                               threadpool: pthreadpool_t; profile: ptr nnp_profile): nnp_status {.
    cdecl, importc: "nnp_convolution_inference".}
## *
##  @brief Computes output of a fully connected layer from input and kernel matrices.
##  @details This function targets training of convolutional neural networks and performs forward propagation.
##           It is optimized for moderate minibatch sizes (64-128) and can be inefficient on a small minibatch.
##           For minibatch size 1, use nnp_fully_connected_inference for optimal performance.
##  @param batch_size The number of vectors on the input and output of the fully connected layer.
##  @param input_channels The number of channels (AKA features, dimensions) in the input matrix.
##  @param output_channels The number of channels (AKA features, dimensions) in the output matrix.
##  @param[in]  input  A 2D matrix input[batch_size][input_channels].
##  @param[in]  kernel A 2D matrix kernel[output_channels][input_channels].
##  @param[out] output A 2D matrix output[batch_size][output_channels].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_fully_connected_output*(batch_size: csize_t; input_channels: csize_t;
                                output_channels: csize_t; input: ptr cfloat;
                                kernel: ptr cfloat; output: ptr cfloat;
                                threadpool: pthreadpool_t;
                                profile: ptr nnp_profile): nnp_status {.cdecl,
    importc: "nnp_fully_connected_output".}
## *
##  @brief Computes output of a fully connected layer for a single input vector and a kernel matrix.
##  @details This function targets prediction with convolutional neural networks and performs forward propagation.
##  @param input_channels The number of channels (AKA features, dimensions) in the input vector.
##  @param output_channels The number of channels (AKA features, dimensions) in the output vector.
##  @param[in]  input  A 1D array input[input_channels] of FP32 elements.
##  @param[in]  kernel A 2D matrix kernel[output_channels][input_channels] of FP32 elements.
##  @param[out] output A 1D array output[output_channels] of FP32 elements.
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_fully_connected_inference*(input_channels: csize_t; output_channels: csize_t;
                                   input: ptr cfloat; kernel: ptr cfloat;
                                   output: ptr cfloat; threadpool: pthreadpool_t): nnp_status {.
    cdecl, importc: "nnp_fully_connected_inference".}
## *
##  @brief Computes output of a fully connected layer for a single input vector and a kernel matrix.
##  @details This function targets prediction with convolutional neural networks and performs forward propagation.
##  @param input_channels The number of channels (AKA features, dimensions) in the input vector.
##  @param output_channels The number of channels (AKA features, dimensions) in the output vector.
##  @param[in]  input  A 1D array input[input_channels] of FP32 elements.
##  @param[in]  kernel A 2D matrix kernel[output_channels][input_channels] of FP16 (ARM alternative format) elements.
##  @param[out] output A 1D array output[output_channels] of FP32 elements.
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_fully_connected_inference_f16f32*(input_channels: csize_t;
    output_channels: csize_t; input: ptr cfloat; kernel: pointer; output: ptr cfloat;
    threadpool: pthreadpool_t): nnp_status {.cdecl,
    importc: "nnp_fully_connected_inference_f16f32".}
## *
##  @brief Computes output of a max-pooling layer for an input tensor.
##  @details This function targets both prediction and training of convolutional neural networks and performs forward
##           propagation. Is is optimized for both large and small minibatch sizes.
##  @param batch_size The number of images on the input and output of the max-pooling layer.
##  @param channels   The number of channels (AKA features, dimensions) in both input and output images.
##  @param input_size Size of input images, excluding implicit zero-padding.
##  @param input_padding Implicit padding of input images. The padding pixels are ignored by the pooling filter, but
##                       affect the output size.
##  @param pooling_size   Size of the pooling filter. Only 2x2 filter are currently supported.
##  @param pooling_stride Stride of the pooling filter. Only 2x2 strides are currently supported.
##  @param[in]  input  A 4D tensor input[batch_size][channels][input_size.height][input_size.width].
##  @param[out] output A 4D tensor output[batch_size][channels][output_size.height][output_size.width] where
##                     output_size.height = ceil(
##                       (input_padding.top + input_size.height + input_padding.bottom - pooling_size.height) /
##                         pooling_stride.height) + 1
##                     output_size.width = ceil(
##                       (input_padding.left + input_size.width + input_padding.right - pooling_size.width) /
##                         pooling_stride.width) + 1
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_max_pooling_output*(batch_size: csize_t; channels: csize_t;
                            input_size: nnp_size; input_padding: nnp_padding;
                            pooling_size: nnp_size; pooling_stride: nnp_size;
                            input: ptr cfloat; output: ptr cfloat;
                            threadpool: pthreadpool_t): nnp_status {.cdecl,
    importc: "nnp_max_pooling_output".}
## *
##  @brief Computes output of a softmax layer for an input matrix.
##  @details This function targets both prediction and training of convolutional neural networks and performs forward
##           propagation. Is is optimized for both large and small minibatch sizes.
##  @param batch_size The number of vectors on the input and output of the softmax layer.
##  @param channels   The number of channels (AKA features, dimensions) in both input and output vectors.
##  @param[in]  input  A 2D matrix input[batch_size][channels].
##  @param[out] output A 2D matrix output[batch_size][channels].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_softmax_output*(batch_size: csize_t; channels: csize_t; input: ptr cfloat;
                        output: ptr cfloat; threadpool: pthreadpool_t): nnp_status {.
    cdecl, importc: "nnp_softmax_output".}
## *
##  @brief Computes output of a rectified linear unit (ReLU) layer for an input matrix.
##  @details This function targets both prediction and training of convolutional neural networks and performs forward
##           propagation. Is is optimized for both large and small minibatch sizes.
##  @param batch_size The number of vectors on the input and output of the ReLU layer.
##  @param channels   The number of channels (AKA features, dimensions) in both input and output matrices.
##  @param[in]  input  A 2D matrix input[batch_size][channels].
##  @param[out] output A 2D matrix output[batch_size][channels].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_relu_output*(batch_size: csize_t; channels: csize_t; input: ptr cfloat;
                     output: ptr cfloat; negative_slope: cfloat;
                     threadpool: pthreadpool_t): nnp_status {.cdecl,
    importc: "nnp_relu_output".}
## *
##  @brief Computes gradient of input of a rectified linear unit (ReLU) layer from gradient of output and input matrices.
##  @details This function targets training of convolutional neural networks and performs backward propagation.
##           Is is optimized for both large and small minibatch sizes.
##  @param batch_size The number of vectors on the input and output of the ReLU layer.
##  @param channels   The number of channels (AKA features, dimensions) in both input and output matrices.
##  @param[in]  input  A 2D matrix input[batch_size][channels].
##  @param[out] output A 2D matrix output[batch_size][channels].
##  @param threadpool A thread pool for parallelization of the computation.
##                    If threadpool is NULL, the computation would run on the caller thread without parallelization.
##

proc nnp_relu_input_gradient*(batch_size: csize_t; channels: csize_t;
                             grad_output: ptr cfloat; input: ptr cfloat;
                             grad_input: ptr cfloat; negative_slope: cfloat;
                             threadpool: pthreadpool_t): nnp_status {.cdecl,
    importc: "nnp_relu_input_gradient".}

# Initialize nnpack automatically
let nn_init_status = nnp_initialize()
if nn_init_status != nnp_status_success:
  raise newException(LibraryError, "Failed to initialize NNPack")
