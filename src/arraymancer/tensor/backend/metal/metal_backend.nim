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

import ./metal_buffer, std/tables

{.passC: "-x objective-c".}
{.passL: "-framework Metal".}
{.passL: "-framework MetalPerformanceShaders".}
{.passL: "-framework Foundation".}

{.emit: """
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

id<MTLDevice> arraymancer_getDefaultDevice() {
  NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
  if (devices && [devices count] > 0) {
    id<MTLDevice> device = [devices objectAtIndex:0];
    [devices release];
    return device;
  }
  return nil;
}

id<MTLCommandQueue> arraymancer_createCommandQueue(id<MTLDevice> device) {
  return [device newCommandQueue];
}

id<MTLBuffer> arraymancer_createBuffer(id<MTLDevice> device, size_t length) {
  return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

void* arraymancer_getBufferContents(id<MTLBuffer> buffer) {
  return [buffer contents];
}

void arraymancer_copyToBuffer(id<MTLBuffer> buffer, const void* data, size_t length) {
  memcpy([buffer contents], data, length);
}

void arraymancer_copyFromBuffer(id<MTLBuffer> buffer, void* data, size_t length) {
  memcpy(data, [buffer contents], length);
}

id<MTLLibrary> arraymancer_createLibraryFromSource(id<MTLDevice> device, const char* source) {
  NSError* error = nil;
  NSString* sourceString = [NSString stringWithUTF8String:source];
  id<MTLLibrary> library = [device newLibraryWithSource:sourceString options:nil error:&error];
  if (error) {
    NSLog(@"Error creating library: %@", error);
    return nil;
  }
  return library;
}

id<MTLFunction> arraymancer_getFunction(id<MTLLibrary> library, const char* name) {
  NSString* funcName = [NSString stringWithUTF8String:name];
  return [library newFunctionWithName:funcName];
}

id<MTLComputePipelineState> arraymancer_createComputePipeline(id<MTLDevice> device, id<MTLFunction> function) {
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (error) {
    NSLog(@"Error creating pipeline: %@", error);
    return nil;
  }
  return pipeline;
}

id<MTLCommandBuffer> arraymancer_createCommandBuffer(id<MTLCommandQueue> queue) {
  return [queue commandBuffer];
}

id<MTLComputeCommandEncoder> arraymancer_createComputeEncoder(id<MTLCommandBuffer> buffer) {
  return [buffer computeCommandEncoder];
}

void arraymancer_setComputePipeline(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline) {
  [encoder setComputePipelineState:pipeline];
}

void arraymancer_setBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, uint index) {
  [encoder setBuffer:buffer offset:0 atIndex:index];
}

void arraymancer_setBytes(id<MTLComputeCommandEncoder> encoder, const void* bytes, size_t length, uint index) {
  [encoder setBytes:bytes length:length atIndex:index];
}

void arraymancer_dispatchThreads1D(id<MTLComputeCommandEncoder> encoder, size_t size, size_t threadgroupSize) {
  MTLSize gridSize = MTLSizeMake(size, 1, 1);
  MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
}

void arraymancer_dispatchThreads2D(id<MTLComputeCommandEncoder> encoder, size_t width, size_t height, size_t threadgroupWidth, size_t threadgroupHeight) {
  MTLSize gridSize = MTLSizeMake(width, height, 1);
  MTLSize tgSize = MTLSizeMake(threadgroupWidth, threadgroupHeight, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
}

void arraymancer_endEncoding(id<MTLComputeCommandEncoder> encoder) {
  [encoder endEncoding];
}

void arraymancer_commitCommandBuffer(id<MTLCommandBuffer> buffer) {
  [buffer commit];
}

void arraymancer_waitForCompletion(id<MTLCommandBuffer> buffer) {
  [buffer waitUntilCompleted];
}

id<MTLBlitCommandEncoder> arraymancer_createBlitEncoder(id<MTLCommandBuffer> buffer) {
  return [buffer blitCommandEncoder];
}

void arraymancer_copyBuffer(id<MTLBlitCommandEncoder> encoder, id<MTLBuffer> src, id<MTLBuffer> dst, size_t length) {
  [encoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:length];
}

void arraymancer_endBlitEncoding(id<MTLBlitCommandEncoder> encoder) {
  [encoder endEncoding];
}
""".}

proc getDefaultDevice(): MTLDevice {.importc: "arraymancer_getDefaultDevice", nodecl.}
proc createCommandQueue(device: MTLDevice): MTLCommandQueue {.importc: "arraymancer_createCommandQueue", nodecl.}
proc createBuffer(device: MTLDevice, length: csize_t): MTLBuffer {.importc: "arraymancer_createBuffer", nodecl.}
proc getBufferContents(buffer: MTLBuffer): pointer {.importc: "arraymancer_getBufferContents", nodecl.}
proc copyToBuffer(buffer: MTLBuffer, data: pointer, length: csize_t) {.importc: "arraymancer_copyToBuffer", nodecl.}
proc copyFromBuffer(buffer: MTLBuffer, data: pointer, length: csize_t) {.importc: "arraymancer_copyFromBuffer", nodecl.}
proc createLibraryFromSource(device: MTLDevice, source: cstring): MTLLibrary {.importc: "arraymancer_createLibraryFromSource", nodecl.}
proc getFunction(library: MTLLibrary, name: cstring): MTLFunction {.importc: "arraymancer_getFunction", nodecl.}
proc createComputePipeline(device: MTLDevice, function: MTLFunction): MTLComputePipelineState {.importc: "arraymancer_createComputePipeline", nodecl.}
proc createCommandBuffer(queue: MTLCommandQueue): MTLCommandBuffer {.importc: "arraymancer_createCommandBuffer", nodecl.}
proc createComputeEncoder(buffer: MTLCommandBuffer): MTLComputeCommandEncoder {.importc: "arraymancer_createComputeEncoder", nodecl.}
proc setComputePipeline(encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState) {.importc: "arraymancer_setComputePipeline", nodecl.}
proc setBuffer(encoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: cuint) {.importc: "arraymancer_setBuffer", nodecl.}
proc setBytes(encoder: MTLComputeCommandEncoder, bytes: pointer, length: csize_t, index: cuint) {.importc: "arraymancer_setBytes", nodecl.}
proc dispatchThreads1D(encoder: MTLComputeCommandEncoder, size, threadgroupSize: csize_t) {.importc: "arraymancer_dispatchThreads1D", nodecl.}
proc dispatchThreads2D(encoder: MTLComputeCommandEncoder, width, height, threadgroupWidth, threadgroupHeight: csize_t) {.importc: "arraymancer_dispatchThreads2D", nodecl.}
proc endEncoding(encoder: MTLComputeCommandEncoder) {.importc: "arraymancer_endEncoding", nodecl.}
proc commitCommandBuffer(buffer: MTLCommandBuffer) {.importc: "arraymancer_commitCommandBuffer", nodecl.}
proc waitForCompletion(buffer: MTLCommandBuffer) {.importc: "arraymancer_waitForCompletion", nodecl.}
proc createBlitEncoder(buffer: MTLCommandBuffer): MTLBlitCommandEncoder {.importc: "arraymancer_createBlitEncoder", nodecl.}
proc copyBuffer(encoder: MTLBlitCommandEncoder, src, dst: MTLBuffer, length: csize_t) {.importc: "arraymancer_copyBuffer", nodecl.}
proc endBlitEncoding(encoder: MTLBlitCommandEncoder) {.importc: "arraymancer_endBlitEncoding", nodecl.}

type
  MetalContext* = object
    device*: MTLDevice
    commandQueue*: MTLCommandQueue
    bufferPool*: MetalBufferPool
    library*: MTLLibrary
    kernelCache*: Table[string, MTLComputePipelineState]
    initialized*: bool

var metalContext*: MetalContext

# Embedded Metal kernel source code
const metalKernelSource = slurp("./metal_kernels.metal")

proc initMetalContext*() =
  if metalContext.initialized:
    return
  metalContext.device = getDefaultDevice()
  if metalContext.device == nil:
    raise newException(IOError, "Failed to get default Metal device. " &
      "This can happen if:\n" &
      "  - You're running in a sandboxed/headless environment\n" &
      "  - You don't have an Apple Silicon Mac (M1/M2/M3)\n" &
      "  - The Metal framework is not accessible")
  metalContext.commandQueue = createCommandQueue(metalContext.device)
  metalContext.bufferPool = initMetalBufferPool()
  metalContext.library = createLibraryFromSource(metalContext.device, metalKernelSource.cstring)
  if metalContext.library == nil:
    raise newException(IOError, "Failed to create Metal library from kernel source")
  metalContext.kernelCache = initTable[string, MTLComputePipelineState]()
  metalContext.initialized = true

proc tryInitMetalContext*(): bool =
  ## Try to initialize Metal context, return true if successful
  if metalContext.initialized:
    return true
  try:
    initMetalContext()
    return true
  except:
    return false

proc isMetalAvailable*(): bool =
  if not metalContext.initialized:
    return tryInitMetalContext()
  return metalContext.initialized and metalContext.device != nil

proc getKernel*(kernelName: string): MTLComputePipelineState =
  ## Get or create a compute pipeline for a kernel function
  if not metalContext.initialized:
    initMetalContext()
  
  if metalContext.kernelCache.hasKey(kernelName):
    return metalContext.kernelCache[kernelName]
  
  let function = getFunction(metalContext.library, kernelName.cstring)
  if function == nil:
    raise newException(ValueError, "Kernel function not found: " & kernelName)
  
  let pipeline = createComputePipeline(metalContext.device, function)
  if pipeline == nil:
    raise newException(IOError, "Failed to create compute pipeline for: " & kernelName)
  
  metalContext.kernelCache[kernelName] = pipeline
  return pipeline

proc createMetalBuffer*(length: int): MetalBuffer =
  if not metalContext.initialized:
    initMetalContext()
  result = MetalBuffer()
  result.buffer = createBuffer(metalContext.device, csize_t(length))
  result.length = length
  result.devicePtr = getBufferContents(result.buffer)

proc uploadToBuffer*(mb: MetalBuffer, data: pointer, length: int) =
  copyToBuffer(mb.buffer, data, csize_t(length))

proc downloadFromBuffer*(mb: MetalBuffer, data: pointer, length: int) =
  copyFromBuffer(mb.buffer, data, csize_t(length))

proc copyBuffer*(src, dst: MetalBuffer, length: int) =
  ## Copy data from one Metal buffer to another
  if not metalContext.initialized:
    initMetalContext()
  
  let commandBuffer = createCommandBuffer(metalContext.commandQueue)
  let blitEncoder = createBlitEncoder(commandBuffer)
  
  copyBuffer(blitEncoder, src.buffer, dst.buffer, csize_t(length))
  endBlitEncoding(blitEncoder)
  commitCommandBuffer(commandBuffer)
  waitForCompletion(commandBuffer)

proc executeElementwiseKernel*(
  kernelName: string,
  buffers: seq[MetalBuffer],
  constants: seq[pointer],
  constantSizes: seq[int],
  gridSize: int
) =
  ## Execute a 1D elementwise kernel
  if not metalContext.initialized:
    initMetalContext()
  
  let pipeline = getKernel(kernelName)
  let commandBuffer = createCommandBuffer(metalContext.commandQueue)
  let encoder = createComputeEncoder(commandBuffer)
  
  setComputePipeline(encoder, pipeline)
  
  # Set buffers
  for i, buf in buffers:
    setBuffer(encoder, buf.buffer, cuint(i))
  
  # Set constants
  for i, constPtr in constants:
    setBytes(encoder, constPtr, csize_t(constantSizes[i]), cuint(buffers.len + i))
  
  # Dispatch threads
  let threadgroupSize = 256
  dispatchThreads1D(encoder, csize_t(gridSize), csize_t(threadgroupSize))
  
  endEncoding(encoder)
  commitCommandBuffer(commandBuffer)
  waitForCompletion(commandBuffer)

proc executeGemmKernel*(
  kernelName: string,
  A, B, C: MetalBuffer,
  M, N, K: int,
  alpha, beta: float32
) =
  ## Execute a GEMM kernel
  if not metalContext.initialized:
    initMetalContext()
  
  let pipeline = getKernel(kernelName)
  let commandBuffer = createCommandBuffer(metalContext.commandQueue)
  let encoder = createComputeEncoder(commandBuffer)
  
  setComputePipeline(encoder, pipeline)
  
  # Set buffers
  setBuffer(encoder, A.buffer, 0)
  setBuffer(encoder, B.buffer, 1)
  setBuffer(encoder, C.buffer, 2)
  
  # Set constants
  var m = M
  var n = N
  var k = K
  var a = alpha
  var b = beta
  setBytes(encoder, addr m, csize_t(sizeof(int)), 3)
  setBytes(encoder, addr n, csize_t(sizeof(int)), 4)
  setBytes(encoder, addr k, csize_t(sizeof(int)), 5)
  setBytes(encoder, addr a, csize_t(sizeof(float32)), 6)
  setBytes(encoder, addr b, csize_t(sizeof(float32)), 7)
  
  # Dispatch threads
  let threadgroupWidth = 8
  let threadgroupHeight = 8
  dispatchThreads2D(encoder, csize_t(N), csize_t(M), csize_t(threadgroupWidth), csize_t(threadgroupHeight))
  
  endEncoding(encoder)
  commitCommandBuffer(commandBuffer)
  waitForCompletion(commandBuffer)

proc metalGemm*[T: SomeFloat](
  transA, transB: bool,
  M, N, K: int,
  alpha: T,
  A: MetalBuffer, lda: int,
  B: MetalBuffer, ldb: int,
  beta: T,
  C: MetalBuffer, ldc: int
) =
  if not metalContext.initialized:
    initMetalContext()

  # Metal only supports float32, not float64
  when T is float32:
    var kernelName: string
    if transA and not transB:
      kernelName = "gemm_a_transpose_f32"
    elif not transA and transB:
      kernelName = "gemm_b_transpose_f32"
    else:
      kernelName = "gemm_naive_f32"
    
    executeGemmKernel(kernelName, A, B, C, M, N, K, alpha, beta)
  else:
    raise newException(ValueError, "Metal only supports float32, not float64")

proc metalGemv*[T: SomeFloat](
  trans: bool,
  M, N: int,
  alpha: T,
  A: MetalBuffer, lda: int,
  x: MetalBuffer, incx: int,
  beta: T,
  y: MetalBuffer, incy: int
) =
  ## Matrix-Vector multiplication: y = alpha * A * x + beta * y
  ## Column-major layout (Fortran style)
  if not metalContext.initialized:
    initMetalContext()

  # Metal only supports float32, not float64
  when T is float32:
    var m = M
    var n = N
    var a = float32(alpha)
    var b = float32(beta)
    
    let pipeline = getKernel("gemv_f32")
    let commandBuffer = createCommandBuffer(metalContext.commandQueue)
    let encoder = createComputeEncoder(commandBuffer)
    
    setComputePipeline(encoder, pipeline)
    setBuffer(encoder, A.buffer, 0)
    setBuffer(encoder, x.buffer, 1)
    setBuffer(encoder, y.buffer, 2)
    setBytes(encoder, addr m, csize_t(sizeof(int)), 3)
    setBytes(encoder, addr n, csize_t(sizeof(int)), 4)
    setBytes(encoder, addr a, csize_t(sizeof(float32)), 5)
    setBytes(encoder, addr b, csize_t(sizeof(float32)), 6)
    
    # Dispatch threads - one thread per output element (M rows)
    let threadgroupSize = 256
    dispatchThreads1D(encoder, csize_t(M), csize_t(threadgroupSize))
    
    endEncoding(encoder)
    commitCommandBuffer(commandBuffer)
    waitForCompletion(commandBuffer)
  else:
    raise newException(ValueError, "Metal only supports float32, not float64")

proc metalElementwise*[T: SomeFloat](
  op: string,
  A, B, C: MetalBuffer,
  size: int
) =
  if not metalContext.initialized:
    initMetalContext()

  # Metal only supports float32, not float64
  when T is float32:
    var kernelName: string
    case op
    of "add": kernelName = "elementwise_add_f32"
    of "sub": kernelName = "elementwise_sub_f32"
    of "mul": kernelName = "elementwise_mul_f32"
    of "div": kernelName = "elementwise_div_f32"
    else: raise newException(ValueError, "Unknown elementwise operation: " & op)
    
    var sizeVar = size
    executeElementwiseKernel(kernelName, @[A, B, C], @[cast[pointer](addr sizeVar)], @[sizeof(int)], size)
  else:
    raise newException(ValueError, "Metal only supports float32, not float64")

proc metalScalarMul*[T: SomeFloat](
  A, C: MetalBuffer,
  scalar: T,
  size: int
) =
  if not metalContext.initialized:
    initMetalContext()
  
  var kernelName = "scalar_mul_f32"
  var sizeVar = size
  var scalarVar = float32(scalar)
  
  let pipeline = getKernel(kernelName)
  let commandBuffer = createCommandBuffer(metalContext.commandQueue)
  let encoder = createComputeEncoder(commandBuffer)
  
  setComputePipeline(encoder, pipeline)
  setBuffer(encoder, A.buffer, 0)
  setBuffer(encoder, C.buffer, 1)
  setBytes(encoder, addr scalarVar, csize_t(sizeof(float32)), 2)
  setBytes(encoder, addr sizeVar, csize_t(sizeof(int)), 3)
  
  let threadgroupSize = 256
  dispatchThreads1D(encoder, csize_t(size), csize_t(threadgroupSize))
  
  endEncoding(encoder)
  commitCommandBuffer(commandBuffer)
  waitForCompletion(commandBuffer)

proc metalScalarAdd*[T: SomeFloat](
  A, C: MetalBuffer,
  scalar: T,
  size: int
) =
  ## Element-wise addition of a scalar to a tensor
  if not metalContext.initialized:
    initMetalContext()
  
  var kernelName = "scalar_add_f32"
  var sizeVar = size
  var scalarVar = float32(scalar)
  
  let pipeline = getKernel(kernelName)
  let commandBuffer = createCommandBuffer(metalContext.commandQueue)
  let encoder = createComputeEncoder(commandBuffer)
  
  setComputePipeline(encoder, pipeline)
  setBuffer(encoder, A.buffer, 0)
  setBuffer(encoder, C.buffer, 1)
  setBytes(encoder, addr scalarVar, csize_t(sizeof(float32)), 2)
  setBytes(encoder, addr sizeVar, csize_t(sizeof(int)), 3)
  
  let threadgroupSize = 256
  dispatchThreads1D(encoder, csize_t(size), csize_t(threadgroupSize))
  
  endEncoding(encoder)
  commitCommandBuffer(commandBuffer)
  waitForCompletion(commandBuffer)

proc metalContiguousCopy*[T: SomeFloat](
  src, dst: MetalBuffer,
  shape, strides: seq[int],
  totalSize: int
) =
  ## Copy data from a strided tensor to a contiguous buffer
  if not metalContext.initialized:
    initMetalContext()
  
  when T is float32:
    let rank = shape.len
    
    # Create buffers for shape and strides
    let shapeBuffer = createMetalBuffer(rank * sizeof(int32))
    let stridesBuffer = createMetalBuffer(rank * sizeof(int32))
    
    # Upload shape and strides
    var shapeData = newSeq[int32](rank)
    var stridesData = newSeq[int32](rank)
    for i in 0 ..< rank:
      shapeData[i] = int32(shape[i])
      stridesData[i] = int32(strides[i])
    
    uploadToBuffer(shapeBuffer, addr shapeData[0], rank * sizeof(int32))
    uploadToBuffer(stridesBuffer, addr stridesData[0], rank * sizeof(int32))
    
    var totalSizeVar = int32(totalSize)
    var rankVar = int32(rank)
    
    let pipeline = getKernel("contiguous_copy_f32")
    let commandBuffer = createCommandBuffer(metalContext.commandQueue)
    let encoder = createComputeEncoder(commandBuffer)
    
    setComputePipeline(encoder, pipeline)
    setBuffer(encoder, src.buffer, 0)
    setBuffer(encoder, dst.buffer, 1)
    setBuffer(encoder, shapeBuffer.buffer, 2)
    setBuffer(encoder, stridesBuffer.buffer, 3)
    setBytes(encoder, addr rankVar, csize_t(sizeof(int32)), 4)
    setBytes(encoder, addr totalSizeVar, csize_t(sizeof(int32)), 5)
    
    let threadgroupSize = 256
    dispatchThreads1D(encoder, csize_t(totalSize), csize_t(threadgroupSize))
    
    endEncoding(encoder)
    commitCommandBuffer(commandBuffer)
    waitForCompletion(commandBuffer)
  else:
    raise newException(ValueError, "Metal only supports float32, not float64")

proc metalDot*[T: SomeFloat](A, B: MetalBuffer, size: int): T =
  ## Compute dot product of two vectors using GPU reduction
  if not metalContext.initialized:
    initMetalContext()
  
  when T is float32:
    let kernelName = "vector_dot_f32"
    var sizeVar = size
    
    # Create a buffer for partial results (one per threadgroup)
    let threadgroupSize = 256
    let numGroups = (size + threadgroupSize - 1) div threadgroupSize
    var partialResult = newSeq[float32](numGroups)
    let partialBuffer = createMetalBuffer(numGroups * sizeof(float32))
    
    let pipeline = getKernel(kernelName)
    let commandBuffer = createCommandBuffer(metalContext.commandQueue)
    let encoder = createComputeEncoder(commandBuffer)
    
    setComputePipeline(encoder, pipeline)
    setBuffer(encoder, A.buffer, 0)
    setBuffer(encoder, B.buffer, 1)
    setBuffer(encoder, partialBuffer.buffer, 2)
    setBytes(encoder, addr sizeVar, csize_t(sizeof(int)), 3)
    
    dispatchThreads1D(encoder, csize_t(size), csize_t(threadgroupSize))
    
    endEncoding(encoder)
    commitCommandBuffer(commandBuffer)
    waitForCompletion(commandBuffer)
    
    # Download partial results and sum them on CPU
    downloadFromBuffer(partialBuffer, addr partialResult[0], numGroups * sizeof(float32))
    
    var sum: float32 = 0.0
    for val in partialResult:
      sum += val
    
    return T(sum)
  else:
    raise newException(ValueError, "Metal only supports float32, not float64")

proc preferCpuGemm*(m, n, k: int): bool =
  const THRESHOLD = 128 * 128 * 128
  return m * n * k < THRESHOLD
