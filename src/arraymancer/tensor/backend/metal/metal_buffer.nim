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

type
  MTLDevice* = pointer
  MTLCommandQueue* = pointer
  MTLCommandBuffer* = pointer
  MTLComputeCommandEncoder* = pointer
  MTLBlitCommandEncoder* = pointer
  MTLBuffer* = pointer
  MTLComputePipelineState* = pointer
  MTLFunction* = pointer
  MTLLibrary* = pointer

  MetalBufferObj* = object
    buffer*: MTLBuffer
    length*: int
    devicePtr*: pointer

  MetalBuffer* = ref MetalBufferObj

  BufferPoolEntry* = object
    buffer*: MetalBuffer
    size*: int
    lastUsed*: int64

  MetalBufferPool* = object
    entries*: seq[BufferPoolEntry]
    maxSize*: int
    currentSize*: int
    accessCounter*: int64

const
  DEFAULT_BUFFER_POOL_MAX_SIZE* = 1024 * 1024 * 1024
  BUFFER_POOL_ENTRY_MAX_AGE* = 1000

proc initMetalBufferPool*(maxSize: int = DEFAULT_BUFFER_POOL_MAX_SIZE): MetalBufferPool =
  result.maxSize = maxSize
  result.currentSize = 0
  result.accessCounter = 0

proc acquireBuffer*(pool: var MetalBufferPool, size: int, device: MTLDevice): MetalBuffer =
  pool.accessCounter.inc

  for i in 0 ..< pool.entries.len:
    if pool.entries[i].size >= size and pool.entries[i].buffer != nil:
      result = pool.entries[i].buffer
      pool.entries[i].lastUsed = pool.accessCounter
      return

  result = nil

proc releaseBuffer*(pool: var MetalBufferPool, buffer: MetalBuffer) =
  if buffer == nil:
    return

  pool.accessCounter.inc

  if pool.currentSize + buffer.length > pool.maxSize:
    var oldestIdx = -1
    var oldestAccess = int64.high
    for i in 0 ..< pool.entries.len:
      if pool.entries[i].lastUsed < oldestAccess:
        oldestAccess = pool.entries[i].lastUsed
        oldestIdx = i

    if oldestIdx >= 0:
      pool.currentSize -= pool.entries[oldestIdx].size
      pool.entries.del(oldestIdx)

  pool.entries.add(BufferPoolEntry(
    buffer: buffer,
    size: buffer.length,
    lastUsed: pool.accessCounter
  ))
  pool.currentSize += buffer.length

proc cleanupPool*(pool: var MetalBufferPool) =
  var i = 0
  while i < pool.entries.len:
    if pool.accessCounter - pool.entries[i].lastUsed > BUFFER_POOL_ENTRY_MAX_AGE:
      pool.currentSize -= pool.entries[i].size
      pool.entries.del(i)
    else:
      i.inc
