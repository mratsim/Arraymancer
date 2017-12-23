import  tables,
        intsets,
        deques,
        times

type
  BlockSize = int
  Allocator = proc (size: Natural): pointer {.noconv.}
  Deallocator = proc (p: pointer) {.noconv.}

  CachedBlock = tuple[epoch: Time, address: ByteAddress, size: BlockSize]
    ## Describe a contiguous allocated and available memory chunk in the pool:
    ##   - starting timestamp of availability,
    ##   - starting address,
    ##   - size in bytes

  DecayingObjectPool = object
    freeBlocks: Table[BlockSize, ByteAddress]
    evictionQueue: Deque[CachedBlock]
    lazyReused: IntSet
    allocator: Allocator
    deallocator: Deallocator
    ## Object pool / caching allocator with timed eviction
    ##   - Keep track of available blocksizes in the pool.
    ##   - Free blocks that are too old expire and are returned to the OS/devices via the deallocator.
    ##   - Eviction is managed by a queue however reallocated objects must be
    ##     be removed from the EvictionQueue as well and can be at arbitrary positions.
    ##     To avoid costly deletion in the middle of the queue, reused objects are tracked
    ##     in lazyReused and will be removed lazily from the queue when they expire but will not
    ##     trigger the deallocator.
    when defined(debug) or defined(test):
      unusedMem: Natural
      usedMem: Natural
      nbAllocations: Natural
      nbDeallocations: Natural
      cacheHits: Natural
      cacheMisses: Natural

proc initDecayingObjectPool(proc_alloc: Allocator, proc_dealloc: Deallocator): DecayingObjectPool =
  result.allocator = proc_alloc
  result.deallocator = proc_dealloc


when isMainModule:
  let foo = initDecayingObjectPool(alloc0, dealloc)
  echo foo