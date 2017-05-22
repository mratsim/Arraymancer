# Design document

This is a notepad to track ideas, challenges, future work and open issues/limitations of Arraymancer.

## Storage convention

Either C or Fortran contiguous arrays are needed for BLAS optimization for Tensor of Rank 1 or 2
* C_contiguous: Row Major - Default. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
* F_contiguous: Col Major. First index is the fastest changing (rows in 2D, depth in 3D) - Rows (fastest), Columns, Depth (slowest)
* Universal: Any strides

## Pending issues
* Slices have universal strides and cannot be use currently with BLAS operations.
BLIS (A BLAS-like library with universal strided matrices) can be considered: https://github.com/flame/blis/wiki/FAQ

## Memory considerations
* Current CPUs cache line is 64 bytes. The Tensor data structure at 32 bytes has an ideal size.
However, every time we retrieve its shape and strides there is a pointer resolution + bounds checking for seq with constant length. See Data structure considerations section

* Most copy operations (from nested arrays/seq, for slice assignments from nested arrays/seq or Tensor) uses iterators and avoid intermediate representation)

## Data structure considerations

* Shape and strides have a static size known at runtime. They might be best implemented as VLAs (Variable Length Array) from an indirection point of view. Inconvenient: 2 tensors will not fit in a cache line.

* `data` is currently stored in a "seq" that always deep copy on var assignement. It doesn't copy on let assignement.

References: [Copy semantics](https://forum.nim-lang.org/t/1793/5) "Parameter passing doesn't copy, var x = foo() doesn't copy but moves let x = y doesn't copy but moves, var x = y does copy but I can use shallowCopy instead of = for that."


As such using seq is far easier than implementing my own shallowCopy / refcounting code which would introduce the following questions:
- How to make sure we can modify in-place if shallow copy is allowed or a ref seq/object is used?
- To avoid reference counting, would it be better to always copy-on-write, in that case wouldn't it be better to pay the cost upfront on assignment?
- How hard will it be to maintain Arraymancer and avoid bugs because a copy-on-write was missed.

    From Scipy: https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#reference-counting

    "If you mis-handle reference counts you can get problems from memory-leaks to segmentation faults.  The only strategy I know of to handle reference counts correctly is blood, sweat, and tears."
Nim GC perf: https://gist.github.com/dom96/77b32e36b62377b2e7cadf09575b8883

In-depth [read](http://blog.stablekernel.com/when-to-use-value-types-and-reference-types-in-swift) (for Swift but applicable): performance, safety, usability

## Performance consideration
* Add OpenMP pragma for parallel computing on `fmap` and self-implemented BLAS operations.
    How to detect that OpenMP overhead is worth it?

## Features

* How to implement integer matrix multiplication and matrix-vector multiplication.
    1. Convert to float64, use BLAS, convert back to int. No issue for int32 has them all. Int64 may lose some precision.
    2. Implement a cache-oblivious matrix multiplication. Implementation in [Julia](https://github.com/JuliaLang/julia/blob/master/base/linalg/matmul.jl#L490). [Paper](http://ocw.raf.edu.rs/courses/electrical-engineering-and-computer-science/6-895-theory-of-parallel-systems-sma-5509-fall-2003/readings/cach_oblvs_thsis.pdf).
* How to implement non-contiguous matrix multiplication and matrix-vector multiplication.
    1. Cache oblivious and any stride generic matrix multiplication (see [Universal stride cache oblivious GEMM in Javascript](https://0fps.net/2013/05/28/cache-oblivious-array-operations/))
    2. Convert the tensor to C major layout with the strided iterator.

## TODO
1. Operations on Universal strides
2. GPU support: Cuda and Magma first. OpenCL when AMD gets its act together.
3. BLAS operation fusion: `transpose(A) * B` or `Ax + y` should be fused in one operation.

999. (Needs thinking) Support sparse matrices. There is Magma and CuSparse for GPU. What to use? Interface should be similar to BLAS and should compile on ARM/embedded devices like Jetson TX1.

## Ideas rejected

1. Have the rank of the Tensor be part of its type. Rejected because impractical for function chaining.
    Advantage: Dispatch and compatibility checking at compile time (Matrix * Matrix and Matrix * Vec)
2. Have the kind of stride (C_contiguous, F_contiguous) be part of its type. Rejected because impractical for function chaining. Furthermore this should not be exposed to the users as it's an implementation detail.

3. Implement offsets and iterator using pointers.
Indexing with a strided array is basically doing a dot product. With a 3x3 matrix, strides are [3,1], in memory, element at position [1,2] will be at 3x1 + 1 x 2 -> 5th position (i.e. we did a dot product)

    > 0 1 2
    
    > 3 4 5

    > 6 7 8

After transposition, strides are [1, 3] and matrix shape:

    > 0 3 6

    > 1 4 7

    > 2 5 8

but the corresponding order in memory is still as before transposition. So pointer must jump by 3 twice, then minus 5, then jump by 3 twice, then minus 5. There is probably a mathematical formula behind but it's much easier and less error-prone to do a dot product, especially for high dimensions.

Since we will do a dot product anyway instead of shifting a pointer by a constant, just doing regular array/sequence indexing is better as we get automatic bounds checking, Nim future improvements and it's much easier to copy a Tensor, no need to recalculate the pointer address. We just need a way to provide a pointer to the beginning of the data to BLAS.

Perf note: from a perf point of view, (integer ?) dot product is vectorized on CPU and GPU, the stride seq will stay in cache, so perf is probably bounded by the non-contiguous memory access. Moving a pointer sometimes by x, sometimes by y, sometimes the other way would also be bounded by memory access (provided a correct and probably cumber some implementation)

## To watch:

* Nim's linalg, nimcuda and nimblas
* Collenchyma/Parenchyma
* Numpy
* Dask
* Mir ndslice and Mir GLAS
* OpenBLAS, Magma, libelemental, Eigen
* BLIS / ulmBLAS
* scijs/ndarray and scijs/cwise (especially universal stride cache oblivious ndarray)
