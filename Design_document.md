# Design document

This is a notepad to track ideas, challenges, future work and open issues/limitations of Arraymancer.

## Storage convention

Either C or Fortran contiguous arrays are needed for BLAS optimization for Tensor of Rank 1 or 2
* C_contiguous: Row Major - Default. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
* F_contiguous: Col Major. First index is the fastest changing (rows in 2D, depth in 3D) - Rows (fastest), Columns, Depth (slowest)
* Universal: Any strides

## Pending issues
* There is no matrix multiplications and Matrix-Vector multiplication for integers.
  You can convert them to `float64` before and then use floating-point BLAS (will cover `int32` without precision loss).

## Memory considerations
* Current CPUs cache line is 64 bytes. The Tensor data structure at 32 bytes has an ideal size.
However, every time we retrieve its shape and strides there is a pointer resolution + bounds checking for seq with constant length. See Data structure considerations section.

* Most copy operations (from nested arrays/seq, for slice assignments from nested arrays/seq or Tensor) uses iterators and avoid intermediate representation.

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
* Limit branching: use `when` or static dispatch instead of `if` for conditions that can be tested at compile-time
* `SlicerMut` is implemented in a readable but inefficient way
```Nim
proc slicerMut*[B, T](t: var Tensor[B, T], slices: varargs[SteppedSlice], val: T) {.noSideEffect.}=
  ## Assign the value to the whole slice
  let sliced = t.slicer(slices)
  for real_idx in sliced.real_indices:
    t.data[real_idx] = val
````
  Instead we could save the current shape/strides of t, use `slicer` then use `mitems` to directly assign the value
  and then reverting shape/strides. That would avoid recomputing t.data[real_idx] at the price of copying shape/stride twice
  and readability.

## Coding-style
* Prefer `when` to `if` for compile-time evaluation
* Let the compiler do its job:
    - proc everywhere, without the `inline` tag
    - template if proc does not work or to access an object field
    - macro as last resort to manipulate AST tree or rewrite code
* Readibility, maintainability and performance are very important (in no particular order)
* Use functional constructs like `map`, `scanr` instead of `for loop` when you don't need side-effects or a iterator

## Features

* How to implement integer matrix multiplication and matrix-vector multiplication.
    1. Convert to float64, use BLAS, convert back to int. No issue for int32 has them all. Int64 may lose some precision.
    2. Implement a cache-oblivious matrix multiplication. Implementation in [Julia](https://github.com/JuliaLang/julia/blob/master/base/linalg/matmul.jl#L490). [Paper](http://ocw.raf.edu.rs/courses/electrical-engineering-and-computer-science/6-895-theory-of-parallel-systems-sma-5509-fall-2003/readings/cach_oblvs_thsis.pdf).


## TODO
1. Array creation utilities (zeros, ones, zeros_like, random ...)
2. Axis iterators
3. GPU support: Cuda and Magma first. OpenCL when AMD gets its act together.
4. BLAS operation fusion: `transpose(A) * B` or `Ax + y` should be fused in one operation.
5. Implement GEMM and GEMV for integers

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
* Scijs/ndarray and scijs/cwise (especially universal stride cache oblivious ndarray)
* Cuda-on-cl
