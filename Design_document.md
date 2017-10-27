# Design document

This is a notepad to track ideas, challenges, future work and open issues/limitations of Arraymancer.

<!-- TOC -->

- [Design document](#design-document)
  - [Storage convention](#storage-convention)
  - [Pending issues](#pending-issues)
  - [Data structure considerations](#data-structure-considerations)
  - [Memory/Perf considerations](#memoryperf-considerations)
  - [CUDA considerations](#cuda-considerations)
  - [Coding-style](#coding-style)
  - [Future features](#future-features)
    - [Software features](#software-features)
    - [Backend/hardware features](#backendhardware-features)
  - [Ideas rejected](#ideas-rejected)
    - [Having an unified Tensor type instead of Tensor, CudaTensor, etc.](#having-an-unified-tensor-type-instead-of-tensor-cudatensor-etc)
    - [Have the rank of the Tensor be part of its type.](#have-the-rank-of-the-tensor-be-part-of-its-type)
    - [Have the kind of stride (C_contiguous, F_contiguous) be part of its type.](#have-the-kind-of-stride-c_contiguous-f_contiguous-be-part-of-its-type)
    - [Implement offsets and iterator using pointers.](#implement-offsets-and-iterator-using-pointers)
    - [Shallow-copy by default:](#shallow-copy-by-default)
    - [Have polymorphic procs depending on a backend parameter](#have-polymorphic-procs-depending-on-a-backend-parameter)
  - [Readings](#readings)
    - [Performance](#performance)

<!-- /TOC -->

## Storage convention

Either C or Fortran contiguous arrays are needed for BLAS optimization for Tensor of Rank 1 or 2
* C_contiguous: Row Major - Default for CPU. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
* F_contiguous: Col Major - Default for CUDA. First index is the fastest changing (rows in 2D, depth in 3D) - Rows (fastest), Columns, Depth (slowest)
* Universal: Any strides

Historically Fortran and all optimized BLAS libraries used column-major layout by default.
Today Fortran, Matlab, R, Julia, OpenGL and CUDA uses column-major layout.
On the other hand, C, Python and several deep learning libraries (Numpy, Torch, Caffe) uses row-major layout.

On CPU, Arraymancer follows the C/Python crowd. A practical bonus is that Matrix-Vector multiplication should be faster (we traverse each column in a row then change row --> row change the slowest).
On CUDA, Arraymancer follows (temporarily) the column-major layout, as many CUBLAS in-place operations expect that layout. Rewrite rules will be used for "cuda()" proc so that, if possible, CudaTensors are initialized directly on the device with column-major layout and don't need conversion.
Arraymancer will use row-major on CUDA when CUBLAS operations are replaced with custom kernels.

## Pending issues
* Some slicing syntax does not work inside generic procs: https://github.com/mratsim/Arraymancer/issues/62

## Data structure considerations

* Shape and strides are stored in a 72 bytes stack data structure (64B 8-element array + 8B int64). Cache-line wise (64B on consumer CPUs), this is not the best but stack allocated array are much better than heap-allocated and GC-managed seq (~40% perf diff by switching from seq)

* For now, shallowCopies will be used only in strategic places, for example when we want to mutate the original reference but use another striding scheme in `slicerMut`. Slicing will not return views.
Contrary to Python, the compiler can do the following optimization:
  - Copy elision
  - Move on assignment
  - Detect if the original Tensor is not used anymore and the copy is unneeded.
If further no-copy optimizations are needed, move optimization with {call} can be used so the compiler automatically choose a no-copy version if only one reference exists: https://nim-lang.org/docs/manual.html#ast-based-overloading-move-optimization
In the future Nim will support a more general `=move` operator and destructors. It will carefully be evaluated see Araq's blog post: https://nim-lang.org/araq/destructors.html

## Memory/Perf considerations
* Current CPUs cache line is 64 bytes. We tried a Tensor data structure at 32 bytes with shape and strides being a seq instead of 8-elem array + actual len: Heap allocation was far too slow. We will probably get further improvement if shape and strides fit each in 64 bytes.
Using uint is dangerous because if we do tensor[0 - 1] it will rollover to tensor[2^32]. Using int32 is possible but in the future we could expect huge sparse tensors that needs int64 indexing and int (int64) is the default in Nim. Also this might be a way to grab users from this limitation of Numpy: https://github.com/ContinuumIO/anaconda-issues/issues/3823, https://github.com/numpy/numpy/issues/5906

* Most copy operations (from nested arrays/seq, for slice assignments from nested arrays/seq or Tensor) uses iterators and avoid intermediate representation.

* Map: Tensor[T] -> Tensor[string] or Tensor[ref AnyObject] uses a non multithreaded slow path because heap allocation does not work with OpenMP.

* The tensor module is already very optimized regarding memory allocations.
Manual memory management is probably overkill, Nim GC is already extremely fast.

* Autograd module must be optimized as well as between each batch the batch temporaries are free.
One way to do that is via a memory pool or a custom allocator (buddy allocator, slab allocator, slub allocator ...). 
One of the challenge is that since Arraymancer is a dynamic framework, some batch may be bigger or smaller than the other so we can't just reuse the same memory location.

## CUDA considerations

* Reclaiming memory: currently all CudaTensors are created via new + finalizer. The finalizer proc is automatically used after (at a non-deterministic time) the object goes out of scope. In case there are memory leaks, it might be because a CudaTensor wasn't created by new, and so need a `=destroy` destructor proc. Discussions on IRC highlight that finalizer is enough for yglukhov's game engine.

* Allocations on Cuda are much more expensive than on CPU and a custom allocator will be needed. (Memory management is already manual anyway)

* Default to column-major layout (Fortran order).
Internally CUDA/CuBLAS works with column major layout. Creating CudaTensor column-major by default may avoid temporary transpose allocation.

* Currently CudaTensor are shallow-copied by default.
From a consistency point of view it would be best if both Tensor and CudaTensor have the same behaviour.
This is pending Nim improvement on assignment operator overloading and destructors + move optimization implementation

* Async operations
Operations on CUDA device ("Host -> GPU" copy, additions, substraction, etc) are non-blocking for the host.
Meaning the program can proceed with CPU computation.
"GPU -> Host" copy operation is blocking to avoid data races.

In the future, independant operations like A+B and C+D might be scheduled in different Cuda Streams for simultaneous processing.


## Coding-style
* Prefer `when` to `if` for compile-time evaluation
* Let the compiler do its job:
    - proc everywhere, `inline` tag to nudge him towards expected optimization (it might choose not to inline after a cost analysis anyway)
    - template if proc does not work or to access an object field
    - macro as last resort to manipulate AST tree or rewrite code
* Readibility, maintainability and performance are very important (in no particular order)
* Use functional constructs like `map`, `scanr` instead of `for loop` when you don't need side-effects or an iterator

## Future features

### Software features

* Implement a Tensor comprehension macro. It may be able to leverage mitems instead of result[i,j] = alpha * (i - j) * (i + j).

* Implement einsum: https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

* Automatically pull nnpack for optimized convolution on CPU (Linux/macOS only)

* Provide config files for Cuda, MKL, OpenMP, etc

* BLAS operation fusion: `transpose(A) * B` or `Ax + y` should be fused in one operation.

* Implement a Scalar[T] concept so that regular float are considered as Tensors

* (Needs thinking) Support sparse matrices. There is Magma and CuSparse for GPU. What to use for CPU? Interface should be similar to BLAS and should compile on ARM/embedded devices like Jetson TX1.

* Implement Bayesian neural networks

* Implement Graph neural networks

### Backend/hardware features

* OpenCL, probably via CLBlast

* AMD Rocm. (I don't have any AMD GPU though)

* Javascript backend. Using the Nim compiler directly is difficult, see PR https://github.com/mratsim/Arraymancer/pull/126, we can start with emscripten though.

* Metal Performance Shader backend for iPhone compat (Can we emulate/test this on macOS?)



## Ideas rejected

### Having an unified Tensor type instead of Tensor, CudaTensor, etc.

Rejected because of maintenance/difficult to debug errors. For example for this data structure:

```Nim
type
  Backend* = enum
    Cpu,
    Cuda

  Tensor*[B: static[Backend]; T] = object
    shape: seq[int]
    strides: seq[int]
    offset: int
    when B == Backend.Cpu:
      data: seq[T]
    else:
      data_ptr: ptr T

template shape*(t: Tensor): seq[int] =
  t.shape
```

The template will not compile due to "Cannot generate B", because due to the conditional when, Nim wants B in all proc declaration. The error points to the type declaration and not the proc declaration which makes it a pain to debug.

Furthermore the comparison operator "==" fails with "Cannot generate B" and I found no solution to that.

Also having more independant types will probably be easier for future features (distributed compute, MPI ?).

### Have the rank of the Tensor be part of its type.
Rejected because impractical for function chaining.
    Advantage: Dispatch and compatibility checking at compile time (Matrix * Matrix and Matrix * Vec)
### Have the kind of stride (C_contiguous, F_contiguous) be part of its type.
Rejected because impractical for function chaining. Furthermore this should not be exposed to the users as it's an implementation detail.

### Implement offsets and iterator using pointers.
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

### Shallow-copy by default:
Rejected until benchmarked otherwise.
If further no-copy optimizations are needed, move optimization with {call} can be used so the compiler automatically choose a no-copy version if only one reference exists: https://nim-lang.org/docs/manual.html#ast-based-overloading-move-optimization

For CudaTensor, value semantics will be implemented.

`data` is currently stored in a "seq" that always deep copy on var assignement. It doesn't copy on let assignement.

If slicing shallow-copies by default like Numpy there is a risk of modifying the original array by mistake. Since Nim is compiled we can hope that the compiler detects cases where original tensor is not reused and moves instead of copying. Excellent read on value semantics vs reference semantics: https://akrzemi1.wordpress.com/2012/02/03/value-semantics/, https://juanchopanzacpp.wordpress.com/2014/05/11/want-speed-dont-always-pass-by-value/ and https://definedbehavior.blogspot.fr/2011/08/value-semantics-copy-elision.html. Nim in-depth discussion: https://forum.nim-lang.org/t/2665/1.

References:
 - [Copy semantics](https://forum.nim-lang.org/t/1793/5) "Parameter passing doesn't copy, var x = foo() doesn't copy but moves let x = y doesn't copy but moves, var x = y does copy but I can use shallowCopy instead of = for that."
 - [Another](https://forum.nim-lang.org/t/1543) "First, it's important to understand that most of the time, you won't need shallowCopy at all. Copying is shallow by default if (1) the left-hand side of an assignment is a let variable or (2) the right-hand side is a function call."

Also using seq is far easier than implementing my own shallowCopy / refcounting code which would introduce the following questions:
- How to make sure we can modify in-place if shallow copy is allowed or a ref seq/object is used?
- To avoid reference counting, would it be better to always copy-on-write, in that case wouldn't it be better to pay the cost upfront on assignment?
- How hard will it be to maintain Arraymancer and avoid bugs because a copy-on-write was missed.

    From Scipy: https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#reference-counting

    "If you mis-handle reference counts you can get problems from memory-leaks to segmentation faults.  The only strategy I know of to handle reference counts correctly is blood, sweat, and tears."
Nim GC perf: https://gist.github.com/dom96/77b32e36b62377b2e7cadf09575b8883

In-depth [read](http://blog.stablekernel.com/when-to-use-value-types-and-reference-types-in-swift) (for Swift but applicable): performance, safety, usability


### Have polymorphic procs depending on a backend parameter
With the [following commit](https://github.com/mratsim/Arraymancer/blob/260386da01c9185f551f8afbe41d2c4beeeee92c/src/arraymancer/init_common.nim) in Cuda branch, all init procs accepted a backend parameter (Cpu, Cuda, ...). In case the backend had dedicated function like "zeros", this would avoid having to create tensor on the Cpu and then copy it to the backend.
The downside is
- Complicating the procs by the use of untyped templates, auto return types, "when t is Tensor" or "when backend is Cpu". This might promote spaghetti code.
- All new backends would require modification to the base procs, with more risks of introducing new bugs.
- In the case of "init" function, it requires the `check_nested_elements` proc in a file, then __Cpu__ and __Cuda__ specific code in another, then a __common__ file with the polymorphic procs. This would make it difficult to understand and contribute to the code.
- Only a few init functions can be used directly on GPU, **ones** and **randomTensor** will require creation on Cpu backend anyway
Two alternatives are possible to avoid that:
- Only provide the base proc for Cpu and have a rewrite rule to transform zeros(...).toCuda() into the direct Cuda function if it exists. (aka Composition)
- Use qualified import, like `ìmport arraymancer as arc` and `ìmport arraymancer/cuda as cu` and then `arc.zeros` or `cu.zeros`

## Readings

### Performance
- Compute-bound, memory-bound and IO-bound optimization: http://leto.net/docs/C-optimization.php
- Implementing matmul from scratch: http://apfel.mathematik.uni-ulm.de/~lehn/ulmBLAS/
- Implementing matmul in Nvidia assembler from scratch: https://github.com/NervanaSystems/maxas/wiki/SGEMM
- In-depth discussion on fast convolution (NCHW vs CHNW representation, Winograd kernel): https://github.com/soumith/convnet-benchmarks/issues/93
- Roofline performance model, arithmetic intensity - CPU-bound vs memory-bound: https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/

