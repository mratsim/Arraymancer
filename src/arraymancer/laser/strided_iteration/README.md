# Strided parallel iteration for tensors

Implementation is generic and work on any tensor types as long
as it implement the following interface:

Input data storage backend must be shallow copied on assignment (reference semantics)
the macro works on aliases to ensure that if the tensor is the result of another routine that routine is only called once, for example x[0..<2, _] will not slice `x` multiple times.

The input type name is not used at all so can be anything.

Routine and fields used, (routines mean proc, template, macros):
  - rank, size:
      routines or fields that return an int
  - shape, strides:
      routines or fields that returns an array, seq or indexable container
      that supports `[]`. Read-only access.
  - unsafe_raw_offset:
      routine or field that returns a ptr UncheckedArray[T]
      or a distinct type with `[]` indexing implemented.
      The address should be the start of the raw data including
      the eventual tensor offset for subslices, i.e. equivalent to
      the address of x[0, 0, 0, ...]
      Needs mutable access for var tensor.

Additionally the `forEach` macro needs an `is_C_contiguous` routine
to allow dispatching to `forEachContiguous` or `forEachStrided`

The code is carefully tuned to produce the most performant and compact iteration scheme. The `forEach` macro however still has to duplicate the code body to dispatch for contiguous and non-contiguous case.

The iteration supports OpenMP and the following parameters (see [openmp.nim](../openmp.nim) file):
  - omp_grain_size: amount of work items per threads (including hyperthreading), default 1024
                    below this threshold, execution is serial
  - use_simd: Tell the compiler to unroll the loop so that SIMD can be used
              i.e. when iterating on float32, AVX2 can work on 256-bit so loops
              will be unrolled by a factor 8

Note that for strided iteration the omp_grain_size is divided by a factor
OMP_NON_CONTIGUOUS_SCALE_FACTOR, which is 4 by default and can be configured
at compile-time.
