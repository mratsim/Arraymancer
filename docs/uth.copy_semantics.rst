==============================
Under the hood: Copy semantics
==============================

Safe vs unsafe: copy vs view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compared to most frameworks, Arraymancer choose to be safe by default
but allows ``unsafe`` operations to optimize for speed and memory. The
tensor resulting from ``unsafe`` operations (no-copy operations) share
the underlying storage with the input tensor (also called views or
shallow copies). This is often a surprise for beginners.

In the future Arraymancer will leverage Nim compiler to automatically
detect when an original is not used and modified anymore to
automatically replace it by the ``unsafe`` equivalent.

For CudaTensors, operations are unsafe by default (including assignmnt
with ``=``) while waiting for further Nim optimizations for manually
managed memory. CudaTensors can be copied safely with ``.clone``