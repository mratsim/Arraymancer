### Package
version       = "0.2.90"
author        = "Mamy André-Ratsimbazafy"
description   = "A n-dimensional tensor (ndarray) library"
license       = "Apache License 2.0"

### Dependencies
requires "nim >= 0.17.2", "nimblas >= 0.1.3", "nimcuda >= 0.1.4"

## Install files
srcDir = "src"

########################################################
# External libs configuration

### BLAS support
## OSX
# switch("define","openblas")
# switch("clibdir", "/usr/local/opt/openblas/lib")
# switch("cincludes", "/usr/local/opt/openblas/include")

### BLIS support
# switch("define","blis")

### MKL support
# Check the mkl switches in the test file for single-threaded and openp version

### Cuda configuration
## Pass -d:cuda to build arraymancer with cuda support
## Use the cuda switches below
## Replace /opt/cuda by your own path
## TODO: auto detection or at least check in common directories
## Note: It is import to gate compiler flags like -march=native  behind Xcompiler "-Xcompiler -march=native"

template cudaSwitches() =
  switch("cincludes", "/opt/cuda/include")
  switch("cc", "gcc") # We trick Nim about nvcc being gcc, pending https://github.com/nim-lang/Nim/issues/6372
  switch("gcc.exe", "/opt/cuda/bin/nvcc")
  switch("gcc.linkerexe", "/opt/cuda/bin/nvcc")
  switch("gcc.cpp.exe", "/opt/cuda/bin/nvcc")
  switch("gcc.cpp.linkerexe", "/opt/cuda/bin/nvcc")
  # Due to the __ldg intrinsics in kernels
  # we only support compute capabilities 3.5+
  # See here: http://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html
  # And wikipedia for GPU capabilities: https://en.wikipedia.org/wiki/CUDA
  switch("gcc.options.always", "-arch=sm_61 --x cu") # Interpret .c files as .cu
  switch("gcc.cpp.options.always", "-arch=sm_61 --x cu -Xcompiler -fpermissive") # Interpret .c files as .cu, gate fpermissive behind Xcompiler

when defined(cuda):
  cudaSwitches

########################################################
# Optimization

# Multithreading
# use the -d:openmp switch
# which passC: -fopenmp to the compiler

# Native processor optimization
# use the -d:native
# which passC: -march=native to the compiler


##########################################################################
## Testing tasks

proc test(name: string, lang: string = "c") =
  if not dirExists "bin":
    mkDir "bin"
  if not dirExists "nimcache":
    mkDir "nimcache"
  --run
  --nimcache: "nimcache"
  switch("out", ("./bin/" & name))
  setCommand lang, "tests/" & name & ".nim"

task test, "Run all tests - Default BLAS":
  test "all_tests"

task test_cuda, "Run all tests - Cuda backend with CUBLAS":
  switch("define","cuda")
  cudaSwitches # Unfortunately the "switch" line doesn't also trigger
               # the "when defined(cuda)" part of this nimble file
               # hence the need to call cudaSwitches explicitly
  test "all_tests_cuda", "cpp"

task test_deprecated, "Run all tests on deprecated static[Backend] procs":
  test "all_tests_deprecated"

task test_openblas, "Run all tests - OpenBLAS":
  ## Should work but somehow Nim doesn't find libopenblas.dylib on MacOS
  when defined(macosx):
    switch("define","blas=openblas")
    switch("clibdir", "/usr/local/opt/openblas/lib")
    switch("cincludes", "/usr/local/opt/openblas/include")
  test "all_tests"

task test_blis, "Run all tests - BLIS":
  switch("define","blis")
  test "all_tests"

task test_native, "Run all tests - march=native":
  switch("define","native")
  test "all_tests"

task test_openmp, "Run all tests - OpenMP":
  switch("define","openmp")
  test "all_tests"

task test_mkl, "Run all tests - Intel MKL - single threaded":
  switch("define","blas=mkl_intel_lp64")
  switch("clibdir", "/opt/intel/mkl/lib/intel64")
  switch("passl", "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a")
  switch("passl", "-lmkl_core")
  switch("passl", "-lmkl_sequential")
  switch("dynlibOverride","mkl_intel_lp64")
  test "all_tests"

task test_mkl_omp, "Run all tests - Intel MKL + OpenMP":
  switch("define","openmp")
  switch("define","blas=mkl_intel_lp64")
  switch("clibdir", "/opt/intel/mkl/lib/intel64")
  switch("passl", "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a")
  switch("passl", "-lmkl_core")
  switch("passl", "-lmkl_gnu_thread")
  switch("passl", "-lgomp")
  switch("dynlibOverride","mkl_intel_lp64")
  test "all_tests"

task test_release, "Run all tests - Release mode":
  switch("define","release")
  test "all_tests"

task gen_doc, "Generate Arraymancer documentation":
  switch("define", "doc")
  exec "nim doc2 src/arraymancer"
