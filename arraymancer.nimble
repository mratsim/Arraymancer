### Package
version       = "0.5.0"
author        = "Mamy André-Ratsimbazafy"
description   = "A n-dimensional tensor (ndarray) library"
license       = "Apache License 2.0"

### Dependencies
requires "nim >= 0.19.0",
  "nimblas >= 0.2.1",
  "nimlapack >= 0.1.1",
  "nimcuda >= 0.1.4",
  "nimcl >= 0.1.3",
  "clblast",
  "stb_image",
  "zip",
  "untar"

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
  switch("define", "cudnn")

when defined(cuda):
  cudaSwitches

template mkl_threadedSwitches() =
  switch("define","openmp")
  switch("stackTrace","off")
  switch("define","blas=mkl_intel_lp64")
  switch("define","lapack=mkl_intel_lp64")
  switch("clibdir", "/opt/intel/mkl/lib/intel64")
  switch("passl", "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a")
  switch("passl", "-lmkl_core")
  switch("passl", "-lmkl_gnu_thread")
  switch("passl", "-lgomp")
  switch("dynlibOverride","mkl_intel_lp64")

template mkl_singleSwitches() =
  switch("define","blas=mkl_intel_lp64")
  switch("define","lapack=mkl_intel_lp64")
  switch("clibdir", "/opt/intel/mkl/lib/intel64")
  switch("passl", "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a")
  switch("passl", "-lmkl_core")
  switch("passl", "-lmkl_sequential")
  switch("dynlibOverride","mkl_intel_lp64")

template cuda_mkl_openmp() =
  mkl_threadedSwitches()
  switch("cincludes", "/opt/cuda/include")
  switch("cc", "gcc") # We trick Nim about nvcc being gcc, pending https://github.com/nim-lang/Nim/issues/6372
  switch("gcc.exe", "/opt/cuda/bin/nvcc")
  switch("gcc.linkerexe", "/opt/cuda/bin/nvcc")
  switch("gcc.cpp.exe", "/opt/cuda/bin/nvcc")
  switch("gcc.cpp.linkerexe", "/opt/cuda/bin/nvcc")
  switch("gcc.options.always", "-arch=sm_61 --x cu -Xcompiler -fopenmp -Xcompiler -march=native")
  switch("gcc.cpp.options.always", "-arch=sm_61 --x cu -Xcompiler -fopenmp -Xcompiler -march=native")

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
  if not dirExists "build":
    mkDir "build"
  if not dirExists "nimcache":
    mkDir "nimcache"
  --run
  switch("out", ("./build/" & name))
  setCommand lang, "tests/" & name & ".nim"

task all_tests, "Run all tests - Intel MKL + OpenMP + Cuda + march=native + release":
  switch("define","cuda")
  cuda_mkl_openmp
  test "full_test_suite", "cpp"

task test, "Run all tests - Default BLAS & Lapack":
  test "tests_cpu"

task test_no_lapack, "Run all tests - Default BLAS without lapack":
  switch("define", "no_lapack")
  test "tests_cpu"

task test_cpp, "Run all tests - Cpp codegen":
  test "tests_cpu", "cpp"

task test_cuda, "Run all tests - Cuda backend with CUBLAS and CuDNN":
  switch("define","cuda")
  switch("define","cudnn")
  cudaSwitches  # Unfortunately the "switch" line doesn't also trigger
                # the "when defined(cuda)" part of this nimble file
                # hence the need to call cudaSwitches explicitly
  test "tests_cuda", "cpp"

task test_opencl, "Run all OpenCL backend tests":
  switch("define", "opencl")
  test "tests_opencl"

# task test_deprecated, "Run all tests on deprecated procs":
#  test "tests_cpu_deprecated"

task test_openblas, "Run all tests - OpenBLAS":
  switch("define","blas=openblas")
  switch("define","lapack=openblas")
  when defined(macosx):
    ## Should work but somehow Nim doesn't find libopenblas.dylib on MacOS
    switch("clibdir", "/usr/local/opt/openblas/lib")
    switch("cincludes", "/usr/local/opt/openblas/include")
  test "tests_cpu"

task test_blis, "Run all tests - BLIS":
  switch("define","blis")
  test "tests_cpu"

task test_native, "Run all tests - march=native":
  switch("define","native")
  test "tests_cpu"

task test_openmp, "Run all tests - OpenMP":
  switch("define","openmp")
  switch("stackTrace","off") # stacktraces interfere with OpenMP
  when defined(macosx): # Default compiler on Mac is clang without OpenMP and gcc is an alias to clang.
                        # Use Homebrew GCC instead for OpenMP support. GCC (v7), must be properly linked via `brew link gcc`
    switch("cc", "gcc")
    switch("gcc.exe", "/usr/local/bin/gcc-7")
    switch("gcc.linkerexe", "/usr/local/bin/gcc-7")
  test "tests_cpu"

task test_mkl, "Run all tests - Intel MKL - single threaded":
  mkl_singleSwitches
  test "tests_cpu"

task test_mkl_omp, "Run all tests - Intel MKL + OpenMP":
  mkl_threadedSwitches
  test "tests_cpu"

task test_release, "Run all tests - Release mode":
  switch("define","release")
  test "tests_cpu"

task gen_doc, "Generate Arraymancer documentation":
  # TODO: Industrialize: something more robust that only check nim files (and not .DS_Store ...)
  for filePath in listFiles("src/tensor/"):
    let modName = filePath[11..^5] # Removing src/tensor/ (11 chars) and .nim (4 chars) # TODO: something more robust
    # Cuda doc is broken https://github.com/nim-lang/Nim/issues/6910
    # Delete doc comment from nimcuda before using this
    exec r"nim doc -o:docs/build/tensor." & modName & ".html " & filePath

  for filePath in listFiles("src/nn_primitives/"):
    let modName = filePath[18..^5] # Removing src/nn_primitives/ (18 chars) and .nim (4 chars) # TODO: something more robust
    # Cuda doc is broken https://github.com/nim-lang/Nim/issues/6910
    # Delete doc comment from nimcuda before using this
    exec r"nim doc -o:docs/build/nnp." & modName & ".html " & filePath

  for filePath in listFiles("src/autograd/"):
    let modName = filePath[13..^5] # Removing src/autograd/ (13 chars) and .nim (4 chars) # TODO: something more robust
    exec r"nim doc -o:docs/build/ag." & modName & ".html " & filePath

  for filePath in listFiles("src/nn/"):
    let modName = filePath[7..^5] # Removing src/nn_primitives/ (18 chars) and .nim (4 chars) # TODO: something more robust
    exec r"nim doc -o:docs/build/nn." & modName & ".html " & filePath

  # TODO auto check subdir
  for filePath in listFiles("src/nn/activation/"):
    let modName = filePath[18..^5]
    exec r"nim doc -o:docs/build/nn_activation." & modName & ".html " & filePath

  for filePath in listFiles("src/nn/layers/"):
    let modName = filePath[14..^5]
    exec r"nim doc -o:docs/build/nn_layers." & modName & ".html " & filePath

  for filePath in listFiles("src/nn/loss/"):
    let modName = filePath[12..^5]
    exec r"nim doc -o:docs/build/nn_loss." & modName & ".html " & filePath

  for filePath in listFiles("src/nn/optimizers/"):
    let modName = filePath[18..^5]
    exec r"nim doc -o:docs/build/nn_optimizers." & modName & ".html " & filePath

  for filePath in listFiles("src/nn_dsl/"):
    let modName = filePath[11..^5]
    exec r"nim doc -o:docs/build/nn_dsl." & modName & ".html " & filePath

  for filePath in listFiles("src/linear_algebra/"):
    let modName = filePath[19..^5]
    exec r"nim doc -o:docs/build/la." & modName & ".html " & filePath

  for filePath in listFiles("src/stats/"):
    let modName = filePath[10..^5]
    exec r"nim doc -o:docs/build/stats." & modName & ".html " & filePath

  for filePath in listFiles("src/ml/clustering/"):
    let modName = filePath[18..^5]
    exec r"nim doc -o:docs/build/ml." & modName & ".html " & filePath

  for filePath in listFiles("src/ml/dimensionality_reduction/"):
    let modName = filePath[32..^5]
    exec r"nim doc -o:docs/build/ml." & modName & ".html " & filePath

  for filePath in listFiles("src/ml/metrics/"):
    let modName = filePath[15..^5]
    exec r"nim doc -o:docs/build/ml." & modName & ".html " & filePath

  for filePath in listFiles("src/io/"):
    let modName = filePath[7..^5]
    exec r"nim doc -o:docs/build/io." & modName & ".html " & filePath

  for filePath in listFiles("src/datasets/"):
    let modName = filePath[13..^5]
    exec r"nim doc -o:docs/build/datasets." & modName & ".html " & filePath

  # Process the rst
  for filePath in listFiles("docs/"):
    if filePath[^4..^1] == ".rst":
      let modName = filePath[5..^5]
      exec r"nim rst2html -o:docs/build/" & modName & ".html " & filePath

  # Copy stylesheets
  cpFile("docs/docutils.css", "docs/build/docutils.css")
  cpFile("docs/nav.css", "docs/build/nav.css")
