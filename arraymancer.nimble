### Package
version       = "0.6.0"
author        = "Mamy André-Ratsimbazafy"
description   = "A n-dimensional tensor (ndarray) library"
license       = "Apache License 2.0"

### Dependencies
requires "nim >= 1.0.0",
  "nimblas >= 0.2.2",
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

## Archlinux
# Contrary to Debian-based distro, blas.dll doesn't supply the cblas interface
# so "-d:blas=cblas" must be passed

### BLIS support
# switch("define","blis")

### MKL support
# Check the mkl switches in the test file for single-threaded and openp version

template mkl_threadedSwitches(switches: var string) =
  switches.add " --stackTrace:off"
  switches.add " --d:blas=mkl_intel_lp64"
  switches.add " --d:lapack=mkl_intel_lp64"
  switches.add " --clibdir:/opt/intel/mkl/lib/intel64"
  switches.add " --passl:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  switches.add " --passl:-lmkl_core"
  switches.add " --passl:-lmkl_gnu_thread"
  switches.add " --passl:-lgomp"
  switches.add " --dynlibOverride:mkl_intel_lp64"

template mkl_singleSwitches(switches: var string) =
  switches.add " --d:blas=mkl_intel_lp64"
  switches.add " --d:lapack=mkl_intel_lp64"
  switches.add " --clibdir:/opt/intel/mkl/lib/intel64"
  switches.add " --passl:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  switches.add " --passl:-lmkl_core"
  switches.add " --passl:-lmkl_sequential"
  switches.add " --dynlibOverride:mkl_intel_lp64"

# ### Cuda configuration
## Pass -d:cuda to build arraymancer with cuda support
## Use the cuda switches below
## Replace /opt/cuda by your own path
## TODO: auto detection or at least check in common directories
## Note: It is import to gate compiler flags like -march=native  behind Xcompiler "-Xcompiler -march=native"

# NVCC config
template cudaSwitches(switches: var string) =
  switches.add " --cincludes:/opt/cuda/include"
  switches.add " --cc:gcc" # We trick Nim about nvcc being gcc, pending https://github.com/nim-lang/Nim/issues/6372
  switches.add " --gcc.exe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.linkerexe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.cpp.exe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.cpp.linkerexe:/opt/cuda/bin/nvcc"
  # Due to the __ldg intrinsics in kernels
  # we only support compute capabilities 3.5+
  # See here: http://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html
  # And wikipedia for GPU capabilities: https://en.wikipedia.org/wiki/CUDA

  # Note: the switches below might conflict with nim.cfg
  # switches.add " --gcc.options.always:\"-arch=sm_61 --x cu\"" # Interpret .c files as .cu
  # switches.add " --gcc.cpp.options.always:\"-arch=sm_61 --x cu -Xcompiler -fpermissive\"" # Interpret .c files as .cu, gate fpermissive behind Xcompiler
  switches.add " -d:cudnn"

template cuda_mkl_openmp(switches: var string) =
  switches.mkl_threadedSwitches()
  switches.add " --cincludes:/opt/cuda/include"
  switches.add " --cc:gcc" # We trick Nim about nvcc being gcc, pending https://github.com/nim-lang/Nim/issues/6372
  switches.add " --gcc.exe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.linkerexe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.cpp.exe:/opt/cuda/bin/nvcc"
  switches.add " --gcc.cpp.linkerexe:/opt/cuda/bin/nvcc"

  # Note: the switches below might conflict with nim.cfg
  # switches.add " --gcc.options.always:\"-arch=sm_61 --x cu -Xcompiler -fopenmp -Xcompiler -march=native\""
  # switches.add " --gcc.cpp.options.always:\"-arch=sm_61 --x cu -Xcompiler -fopenmp -Xcompiler -march=native\""

# Clang config - make sure Clang supports your CUDA SDK version
# https://gist.github.com/ax3l/9489132
# https://llvm.org/docs/CompileCudaWithLLVM.html
# | clang++ | supported CUDA release | supported SMs |
# | ------- | ---------------------- | ------------- |
# | 3.9-5.0 | 7.0-8.0                | 2.0-(5.0)6.0  |
# | 6.0     | [7.0-9.0](https://github.com/llvm-mirror/clang/blob/release_60/include/clang/Basic/Cuda.h) | [(2.0)3.0-7.0](https://github.com/llvm-mirror/clang/blob/release_60/lib/Basic/Targets/NVPTX.cpp#L163-L188) |
# | 7.0     | [7.0-9.2](https://github.com/llvm-mirror/clang/blob/release_70/include/clang/Basic/Cuda.h) | [(2.0)3.0-7.2](https://github.com/llvm-mirror/clang/blob/release_70/lib/Basic/Targets/NVPTX.cpp#L196-L223) |
# | 8.0     | [7.0-10.0](https://github.com/llvm-mirror/clang/blob/release_80/include/clang/Basic/Cuda.h) | [(2.0)3.0-7.5](https://github.com/llvm-mirror/clang/blob/release_70/lib/Basic/Targets/NVPTX.cpp#L199-L228) |
# | trunk   | [7.0-10.1](https://github.com/llvm-mirror/clang/blob/master/include/clang/Basic/Cuda.h) | [(2.0)3.0-7.5](https://github.com/llvm-mirror/clang/blob/master/lib/Basic/Targets/NVPTX.cpp#L200-L229) |
#
# template cudaSwitches(switches: var string) =
#   switches.add " --cincludes:/opt/cuda/include"
#   switches.add " --clibdir:/opt/cuda/lib"
#   switches.add " --cc:clang"
#   switches.add " --clang.cpp.options.always:\"--cuda-path=/opt/cuda -lcudart_static -x cuda --cuda-gpu-arch=sm_61 --cuda-gpu-arch=sm_75\""
#   switches.add " -d:cudnn"

# template cuda_mkl_openmp(switches: var string) =
#   switches.mkl_threadedSwitches()
#   switches.add " --cincludes:/opt/cuda/include"
#   switches.add " --clibdir:/opt/cuda/lib"
#   switches.add " --cc:clang"
#   switches.add " --clang.cpp.options.always:\"--cuda-path=/opt/cuda -lcudart_static -x cuda --cuda-gpu-arch=sm_61 --cuda-gpu-arch=sm_75 -fopenmp -march=native\""
#   switches.add " -d:cudnn"

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

proc test(name, switches = "", split = false, lang = "c") =
  if not dirExists "build":
    mkDir "build"
  if not split:
    exec "nim " & lang & " -o:build/" & name & switches & " -r tests/" & name & ".nim"
  else:
    exec "nim " & lang & " -o:build/" & name & switches & " -r tests/_split_tests/" & name & ".nim"

task all_tests, "Run all tests - Intel MKL + Cuda + OpenCL + OpenMP":
  var switches = " -d:cuda -d:opencl -d:openmp"
  switches.cuda_mkl_openmp()
  test "full_test_suite", switches, split=false, lang="cpp"

# Split tests are unnecessary after 1.0.0 (no more 3GB+ memory used when compiling)
#
# task test, "Run all tests - Default BLAS & Lapack":
#   test "tests_tensor_part01", "", split = true
#   test "tests_tensor_part02", "", split = true
#   test "tests_tensor_part03", "", split = true
#   test "tests_tensor_part04", "", split = true
#   test "tests_tensor_part05", "", split = true
#   test "tests_cpu_remainder", "", split = true
#
# task test_no_lapack, "Run all tests - Default BLAS without lapack":
#   let switch = " -d:no_lapack"
#   test "tests_tensor_part01", switch, split = true
#   test "tests_tensor_part02", switch, split = true
#   test "tests_tensor_part03", switch, split = true
#   test "tests_tensor_part04", switch, split = true
#   test "tests_tensor_part05", switch, split = true
#   test "tests_cpu_remainder", switch, split = true

task test, "Run all tests - Default BLAS & Lapack":
  test "tests_cpu", "", split = false

task test_no_lapack, "Run all tests - Default BLAS without lapack":
  let switch = " -d:no_lapack"
  test "tests_cpu", switch, split = false

task test_cpp, "Run all tests - Cpp codegen":
  test "tests_cpu", "", split = false, "cpp"

task test_cuda, "Run all tests - Cuda backend with CUBLAS and CuDNN":
  var switches = " -d:cuda -d:cudnn"
  switches.add " -d:blas=cblas" # Archlinux, comment out on Debian/Ubuntu
  switches.cudaSwitches()
  test "tests_cuda", switches, split = false, "cpp"

task test_opencl, "Run all OpenCL backend tests":
  var switches = " -d:opencl"
  switches.add " -d:blas=cblas" # Archlinux, comment out on Debian/Ubuntu
  test "tests_opencl", switches, split = false, "cpp"

# task test_deprecated, "Run all tests on deprecated procs":
#  test "tests_cpu_deprecated"

task test_openblas, "Run all tests - OpenBLAS":
  var switches = " -d:blas=openblas -d:lapack=openblas"
  when defined(macosx):
    ## Should work but somehow Nim doesn't find libopenblas.dylib on MacOS
    switches.add " --clibdir:/usr/local/opt/openblas/lib"
    switches.add " --cincludes:/usr/local/opt/openblas/include"
  test "tests_cpu", switches

task test_blis, "Run all tests - BLIS":
  test "tests_cpu", " -d:blis"

task test_native, "Run all tests - march=native":
  test "tests_cpu", " -d:native"

task test_openmp, "Run all tests - OpenMP":
  var switches = " -d:openmp"
  switches.add " --stackTrace:off" # stacktraces interfere with OpenMP
  when defined(macosx): # Default compiler on Mac is clang without OpenMP and gcc is an alias to clang.
                        # Use Homebrew GCC instead for OpenMP support. GCC (v8), must be properly linked via `brew link gcc`
    switches.add " --cc:gcc"
    switches.add " --gcc.exe:/usr/local/bin/gcc-8"
    switches.add " --gcc.linkerexe:/usr/local/bin/gcc-8"
  test "tests_cpu", switches

task test_mkl, "Run all tests - Intel MKL - single threaded":
  var switches: string
  switches.mkl_singleSwitches()
  test "tests_cpu", switches

task test_mkl_omp, "Run all tests - Intel MKL + OpenMP":
  var switches = " -d:openmp"
  switches.mkl_threadedSwitches()
  test "tests_cpu", switches

task test_release, "Run all tests - Release mode":
  test "tests_cpu", " -d:release"


import docs / [docs, generateNimdocCfg]
task gen_docs, "Generate Arraymancer documentation":
  # generate nimdoc.cfg file so we can generate the correct header for the
  # index.html page without having to mess with the HTML manually.
  genNimdocCfg("src/")
  # build the actual docs and the index
  buildDocs("src/", "docs/build")
  # Copy our stylesheets
  cpFile("docs/docutils.css", "docs/build/docutils.css")
  cpFile("docs/nav.css", "docs/build/nav.css")
  # Process the rst
  for filePath in listFiles("docs/"):
    if filePath[^4..^1] == ".rst":
      let modName = filePath[5..^5]
      exec r"nim rst2html -o:docs/build/" & modName & ".html " & filePath
