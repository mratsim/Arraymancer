Main Arraymancer contributors (sorted alphabetically)

### Andrea Ferretti (@andreaferetti)
  - Autograd of mean along an axis

### Angel Ezquerra (@AngelEzquerra)
  - Add features that were missing from Numpy and Matlab, such as:
    - Missing math functions and operators (contains, convolve, diff_discrete, median, sinc, roll, flatten, append, moveaxes, unique, etc.)
    - Add or improve diagonal, triangular and other special matrix functions (toepliz, hankel, vander, etc.)
  - Improve Complex and bool tensor support as well as mixed real-Complex ops
  - Add support for combining span slices with negative steps (e.g. `t[_|-1]`)
  - Add support for doing a masked fill from a tensor
  - Remove some compilation warnings
  - Error message and documentation improvements
  - Miscellaneous fixes

### Eduardo Bart (@edubart)
  - OpenMP
  - Several performance optimizations and fixes including
    - Strided iterators
    - Uninitialized seq
  - Shapeshifting procs
  - Developing the ecosystem with [arraymancer-vision](https://github.com/edubart/arraymancer-vision) and [arraymancer-demos](https://github.com/edubart/arraymancer-demos)

### Fabian Keller (@bluenote10)
  - CSV and toSeq exports
  - Tensor plotting tool
  - several fixes

### Mamy Ratsimbazafy (@mratsim)
  - Lead dev

### Manguluka (@manguluka)
  - tanh activation

### Xander Johnson (@metasyn)
  - Kmeans clustering
  - Automation of MNIST download, caching and reading from compressed gzip
  - IMDB dataset loader

### Vindaar (@vindaar)
  - HDF5
