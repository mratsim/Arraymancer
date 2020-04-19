import macros, strformat, strutils, sequtils, sets, tables, algorithm

from os import parentDir, getCurrentCompilerExe, DirSep, extractFilename, `/`, setCurrentDir

when defined(nimdoc):
  from os import getCurrentDir, paramCount, paramStr

#[
This file is a slightly modified version of the same file of `nimterop`:
https://github.com/nimterop/nimterop/blob/master/nimterop/docs.nim
]#


proc getNimRootDir(): string =
  #[
  hack, but works
  alternatively (but more complex), use (from a nim file, not nims otherwise
  you get Error: ambiguous call; both system.fileExists):
  import "$nim/testament/lib/stdtest/specialpaths.nim"
  nimRootDir
  ]#
  fmt"{currentSourcePath}".parentDir.parentDir.parentDir

const
  DirSep = when defined(windows): '\\' else: '/'

proc execAction(cmd: string): string =
  var
    ccmd = ""
    ret = 0
  when defined(Windows):
    ccmd = "cmd /c " & cmd
  elif defined(posix):
    ccmd = cmd
  else:
    doAssert false

  (result, ret) = gorgeEx(ccmd)
  doAssert ret == 0, "Command failed: " & $ret & "\ncmd: " & ccmd & "\nresult:\n" & result

template genRemove(name: untyped): untyped =
  proc `name`(s, toRemove: string): string =
    result = s
    result.`name`(toRemove)
genRemove(removePrefix)
genRemove(removeSuffix)

proc getFiles*(path: string): seq[string] =
  # Add files and dirs here, which should be skipped.
  #const excludeDirs = []
  #let ExcludeDirSet = toSet(excludeDirs)
  #if path.extractFilename in ExcludeDirSet: return
  # The files below are not valid by themselves, they are only included
  # from other files
  const excludeFiles = [ "blas_l3_gemm_aux.nim",
                         "blas_l3_gemm_data_structure.nim",
                         "blas_l3_gemm_macro_kernel.nim",
                         "blas_l3_gemm_micro_kernel.nim",
                         "blas_l3_gemm_packing.nim",
                         "p_checks_cuda.nim",
                         "p_checks_opencl.nim",
                         "blis_api.nim" ]
  let ExcludeFileSet = toSet(excludeFiles)

  for file in listFiles(path):
    if file.endsWith(".nim") and file.extractFilename notin ExcludeFileSet:
      result.add file
  for dir in listDirs(path):
    result.add getFiles(dir)

import nimDocTemplates

proc buildDocs*(path: string, docPath: string, baseDir = getProjectPath() & $DirSep,
                defaultFlags = "",
                masterBranch = "master",
                defines: openArray[string] = @[]) =
  ## Generate docs for all nim files in `path` and output all HTML files to the
  ## `docPath` in a flattened form (subdirectories are removed).
  ##
  ## If duplicate filenames are detected, they will be printed at the end.
  ##
  ## `baseDir` is the project path by default and `files` and `path` are relative
  ## to that directory. Set to "" if using absolute paths.
  ##
  ## `masterBranch` is the name of the default branch to which the docs should link
  ## when clicking the `Source` button below a procedure etc.
  ##
  ## `defines` is a list of `-d:xxx` define flags (the `xxx` part) that should be passed
  ## to `nim doc` so that `getHeader()` is invoked correctly.
  ##
  ## Use the `--publish` flag with nimble to publish docs contained in
  ## `path` to Github in the `gh-pages` branch. This requires the ghp-import
  ## package for Python: `pip install ghp-import`
  ##
  ## WARNING: `--publish` will destroy any existing content in this branch.
  ##
  ## NOTE: `buildDocs()` only works correctly on Windows with Nim 1.0+ since
  ## https://github.com/nim-lang/Nim/pull/11814 is required.
  ##
  ##
  when defined(windows) and (NimMajor, NimMinor, NimPatch) < (1, 0, 0):
    echo "buildDocs() unsupported on Windows for Nim < 1.0 - requires PR #11814"
  else:
    let
      baseDir =
        if baseDir == $DirSep:
          getCurrentDir() & $DirSep
        else:
          baseDir
      docPath = baseDir & docPath
      path = baseDir & path
      defStr = block:
        var defStr = " " & defaultFlags
        for def in defines:
          defStr &= " -d:" & def
        defStr
      nim = getCurrentCompilerExe()

    # now we walk the whole `path` and build the documentation for each `.nim` file.
    # While doing that we flatten the directory structure for the generated HTML files.
    # `src/foo/bar/baz.nim` just becomes
    # `docPath/baz.html`.
    # This allows for all files to be in the `docPath` directory, which means each
    # file will be able to find the `dochack.js` file, which will be put into
    # the `docPath` directory, too (the inclusion of the `dochack.js` is done statically
    # via our generated nimdoc.cfg file and is fixed for each generated HTML).
    let files = getFiles(path)
    var idx = 0
    var fileSet = initHashSet[string]()
    var duplSet = initHashSet[string]()
    for file in files:
      let baseName = file.extractFilename()
      let relPath = file.removePrefix(path).removeSuffix(baseName)
      let prefix = relPath.strip(chars = {'/'}) # remove possible trailing `/`
        .split('/') # split path parts
        .join(".") # concat by `.` instead
      var outfile = baseName.replace(".nim", ".html")
      if outfile in fileSet:
        duplSet.incl outfile
      else:
        fileSet.incl outfile
      outfile = docPath / outfile
      echo "Processing: ", outfile, " [", idx, "/", files.len, "]"
      # NOTE: Changing the current working directory to the project path is required in order for
      # `git.commit:` to work! Otherwise we sit in `docs` and for some reason the relative path
      # will eat one piece of the resulting `source` links and thereby removing the actual branch
      # and we end up with a broken link!
      echo execAction(&"cd {getProjectPath()} && {nim} doc {defStr} --git.commit:{masterBranch} -o:{outfile} --index:on {file}")
      inc idx
    ## now build  the index
    echo execAction(&"{nim} buildIndex -o:{docPath}/theindex.html {docPath}")
    when declared(getNimRootDir):
      #[
      NOTE: running it locally doesn't work anymore on modern chromium browser,
      because they block "access from origin 'null' due to CORS policy".
      this enables doc search, works at least locally with:
      cd {docPath} && python -m SimpleHTTPServer 9009
      ]#
      echo execAction(&"{nim} js -o:{docPath}/dochack.js {getNimRootDir()}/tools/dochack/dochack.nim")

    for i in 0 .. paramCount():
      if paramStr(i) == "--publish":
        echo execAction(&"cd {docPath} && ghp-import --no-jekyll -fp {docPath}")
        break

    # echo "Processed files: ", fileSet
    if duplSet.card > 0:
      echo "WARNING: Duplicate filenames detected: ", duplSet


let nameMap = {
  "dsl_core" : "Neural network: Declaration",
  "relu" : "Activation: Relu (Rectified linear Unit)",
  "sigmoid" : "Activation: Sigmoid",
  "tanh" : "Activation: Tanh",
  "conv2D" : "Layers: Convolution 2D",
  "embedding" : "Layers: Embedding",
  "gru" : "Layers: GRU (Gated Linear Unit)",
  "linear" : "Layers: Linear/Dense",
  "maxpool2D" : "Layers: Maxpool 2D",
  "cross_entropy_losses" : "Loss: Cross-Entropy losses",
  "mean_square_error_loss" : "Loss: Mean Square Error",
  "softmax" : "Softmax",
  "optimizers" : "Optimizers",
  "init" : "Layers: Initializations",

  "reshape_flatten" : "Reshape & Flatten",

  "decomposition" : "Eigenvalue decomposition",
  "decomposition_rand" : "Randomized Truncated SVD",
  "least_squares" : "Least squares solver",
  "linear_systems" : "Linear systems solver",
  "special_matrices" : "Special linear algebra matrices",
  "stats" : "Statistics",
  "pca" : "Principal Component Analysis (PCA)",
  "accuracy_score" : "Accuracy score",
  "common_error_functions" : "Common errors, MAE and MSE (L1, L2 loss)",
  "kmeans" : "K-Means",

  "mnist" : "MNIST",
  "imdb" : "IMDB",
  "io_csv" : "CSV reading and writing",
  "io_hdf5" : "HDF5 files reading and writing",
  "io_image" : "Images reading and writing",
  "io_npy" : "Numpy files reading and writing",

  "autograd_common" : "Data structure",
  "gates_basic" : "Basic operations",
  "gates_blas" : "Linear algebra operations",
  "gates_hadamard" : "Hadamard product (elementwise matrix multiply)",
  "gates_reduce" : "Reduction operations",
  "gates_shapeshifting_concat_split" : "Concatenation, stacking, splitting, chunking operations",
  "gates_shapeshifting_views" : "Linear algebra operations",

  "nnp_activation" : "Activations",
  "nnp_convolution" : "Convolution 2D",
  "nnp_conv2d_cudnn" : "Convolution 2D - CuDNN",
  "nnp_embedding" : "Embeddings",
  "nnp_gru" : "Gated Recurrent Unit (GRU)",
  "nnp_linear" : "Linear / Dense layer",
  "nnp_maxpooling" : "Maxpooling",
  "nnp_numerical_gradient" : "Numerical gradient",
  "nnp_sigmoid_cross_entropy" : "Sigmoid Cross-Entropy loss",
  "nnp_softmax_cross_entropy" : "Softmax Cross-Entropy loss",
  "nnp_softmax" : "Softmax"
}.toTable

proc wrap(name: string): string =
  const tmpl = """<li><a href="$#">$#</a></li>"""
  if name in nameMap:
    result = tmpl % [name & ".html", nameMap[name]]
  else:
    result = tmpl % [name & ".html", name]

proc getHeaderMap(path: string): seq[seq[string]] =
  ## returns a nesteed seq where each element is a `seq[string]` containing
  ## all elements to be added to the header at the index. The index
  ## corresponds to the `$N` of the `nimDocTemplates.headerTmpl` field.
  const excludeFiles = [ "nn", # only imports and exports `NN` files
                         "nn_dsl", # only imports and exports `NN DSL` files
                         "ml", # only imports and exports `ML` files
                         "io", # only imports and exports `io` files
                         "autograd", # only imports and exports `autograd` files
                         "blis" # doesn't import or export anything
  ]
  let ExcludeFileSet = toSet(excludeFiles)
  # map of the different header categories
  let catMap = { "tensor" : 1,
                 "nn" : 2,
                 "nn_dsl" : 2,
                 "linear_algebra" : 3,
                 "stats" : 3,
                 "ml" : 3,
                 "datasets" : 4,
                 "io" : 4,
                 "autograd" : 5 ,
                 "nn_primitives" : 6,
                 "nlp" : 7,
                 "math_ops_fusion" : 7,
                 "laser" : 7,
                 "private" : 7}.toTable

  # `indexOverride` is used to override the index of the header the file
  # is added to. Some files may be part of e.g. `tensor` but shouldn't be
  # listed there, since they aren't that important.
  # NOTE: the elements here are ``filenames`` and ``not`` directories!
  let indexOverride = { "global_config" : 7 }.toTable
  let files = getFiles(path)

  result = newSeq[seq[string]](7)
  for file in files:
    let baseName = file.extractFilename()
    let outfile = baseName.replace(".nim", "")
    if outfile in ExcludeFileSet: continue
    let subDir = file.removePrefix(path).split('/')[0]
    if subDir in catMap:
      var idx: int
      if outfile notin indexOverride:
        idx = catMap[subDir] - 1
      else:
        idx = indexOverride[outfile] - 1
      result[idx].add outfile

proc genNimdocCfg*(path: string) =
  ## This proc generates the `nimdoc.cfg`, which sits at the root of the
  ## arraymancer repository. We generate it so that we can combine the
  ## front page template derived from flyx's NimYaml: https://github.com/flyx/NimYAML
  ## with the standard Nim document generation. We generate the fields for
  ## the header links from the actual files found in each diretory.
  ##
  ## NOTE: manual intervention is required for each directory that is added
  ## and should show up as its own tab in the header. Essentially look at the
  ## `$<number>` spans in the `docFileTmpl` above to see what to do.
  let headerMap = getHeaderMap(path)
  # create the strings based on the header map for each span
  var spans = newSeq[string](7)
  for idx in 0 ..< spans.len:
    spans[idx] = headerMap[idx].sorted.mapIt(wrap(it)).join("\n")
  # fill the HTML generation template from the filenames
  let htmlTmpl = headerTmpl % [ spans[0], spans[1], spans[2],
                                spans[3], spans[4], spans[5],
                                spans[6]]
  # first "header"
  var fdata = ""
  fdata.add("# Arraymancer documentation generation\n\n")
  fdata.add(&"git.url = \"{gitUrl}\"\n\n")
  fdata.add(&"doc.item.seesrc = \"\"\"{docItemSeeSrc}\"\"\"\n\n")
  # finally write the HTML document template
  fdata.add(&"doc.file = \"\"\"{docFileTmpl}{htmlTmpl}\"\"\"\n")

  # now build the content for the spans
  writeFile(getProjectPath() & $DirSep & "nimdoc.cfg", fdata)
