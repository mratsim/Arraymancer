import macros, strformat, strutils, sequtils, sets

from os import parentDir, getCurrentCompilerExe, DirSep, extractFilename, `/`

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

proc buildDocs*(path: string, docPath: string, baseDir = getProjectPath() & $DirSep,
                defines: openArray[string] = @[]) =
  ## Generate docs for all nim files in `path` and output all HTML files to the
  ## `docPath` in a flattened form (subdirectories are removed).
  ##
  ## If duplicate filenames are detected, they will be printed at the end.
  ##
  ## `baseDir` is the project path by default and `files` and `path` are relative
  ## to that directory. Set to "" if using absolute paths.
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
        var defStr = ""
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
      # now call `nim doc` on each file
      echo execAction(&"{nim} doc {defStr} -o:{outfile} --index:on {file}")
      inc idx
    ## now build  the index
    echo execAction(&"{nim} buildIndex -o:{docPath}/theindex.html {docPath}")
    when declared(getNimRootDir):
      #[
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
