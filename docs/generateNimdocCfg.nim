import strutils, docs, tables, sequtils, macros, strformat, algorithm
from os import parentDir, getCurrentCompilerExe, DirSep, extractFilename, `/`

## Arraymancer documentation generation


const gitUrl* = "https://github.com/mratsim/arraymancer"

const docItemSeeSrc* = """&nbsp;&nbsp;<a
href="${url}/tree/${commit}/${path}#L${line}"
class="link-seesrc" target="_blank">Source</a>
<a href="${url}/edit/master/${path}#L${line}" class="link-seesrc" target="_blank" >Edit</a>
"""

# TODO: industrialize similar to Nim website: https://github.com/nim-lang/Nim/blob/e758b9408e8fe935117f7f793164f1c9b74cec06/tools/nimweb.nim#L45
# And: https://github.com/nim-lang/Nim/blob/d3f966922ef4ddd05c137f82e5b2329b3d5dc485/web/website.ini#L31

# TODO: move the technical reference to the end (need some CSS so that elements are properly placed)

const docFileTmpl* = """<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<!--  This file is generated by Nim. -->
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<!-- Favicon -->
<link rel="shortcut icon" href="data:image/x-icon;base64,AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AAAAAAUAAAAF////AP///wD///8A////AP///wD///8A////AP///wD///8A////AAAAAAIAAABbAAAAlQAAAKIAAACbAAAAmwAAAKIAAACVAAAAWwAAAAL///8A////AP///wD///8A////AAAAABQAAADAAAAAYwAAAA3///8A////AP///wD///8AAAAADQAAAGMAAADAAAAAFP///wD///8A////AP///wAAAACdAAAAOv///wD///8A////AP///wD///8A////AP///wD///8AAAAAOgAAAJ3///8A////AP///wAAAAAnAAAAcP///wAAAAAoAAAASv///wD///8A////AP///wAAAABKAAAAKP///wAAAABwAAAAJ////wD///8AAAAAgQAAABwAAACIAAAAkAAAAJMAAACtAAAAFQAAABUAAACtAAAAkwAAAJAAAACIAAAAHAAAAIH///8A////AAAAAKQAAACrAAAAaP///wD///8AAAAARQAAANIAAADSAAAARf///wD///8AAAAAaAAAAKsAAACk////AAAAADMAAACcAAAAnQAAABj///8A////AP///wAAAAAYAAAAGP///wD///8A////AAAAABgAAACdAAAAnAAAADMAAAB1AAAAwwAAAP8AAADpAAAAsQAAAE4AAAAb////AP///wAAAAAbAAAATgAAALEAAADpAAAA/wAAAMMAAAB1AAAAtwAAAOkAAAD/AAAA/wAAAP8AAADvAAAA3gAAAN4AAADeAAAA3gAAAO8AAAD/AAAA/wAAAP8AAADpAAAAtwAAAGUAAAA/AAAA3wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAAD/AAAA/wAAAP8AAADfAAAAPwAAAGX///8A////AAAAAEgAAADtAAAAvwAAAL0AAADGAAAA7wAAAO8AAADGAAAAvQAAAL8AAADtAAAASP///wD///8A////AP///wD///8AAAAAO////wD///8A////AAAAAIcAAACH////AP///wD///8AAAAAO////wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A//8AAP//AAD4HwAA7/cAAN/7AAD//wAAoYUAAJ55AACf+QAAh+EAAAAAAADAAwAA4AcAAP5/AAD//wAA//8AAA=="/>
<link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA3XAAAN1wFCKJt4AAAAB3RJTUUH4QQQEwksSS9ZWwAAAk1JREFUWMPtll2ITVEUx39nn/O7Y5qR8f05wtCUUr6ZIS++8pEnkZInPImneaCQ5METNdOkeFBKUhMPRIkHKfEuUZSUlGlKPN2TrgfncpvmnntnmlEyq1Z7t89/rf9a6+y99oZxGZf/XeIq61EdtgKXgdXA0xrYAvBjOIF1AI9zvjcC74BSpndrJPkBWDScTF8Aa4E3wDlgHbASaANmVqlcCnwHvgDvgVfAJ+AikAAvgfVZwLnSVZHZaOuKoQi3ZOMi4NkYkpe1p4J7A8BpYAD49hfIy/oqG0+hLomiKP2L5L+1ubn5115S+3OAn4EnwBlgMzCjyt6ZAnQCJ4A7wOs88iRJHvw50HoujuPBoCKwHWiosy8MdfZnAdcHk8dxXFJ3VQbQlCTJvRBCGdRbD4M6uc5glpY3eAihpN5S5w12diSEcCCEcKUO4ljdr15T76ur1FDDLIQQ3qv71EdDOe3Kxj3leRXyk+pxdWnFWod6Wt2bY3de3aSuUHcPBVimHs7mK9WrmeOF6lR1o9qnzskh2ar2qm1qizpfXaPeVGdlmGN5pb09qMxz1Xb1kLqgzn1RyH7JUXW52lr5e/Kqi9qpto7V1atuUzfnARrV7jEib1T76gG2qxdGmXyiekkt1GswPTtek0aBfJp6YySGBfWg2tPQ0FAYgf1stUfdmdcjarbYJEniKIq6gY/Aw+zWHAC+p2labGpqiorFYgGYCEzN7oQdQClN07O1/EfDyGgC0ALMBdYAi4FyK+4H3gLPsxfR1zRNi+NP7nH5J+QntnXe5B5mpfQAAAAASUVORK5CYII=">

<!-- Google fonts -->
<link href='https://fonts.googleapis.com/css?family=Lato:400,600,900' rel='stylesheet' type='text/css'/>
<link href='https://fonts.googleapis.com/css?family=Source+Code+Pro:400,500,600' rel='stylesheet' type='text/css'/>

<!-- CSS -->
<title>$title</title>
<link rel="stylesheet" type="text/css" href="$nimdoccss">

<script type="text/javascript" src="dochack.js"></script>

<script type="text/javascript">
function main() {
  var pragmaDots = document.getElementsByClassName("pragmadots");
  for (var i = 0; i < pragmaDots.length; i++) {
    pragmaDots[i].onclick = function(event) {
      // Hide tease
      event.target.parentNode.style.display = "none";
      // Show actual
      event.target.parentNode.nextElementSibling.style.display = "inline";
    }
  }

  const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');
  function switchTheme(e) {
      if (e.target.checked) {
          document.documentElement.setAttribute('data-theme', 'dark');
          localStorage.setItem('theme', 'dark');
      } else {
          document.documentElement.setAttribute('data-theme', 'light');
          localStorage.setItem('theme', 'light');
      }
  }

  toggleSwitch.addEventListener('change', switchTheme, false);


  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', "dark");
    toggleSwitch.checked = true;
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.setAttribute('data-theme', "light");
    toggleSwitch.checked = false;
  } else {
    const currentTheme = localStorage.getItem('theme') ? localStorage.getItem('theme') : null;
    if (currentTheme) {
      document.documentElement.setAttribute('data-theme', currentTheme);

      if (currentTheme === 'dark') {
        toggleSwitch.checked = true;
      }
    }
  }
}
</script>

</head>

<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Arraymancer - $title</title>

<link href="docutils.css" rel="stylesheet" type="text/css"/>
<link href="nav.css" rel="stylesheet" type="text/css"/>

<link href='http://fonts.googleapis.com/css?family=Raleway:400,600,900' rel='stylesheet' type='text/css'/>
<link href='http://fonts.googleapis.com/css?family=Source+Code+Pro:400,500,600' rel='stylesheet' type='text/css'/>

<a href="https://github.com/mratsim/arraymancer"><img style="position: fixed; top: 0; right: 0; border: 0; z-index: 10;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>

<body onload="main()">
<div class="document" id="documentId">
  <div class="container">
    <h1 class="title">$title</h1>
    $content
    <div class="row">
      <div class="twelve-columns footer">
        <span class="nim-sprite"></span>
        <br/>
        <small style="color: var(--hint);">Made with Nim. Generated: $date $time UTC</small>
      </div>
    </div>
  </div>
p</div>
$analytics
"""

const headerTmpl* = """
<header>
  <a class="pagetitle" href="index.html">Arraymancer</a>
  <span>
    <a href="#">Technical reference</a>
    <ul class="monospace">
      <span>
        <a href="#">Core tensor API</a>
        <ul class="monospace">
          $1
        </ul>
      </span>
      <span>
        <a href="#">Neural network API</a>
        <ul class="monospace">
          $2
        </ul>
      </span>
      <span>
        <a href="#">Linear algebra, stats, ML</a>
        <ul class="monospace">
          $3
        </ul>
      </span>
      <span>
        <a href="#">IO & Datasets</a>
        <ul class="monospace">
          $4
        </ul>
      </span>
      <span>
        <a href="#">Autograd</a>
        <ul class="monospace">
          $5
        </ul>
      </span>
      <span>
        <a href="#">Neuralnet primitives</a>
        <ul class="monospace">
          $6
        </ul>
      </span>
    </ul>
  </span>
  <span>
    <a href="#">Tutorial</a>
    <ul class="monospace">
      <li><a href="tuto.first_steps.html">First steps</a></li>
      <li><a href="tuto.slicing.html">Taking a slice of a tensor</a></li>
      <li><a href="tuto.linear_algebra.html">Matrix & vectors operations</a></li>
      <li><a href="tuto.broadcasting.html">Broadcasted operations</a></li>
      <li><a href="tuto.shapeshifting.html">Transposing, Reshaping, Permuting, Concatenating</a></li>
      <li><a href="tuto.map_reduce.html">Map & Reduce</a></li>
      <li><a href="tuto.iterators.html">Basic iterators</a></li>
    </ul>
  </span>
  <span>
    <a href="#">Spellbook (How-To&apos;s)</a>
    <ul class="monospace">
      <li><a href="howto.type_conversion.html">How to convert a Tensor type?</a></li>
      <li><a href="howto.ufunc.html">How to create a new universal function?</a></li>
      <li><a href="howto.perceptron.html">How to create a multilayer perceptron?</a></li>
    </ul>
  </span>
  <span>
    <a href="#">Under the hood</a>
    <ul class="monospace">
      <li><a href="uth.speed.html">How Arraymancer achieves its speed?</a></li>
      <li><a href="uth.copy_semantics.html">Why does `=` share data by default aka reference semantics?</a></li>
      <li><a href="uth.opencl_cuda_nim.html">Working with OpenCL and Cuda in Nim</a></li>
    </ul>
  </span>
</header>
</body>
</html>
"""

let nameMap = {
  "dsl_core.html" : "Neural network: Declaration",
  "relu.html" : "Activation: Relu (Rectified linear Unit)",
  "sigmoid.html" : "Activation: Sigmoid",
  "tanh.html" : "Activation: Tanh",
  "conv2D.html" : "Layers: Convolution 2D",
  "embedding.html" : "Layers: Embedding",
  "gru.html" : "Layers: GRU (Gated Linear Unit)",
  "linear.html" : "Layers: Linear/Dense",
  "maxpool2D.html" : "Layers: Maxpool 2D",
  "cross_entropy_losses.html" : "Loss: Cross-Entropy losses",
  "mean_square_error_loss.html" : "Loss: Mean Square Error",

  "reshape_flatten.html" : "Reshape & Flatten",

  "decomposition.html" : "Eigenvalue decomposition",
  "decomposition_rand.html" : "Randomized Truncated SVD",
  "least_squares.html" : "Least squares solver",
  "linear_systems.html" : "Linear systems solver",
  "special_matrices.html" : "Special linear algebra matrices",
  "stats.html" : "Statistics",
  "pca.html" : "Principal Component Analysis (PCA)",
  "accuracy_score.html" : "Accuracy score",
  "common_error_functions.html" : "Common errors, MAE and MSE (L1, L2 loss)",
  "kmeans.html" : "K-Means",

  "mnist.html" : "MNIST",
  "imdb.html" : "IMDB",
  "io_csv.html" : "CSV reading and writing",
  "io_hdf5.html" : "HDF5 files reading and writing",
  "io_image.html" : "Images reading and writing",
  "io_npy.html" : "Numpy files reading and writing",

  "autograd_common.html" : "Data structure",
  "gates_basic.html" : "Basic operations",
  "gates_blas.html" : "Linear algebra operations",
  "gates_hadamard.html" : "Hadamard product (elementwise matrix multiply)",
  "gates_reduce.html" : "Reduction operations",
  "gates_shapeshifting_concat_split.html" : "Concatenation, stacking, splitting, chunking operations",
  "gates_shapeshifting_views.html" : "Linear algebra operations",

  "nnp_activation.html" : "Activations",
  "nnp_convolution.html" : "Convolution 2D",
  "nnp_conv2D_cudnn.html" : "Convolution 2D - CuDNN",
  "nnp_embedding.html" : "Embeddings",
  "nnp_gru.html" : "Gated Recurrent Unit (GRU)",
  "nnp_linear.html" : "Linear / Dense layer",
  "nnp_maxpooling.html" : "Maxpooling",
  "nnp_numerical_gradient.html" : "Numerical gradient",
  "nnp_sigmoid_cross_entropy.html" : "Sigmoid Cross-Entropy loss",
  "nnp_softmax_cross_entropy.html" : "Softmax Cross-Entropy loss",
}.toTable

proc wrap(name: string): string =
  const tmpl = """<li><a href="$#">$#</a></li>"""
  if name in nameMap:
    result = tmpl % [name, nameMap[name]]
  else:
    result = tmpl % [name, name]

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
  let files = getFiles(path)
  let catMap = { "tensor" : 1,
                 "nn" : 2,
                 "nn_dsl" : 2,
                 "la" : 3,
                 "stats" : 3,
                 "ml" : 3,
                 "datasets" : 4,
                 "io" : 4,
                 "ag" : 5 ,
                 "nn_primitives" : 6 }.toTable
  var spanMap = newSeq[seq[string]](6)

  for file in files:
    let baseName = file.extractFilename()
    var outfile = baseName.replace(".nim", ".html")
    let subDir = file.parentDir.extractFilename
    if subDir in catMap:
      echo subDir
      let idx = catMap[subDir] - 1
      spanMap[idx].add outfile
    else:
      echo "!! subDir ", subDir

  var spans = newSeq[string](6)
  for idx in 0 ..< spans.len:
    spans[idx] = spanMap[idx].sorted.mapIt(wrap(it)).join("\n")
  # fill the HTML generation template from the filenames
  echo spans
  let htmlTmpl = headerTmpl % [ spans[0], spans[1], spans[2],
                                spans[3], spans[4], spans[5] ]

  # first "header"
  var fdata = ""
  fdata.add("# Arraymancer documentation generation\n\n")
  fdata.add(&"git.url = \"{gitUrl}\"\n\n")
  fdata.add(&"doc.item.seesrc = \"\"\"{docItemSeeSrc}\"\"\"\n\n")
  # finally write the HTML document template
  fdata.add(&"doc.file = \"\"\"{docFileTmpl}{htmlTmpl}\"\"\"\n")

  # now build the content for the spans
  writeFile(getProjectPath() & $DirSep & "nimdoc.cfg", fdata)
