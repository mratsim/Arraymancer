# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import  ../accessors_macros_syntax,
        ../../std_version_types,
        macros

# span is equivalent to `:` in Python. It returns the whole axis range.
# Tensor[_, 3] will be replaced by Tensor[span, 3]
const Span = SteppedSlice(b: 1, step: 1, b_from_end: true)


# #########################################################################
# Slicing macros - desugaring AST

macro desugar*(args: untyped): void =
  ## Transform all syntactic sugar in arguments to integer or SteppedSlices
  ## It will then be dispatched to "atIndex" (if specific integers)
  ## or "slicer" if there are SteppedSlices

  # echo "\n------------------\nOriginal tree"
  # echo args.treerepr
  var r = newNimNode(nnkArglist)

  for nnk in children(args):

    ###### Traverse top tree nodes and one-hot-encode the different conditions

    # Node is "_"
    let nnk_joker = eqIdent(nnk, "_")

    # Node is of the form "* .. *"
    let nnk0_inf_dotdot = (
      nnk.kind == nnkInfix and
      eqIdent(nnk[0], "..")
    )

    # Node is of the form "* ..< *" or "* ..^ *"
    let nnk0_inf_dotdot_alt = (
      nnk.kind == nnkInfix and (
        eqIdent(nnk[0], "..<") or
        eqident(nnk[0], "..^")
      )
    )

    # Node is of the form "* .. *", "* ..< *" or "* ..^ *"
    let nnk0_inf_dotdot_all = (
      nnk0_inf_dotdot or
      nnk0_inf_dotdot_alt
    )

    # Node is of the form "^ *"
    let nnk0_pre_hat = (
      nnk.kind == nnkPrefix and
      eqIdent(nnk[0], "^")
    )

    # Node is of the form "_ `op` *"
    let nnk1_joker = (
      nnk.kind == nnkInfix and
      eqIdent(nnk[1], "_")
    )

    # Node is of the form "_ `op` *"
    let nnk10_hat = (
      nnk.kind == nnkInfix and
      nnk[1].kind == nnkPrefix and
      eqident(nnk[1][0], "^")
    )

    # Node is of the form "* `op` _"
    let nnk2_joker = (
      nnk.kind == nnkInfix and
      eqident(nnk[2], "_")
    )

    # Node is of the form "* `op` * | *" or "* `op` * |+ *"
    let nnk20_bar_pos = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and (
        eqident(nnk[2][0], "|") or
        eqIdent(nnk[2][0], "|+")
      )
    )

    # Node is of the form "* `op` * |- *"
    let nnk20_bar_min = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and
      eqIdent(nnk[2][0], "|-")
    )

    # Node is of the form "* `op` * | *" or "* `op` * |+ *" or "* `op` * |- *"
    let nnk20_bar_all = nnk20_bar_pos or nnk20_bar_min

    # Node is of the form "* `op1` _ `op2` *"
    let nnk21_joker = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and
      eqIdent(nnk[2][1], "_")
    )

    ###### Core desugaring logic
    if nnk_joker:
      ## [_, 3] into [Span, 3]
      r.add(bindSym("Span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
      ## [_.._, 3] into [Span, 3]
      r.add(bindSym("Span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk20_bar_all and nnk21_joker:
      ## [_.._|2, 3] into [0..^1|2, 3]
      ## [_.._|+2, 3] into [0..^1|2, 3]
      ## [_.._|-2 doesn't make sense and will throw out of bounds
      r.add(infix(newIntLitNode(0), "..^", infix(newIntLitNode(1), $nnk[2][0], nnk[2][2])))
    elif nnk0_inf_dotdot_all and nnk1_joker and nnk20_bar_all:
      ## [_..10|1, 3] into [0..10|1, 3]
      ## [_..^10|1, 3] into [0..^10|1, 3]   # ..^ directly creating SteppedSlices may introduce issues in seq[0..^10]
                                            # Furthermore ..^10|1, would have ..^ with precedence over |
      ## [_..<10|1, 3] into [0..<10|1, 3]
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2][1], $nnk[2][0], nnk[2][2])))
    elif nnk0_inf_dotdot_all and nnk1_joker:
      ## [_..10, 3] into [0..10|1, 3]
      ## [_..^10, 3] into [0..^10|1, 3]   # ..^ directly creating SteppedSlices from int in 0..^10 may introduce issues in seq[0..^10]
      ## [_..<10, 3] into [0..<10|1, 3]
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_inf_dotdot and nnk2_joker:
      ## [1.._, 3] into [1..^1|1, 3]
      r.add(infix(nnk[1], "..^", infix(newIntLitNode(1), "|", newIntLitNode(1))))
    elif nnk0_inf_dotdot and nnk20_bar_pos and nnk21_joker:
      ## [1.._|1, 3] into [1..^1|1, 3]
      ## [1.._|+1, 3] into [1..^1|1, 3]
      r.add(infix(nnk[1], "..^", infix(newIntLitNode(1), "|", nnk[2][2])))
    elif nnk0_inf_dotdot and nnk20_bar_min and nnk21_joker:
      ## Raise error on [5.._|-1, 3]
      raise newException(IndexDefect, "Please use explicit end of range " &
                       "instead of `_` " &
                       "when the steps are negative")
    elif nnk0_inf_dotdot_all and nnk10_hat and nnk20_bar_all:
      # We can skip the parenthesis in the AST
      ## [^1..2|-1, 3] into [^(1..2|-1), 3]
      r.add(prefix(infix(nnk[1][1], $nnk[0], nnk[2]), "^"))
    elif nnk0_inf_dotdot_all and nnk10_hat:
      # We can skip the parenthesis in the AST
      ## [^1..2*3, 3] into [^(1..2*3|1), 3]
      ## [^1..0, 3] into [^(1..0|1), 3]
      ## [^1..<10, 3] into [^(1..<10|1), 3]
      ## [^10..^1, 3] into [^(10..^1|1), 3]
      r.add(prefix(infix(nnk[1][1], $nnk[0], infix(nnk[2],"|",newIntLitNode(1))), "^"))
    elif nnk0_inf_dotdot_all and nnk20_bar_all:
      ## [1..10|1] as is
      ## [1..^10|1] as is
      r.add(nnk)
    elif nnk0_inf_dotdot_all:
      ## [1..10, 3] to [1..10|1, 3]
      ## [1..^10, 3] to [1..^10|1, 3]
      ## [1..<10, 3] to [1..<10|1, 3]
      r.add(infix(nnk[1], $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_pre_hat:
      ## [^2, 3] into [^2..^2|1, 3]
      r.add(prefix(infix(nnk[1], "..^", infix(nnk[1],"|",newIntLitNode(1))), "^"))
    else:
      r.add(nnk)
  # echo "\nAfter modif"
  # echo r.treerepr
  return r
