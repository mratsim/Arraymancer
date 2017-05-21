# Copyright 2017 Mamy André-Ratsimbazafy
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


## This file with slicing syntactic sugar.
## Foo being:
# Tensor of shape 5x5 of type "int" on backend "Cpu"
# |1      1       1       1       1|
# |2      4       8       16      32|
# |3      9       27      81      243|
# |4      16      64      256     1024|
# |5      25      125     625     3125|
#
## Target supported syntax is in `test_accessors_slicer`
#
### Slicing
# Basic indexing - foo[2, 3]
# Basic indexing - foo[1+1, 2*2*1]
# Basic slicing - foo[1..2, 3]
# Basic slicing - foo[1+1..4, 3-2..2]
# Span slices - foo[_, 3]
# Span slices - foo[1.._, 3]
# Span slices - foo[_..3, 3]
# Span slices - foo[_.._, 3]
# Stepping - foo[1..3|2, 3]
# Span stepping - foo[_.._|2, 3]
# Span stepping - foo[_.._|+2, 3]
# Span stepping - foo[1.._|1, 2..3]
# Span stepping - foo[_..<4|2, 3]
# Slicing until at n from the end - foo[0..^4, 3]
# Span Slicing until at n from the end - foo[_..^2, 3]
# Stepped Slicing until at n from the end - foo[1..^1|2, 3]
# Slice from the end - foo[^1..0|-1, 3]
# Slice from the end - expect non-negative step error - foo[^1..0, 3]
# Slice from the end - foo[^(2*2)..2*2, 3]
# Slice from the end - foo[^3..^2, 3]

### Assignement
# Slice to a single value - foo[1..2, 3..4] = 999
# Slice to an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
# Slice to values from a view/Tensor - foo[^2..^1,2..4] = bar
# Slice to values from view of same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

######################################
type SteppedSlice* = object
    ## A slice with step information and start is from beginning or end of range
    a, b: int
    step: int
    a_from_end: bool
    b_from_end: bool

## Necessary to avoid parenthesis due to operator precedence of | over ..
## [0..10|1] is intepreted as [0..(10|1)]
type Step* = object
    ## Holds the end of rang eand step
    b: int
    step: int

proc check_steps(a,b, step:int) {.noSideEffect.}=
    ## Though it might be convenient to automatically step in the correct direction like in Python
    ## I choose not to do it as this might introduce the typical silent bugs typechecking/Nim is helping avoid.
    if ((b-a) * step < 0):
        raise newException(IndexError, "Your slice start: " &
                                $a &
                                ", and stop: " &
                                $b &
                                ", or your step: " &
                                $step &
                                """, are not correct. If your step is positive
                                start must be inferior to stop and inversely if your step is negative
                                start must be superior to stop.""")

proc check_shape(a, b: Tensor|openarray) {.noSideEffect.}=
    ## Compare shape
    if a.shape != b.shape:
        raise newException(IndexError, "Your tensors or openarrays do not have the same shape: " & $a.shape.join("x") &
                                " and " & $b.shape.join("x"))

## Procs to manage all integer, slice, SteppedSlice 
proc `|`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
    return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|`*(b, step: int): Step {.noSideEffect, inline.}=
    return Step(b: b, step: step)

proc `|`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
    result = ss
    result.step = step

proc `|+`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
    ## Alias
    return `|`(s, step)

proc `|+`*(b, step: int): Step {.noSideEffect, inline.}=
    ## Alias
    return `|`(b, step)

proc `|+`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
    ## Alias
    return `|`(ss, step)

proc `|-`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
    return SteppedSlice(a: s.a, b: s.b, step: -step)

proc `|-`*(b, step: int): Step {.noSideEffect, inline.}=
    return Step(b: b, step: -step)

proc `|-`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
    result = ss
    result.step = -step

proc `..`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
    return SteppedSlice(a: a, b: s.b, step: s.step)

proc `..<`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
    return SteppedSlice(a: a, b: <s.b, step: s.step)

proc `..^`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
    return SteppedSlice(a: a, b: s.b, step: s.step, b_from_end: true)

proc `^`*(s: SteppedSlice): SteppedSlice {.noSideEffect, inline.} =
    ## Note: This does not automatically inverse stepping, what if we want ^5..^1
    result = s
    result.a_from_end = not result.a_from_end

proc `^`*(s: Slice): SteppedSlice {.noSideEffect, inline.} =
    ## Note: This does not automatically inverse stepping, what if we want ^5..^1
    return SteppedSlice(a: s.a, b: s.b, step: 1, a_from_end: true)

## span is equivalent to `:` in Python. It returns the whole axis range.
## Tensor[_, 3] will be replaced by Tensor[span, 3]
const span* = SteppedSlice(b: 1, step: 1, b_from_end: true)

macro desugar(args: untyped): typed =
    ## Transform all syntactic sugar in arguments to integer or SteppedSlices
    ## It will then be dispatched to "atIndex" (if specific integers)
    ## or "slicer" if there are SteppedSlices

    # echo "\n------------------\nOriginal tree"
    # echo args.treerepr
    var r = newNimNode(nnkArglist)

    for nnk in children(args):

        ###### Traverse top tree nodes and one-hot-encode the different conditions

        # Node is "_"
        let nnk_joker = nnk == ident("_")

        # Node is of the form "* .. *"
        let nnk0_inf_dotdot = (
            nnk.kind == nnkInfix and
            nnk[0] == ident("..")
        )

        # Node is of the form "* ..< *" or "* ..^ *"
        let nnk0_inf_dotdot_alt = (
            nnk.kind == nnkInfix and (
                nnk[0] == ident("..<") or
                nnk[0] == ident("..^")
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
            nnk[0] == ident("^")
        )

        # Node is of the form "_ `op` *"
        let nnk1_joker = (
            nnk.kind == nnkInfix and
            nnk[1] == ident("_")
        )

        # Node is of the form "_ `op` *"
        let nnk10_hat = (
            nnk.kind == nnkInfix and
            nnk[1].kind == nnkPrefix and
            nnk[1][0] == ident("^")
        )

        # Node is of the form "* `op` _"
        let nnk2_joker = (
            nnk.kind == nnkInfix and
            nnk[2] == ident("_")
        )

        # Node is of the form "* `op` * | *" or "* `op` * |+ *"
        let nnk20_bar_pos = (
            nnk.kind == nnkInfix and
            nnk[2].kind == nnkInfix and (
                nnk[2][0] == ident("|") or
                nnk[2][0] == ident("|+")
            )
        )

        # Node is of the form "* `op` * |- *"
        let nnk20_bar_min = (
            nnk.kind == nnkInfix and
            nnk[2].kind == nnkInfix and
            nnk[2][0] == ident("|-")
        )

        # Node is of the form "* `op` * | *" or "* `op` * |+ *" or "* `op` * |- *"
        let nnk20_bar_all = nnk20_bar_pos or nnk20_bar_min

        # Node is of the form "* `op1` _ `op2` *"
        let nnk21_joker = (
            nnk.kind == nnkInfix and
            nnk[2].kind == nnkInfix and
            nnk[2][1] == ident("_")
        )

        ###### Core desugaring logic
        if nnk_joker:
            ## [_, 3] into [span, 3]
            r.add(ident("span"))
        elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
            ## [_.._, 3] into [span, 3]
            r.add(ident("span"))
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
            raise newException(IndexError, "Please use explicit end of range " &
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

proc slicer*[B, T](t: Tensor[B, T], slices: varargs[SteppedSlice]): Tensor[B, T] {.noSideEffect.}=
    ## Take a Tensor and SteppedSlices
    ## Returns:
    ##    A view of the original Tensor
    ##    Data is not changed, only offset and strides are changed to achieve the desired effect.
    ##    TODO: Currently, BLAS needs C-contiguous data and does not work with "Universal" strides.
    ##          Provide a way to convert from Universal to C-contiguous. (Strided iterator should make that easy)

    result = t # For t.data, seq semantics should copy only on write. TODO: Test

    for i, slice in slices:
        # Check if we start from the end
        let a = if slice.a_from_end: result.shape[i] - slice.a
                else: slice.a

        let b = if slice.b_from_end: result.shape[i] - slice.b
                else: slice.b

        # Bounds checking
        when compileOption("boundChecks"): check_steps(a,b, slice.step)

        # Compute offset:
        result.offset += a * result.strides[i]
        # Now change shape and strides
        result.strides[i] *= slice.step
        result.shape[i] = abs((b-a) div slice.step) + 1

macro inner_typed_dispatch(t: typed, args: varargs[typed]): untyped =
    ## Typed macro so that isAllInt has typed context and we can dispatch.
    ## If args are all int, we dispatch to atIndex and return T
    ## Else, all ints are converted to SteppedSlices and we return a Tensor.
    ## Note, normal slices and `_` were already converted in the `[]` macro
    ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though
    if isAllInt(args):
        result = newCall("atIndex", t)
        for slice in args:
            result.add(slice)
    else:
        result = newCall("slicer", t)
        for slice in args:
            if isInt(slice):
                ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
                result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
            else:
                result.add(slice)

macro `[]`*[B, T](t: Tensor[B,T], args: varargs[untyped]): untyped =
    let new_args = getAST(desugar(args))

    result = quote do:
        inner_typed_dispatch(`t`, `new_args`)

proc slicerMut*[B, T](t: var Tensor[B, T], slices: varargs[SteppedSlice], val: T) {.noSideEffect.}=
    ## Assign the value to the whole slice
    let sliced = t.slicer(slices)
    for real_idx in sliced.real_indices:
        t.data[real_idx] = val

proc slicerMut*[B, T](t: var Tensor[B, T], slices: varargs[SteppedSlice], oa: openarray) =
    ## Assign value from openarrays
    ## The openarray must have the same shape as the slice
    let sliced = t.slicer(slices)
    when compileOption("boundChecks"):
        check_shape(sliced, oa)

    let data = toSeq(flatIter(oa))
    when compileOption("boundChecks"):
        check_nested_elements(oa.shape, data.len)

    # Unfortunately we need to loop twice over data/oa
    # Reason 1: we can't check the iterator length before consuming it
    # Reason 2: we can't capture an open array, i.e. do zip(sliced.real_indices, flatClosureIter(oa))
    for real_idx, val in zip(sliced.real_indices, data):
        t.data[real_idx] = val

proc slicerMut*[B1, B2, T](t: var Tensor[B1, T], slices: varargs[SteppedSlice], t2: Tensor[B2, T]) {.noSideEffect.}=
    ## Assign the value to the whole slice
    let sliced = t.slicer(slices)

    when compileOption("boundChecks"): check_shape(sliced, t2)

    for real_idx, val in zip(sliced.real_indices, t2.values):
        t.data[real_idx] = val

macro inner_typed_dispatch_mut(t: typed, args: varargs[typed], val: typed): untyped =
    ## Assign `val` to Tensor T at slice/position `args`
    if isAllInt(args):
        result = newCall("atIndexMut", t)
        for slice in args:
            result.add(slice)
        result.add(val)
    else:
        result = newCall("slicerMut", t)
        for slice in args:
            if isInt(slice):
                ## Convert [10, 1..10|1] to [10..10|1, 1..10|1]
                result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
            else:
                result.add(slice)
        result.add(val)

macro `[]=`*[B, T](t: var Tensor[B,T], args: varargs[untyped]): untyped =
    ## varargs[untyped] consumes all arguments so the actual value should be popped
    ## https://github.com/nim-lang/Nim/issues/5855

    var tmp = args
    let val = tmp.pop
    let new_args = getAST(desugar(tmp))

    result = quote do:
        inner_typed_dispatch_mut(`t`, `new_args`,`val`)