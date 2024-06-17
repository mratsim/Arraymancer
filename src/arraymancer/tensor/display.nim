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

import  ./private/p_display,
        ./data_structure
import std / typetraits

proc pretty*[T](t: Tensor[T], precision: int = -1, showHeader: static bool = true): string =
  ## Pretty-print a Tensor as a "table" with a given precision and optional header
  ##
  ## Pretty-print a Tensor with options to set a custom `precision` for float
  ## values and to show or hide a header describing the tensor type and shape.
  ##
  ## Inputs:
  ##   - Input Tensor.
  ##   - precision: The number of decimals printed (for float tensors),
  ##                _including_ the decimal point.
  ##   - showHeader: If true (the default) show a dscription header
  ##                 indicating the tensor type and shape.
  ## Result:
  ##   - A string containing a "pretty-print" representation of the Tensor.
  ##
  ## Examples:
  ## ```nim
  ## let t = arange(-2.0, 4.0).reshape(2, 3)
  ##
  ## echo t.pretty()
  ## # Tensor[system.float] of shape "[2, 3]" on backend "Cpu"
  ## # |-2    -1     0|
  ## # |1      2     3|
  ##
  ## # Note that the precision counts
  ## echo t.pretty(2)
  ## # Tensor[system.float] of shape "[2, 3]" on backend "Cpu"
  ## # |-2.0    -1.0     0.0|
  ## # |1.0      2.0     3.0|
  ## ```
  const specifier = if showHeader: "" else: "||"
  t.prettyImpl(precision = precision, specifier = specifier)

proc pretty*[T](t: Tensor[T], specifier: static string = ""): string =
  ## Pretty-print a Tensor with the option to set a custom format `specifier`
  ##
  ## The "format specifier" is similar to those used in format strings, with
  ## the addition of a few, tensor specific modifiers (shown below).
  ##
  ## Inputs:
  ##   - Input Tensor
  ##   - specifier: A format specifier similar to those used in format strings,
  ##                with the addition of a few, tensor specific modifiers which
  ##                can be combined to achieve different results:
  ##                - "[:]": Display the tensor as if it were a nim "array".
  ##                         This makes it easy to use the representation of a
  ##                         tensor in your own code. No header is shown.
  ##                - "[]": Same as "[:]" but displays the tensor in a single
  ##                        line. No header is shown.
  ##                - "<>": Combined with the 2 above (i.e. "<>[:]" or "<>[]")
  ##                        adds a header with basic tensor info (type and
  ##                        shape). "<:>" can be used as a shortcut for "<>[:]"
  ##                        while "<>" on its own is equivalent to "<>[]".
  ##                        Can also be combined with "<>||" (see below).
  ##                - "||": "Pretty-print" the tensor _without_ a header. This
  ##                        can also be combined with "<>" (i.e. "<>||") to
  ##                        explicitly enable the default mode, which is pretty
  ##                        printing with a header.
  ## Notes:
  ##   - Note that in addition to these we support all of the standard format
  ##     specifiers, such as "f", "g", "+", etc (and including, in nim 2.2 and
  ##     above, the 'j' specifier for complex tensors). For a list of supported
  ##     format specifiers please check the documentation of nim's `strformat`
  ##     module.
  ##   - This version of this function does not have a `showHeader` argument
  ##     because to disable the header you must add a "n" to the format
  ##     specifier.
  ##
  ## Examples:
  ## ```nim
  ## let t_int = arange(-2, 22, 4).reshape(2, 3)
  ##
  ## # You can specify a format for the elements in the tensor
  ## # Note that the default is "pretty-printing" the tensor
  ## # _and_ showing a header describing its type and shape
  ## echo t_int.pretty("+05X")
  ## # Tensor[system.int] of shape "[2, 3]" on backend "Cpu"
  ## # |-0002    +0002    +0006|
  ## # |+000a    +000e    +0012|
  ##
  ## # The header can be disabled by using "||"
  ## echo t_int.pretty("+05X||")
  ## # |-0002    +0002    +0006|
  ## # |+000a    +000e    +0012|
  ##
  ## # Use the "[:]" format specifier to print the tensor as a
  ## # "multi-line array" _without_ a header
  ## echo t_int.pretty("[:]")
  ## # [[-2, 2, 6],
  ## #  [10, 14, 18]]
  ##
  ## # Enable the header adding "<>" (i.e. "<>[:]") or the shorter "<:>"
  ## echo t_int.pretty("<:>")
  ## # Tensor[int]<2,3>:
  ## # [[-2, 2, 6],
  ## #  [10, 14, 18]]
  ##
  ## # The "[]" specifier is similar to "[:]" but prints on a single line
  ## echo t_int.pretty("[]")
  ## # Tensor[float]<2,3>:[[-2.00, -1.00, +0.00], [+1.00, +2.00, +3.00]]
  ##
  ## # You can also enable the header using "<>" or "<>[]"
  ## echo t_int.pretty("<>")
  ## # [[-2.00, -1.00, +0.00], [+1.00, +2.00, +3.00]]
  ##
  ## # You can combine "[]", "[:]", "<>" and "<:>" with a regular format spec:
  ## let t_float = arange(-2.0, 22.0, 4.0).reshape(2, 3)
  ##
  ## echo t_float.pretty("6.2f<:>")
  ## # Tensor[int]<2,3>:
  ## # [[ -2.00,   2.00,   6.00],
  ## #  [ 10.00,  14.00,  18.00]]
  ## ```
  t.prettyImpl(precision = -1, specifier = specifier)

proc `$`*[T](t: Tensor[T]): string =
  ## Pretty-print a tensor (when using ``echo`` for example)
  t.pretty()

proc `$$`*[T](t: Tensor[T]): string =
  ## Print the "elements" of a tensor as a multi-line array
  t.pretty(specifier = "<:>")

proc `$<>`*[T](t: Tensor[T]): string =
  ## Print the "elements" of a tensor as a single-line array
  t.pretty(specifier = "<>")

proc formatValue*[T](result: var string, t: Tensor[T], specifier: static string) =
  ## Standard format implementation for `Tensor`. It makes little
  ## sense to call this directly, but it is required to exist
  ## by the `&` macro.
  ##
  ## For Tensors, we add some additional specifiers which can be combined to
  ## achieve different results:
  ## - "[:]": Display the tensor as if it were a nim "array".
  ##          This makes it easy to use the representation of a
  ##          tensor in your own code. No header is shown.
  ## - "[]": Same as "[:]" but displays the tensor in a single
  ##         line. No header is shown.
  ## - "<>": Combined with the 2 above (i.e. "<>[:]" or "<>[]")
  ##         adds a header with basic tensor info (type and
  ##         shape). "<:>" can be used as a shortcut for "<>[:]"
  ##         while "<>" on its own is equivalent to "<>[]".
  ##         Can also be combined with "<>||" (see below).
  ## - "||": "Pretty-print" the tensor _without_ a header. This
  ##         can also be combined with "<>" (i.e. "<>||") to
  ##         explicitly enable the default mode, which is pretty
  ##         printing with a header.
  ## - 'j': Formats complex values as (A+Bj) like in mathematics.
  ##        Ignored for non Complex tensors
  if specifier.len == 0:
    result.add $t
  else:
    result.add t.pretty(specifier = specifier)
