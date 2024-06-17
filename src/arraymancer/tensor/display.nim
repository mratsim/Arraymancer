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
  ##   - showHeader: If true (the default) show a description header
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
  ##                which are used to control how the tensor and its elements
  ##                are displayed.
  ##                All of the standard format specifiers, such as "f", "g",
  ##                "+", etc. (and including, in nim 2.2 and above, the 'j'
  ##                specifier for complex tensors) can be used (check the
  ##                documentation of nim's `strformat` module for more info).
  ##                In addition to those standard format specifiers which
  ##                control how the elements of the tensor are displayed, you
  ##                can use a few, tensor specific modifiers which can be
  ##                combined to achieve different results:
  ##                - "[:]": Display the tensor as if it were a nim "array".
  ##                         This makes it easy to use the representation of a
  ##                         tensor in your own code. No header is shown.
  ##                - "<>[:]": Same as "[:]" but displays a header describing
  ##                           the tensor type and shape.
  ##                - "[]": Same as "[:]" but displays the tensor in a single
  ##                        line. No header is shown.
  ##                - "<>[:]": Same as "[]" but displays a header describing
  ##                           the tensor type and shape.
  ##                - "||": "Pretty-print" the tensor _without_ a header.
  ##                - "<>||": Same as "||" but displays a header describing the
  ##                          tensor type and shape. This is the default display
  ##                          mode.
  ##                - "<>": When used on its own (i.e. not combined with "[:]",
  ##                        "[]" and "||" as shown above) it is equivalent to
  ##                        "<>[]" (i.e. single-line, nim-like representation
  ##                        with a header). Cannot wrap the element specifier.
  ##
  ## Notes:
  ##   - The default format (i.e. when none of these tensor-specific tokens is
  ##     used) is pretty printing with a header (i.e. "<>||").
  ##   - These tensor specific modifiers can placed at the start or at the end
  ##     of the format specifier (e.g. "<>[:]06.2f", or "06.2f<>[:]"), or they
  ##     can "wrap" the element specifier (e.g. "<>[:06.2f]").
  ##   - When wrapping you cannot use the "compact forms" of these specifiers
  ##     (i.e. "<:>" and "<|>").
  ##   - When using the wrapping form of "[:]", start with "[:" and end with
  ##     either "]" or ":]" (i.e. both "[:06.2f]" and "[:06.2f:]" are valid).
  ##   - This version of this function does not have a `showHeader` argument
  ##     because to disable the header you can simply select the right format
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
  ## # |+000A    +000E    +0012|
  ##
  ## # The header can be disabled by using "||"
  ## echo t_int.pretty("+05X||")
  ## # |-0002    +0002    +0006|
  ## # |+000A    +000E    +0012|
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
  ## # [[-2, 2, 6], [10, 14, 18]]
  ##
  ## # You can also enable the header using "<>" or "<>[]"
  ## echo t_int.pretty("<>")
  ## # Tensor[int]<2,3>:[[-2, 2, 6], [10, 14, 18]]
  ##
  ## # You can combine "[]", "[:]", "<>" and "<:>" with a regular format spec:
  ## let t_float = arange(-2.0, 22.0, 4.0).reshape(2, 3)
  ##
  ## echo t_float.pretty("6.2f<:>")
  ## # Tensor[int]<2,3>:
  ## # [[ -2.00,   2.00,   6.00],
  ## #  [ 10.00,  14.00,  18.00]]
  ##
  ## # The equivalent to the wrapping form is:
  ## echo t_float.pretty("<>[:6.2f]")
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
  ## Standard format implementation for `Tensor`
  ##
  ## Note that it makes little sense to call this directly, but it is required
  ## to exist by the `&` macro.
  ##
  ## See the documentation of the `pretty` procedure for more information on the
  ## supported format specifiers (which include a number of tensor specific
  ## tokens).
  if specifier.len == 0:
    result.add $t
  else:
    result.add t.pretty(specifier = specifier)
