# Arraymancer & Laser projects
# Copyright (c) 2017 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../dynamic_stack_arrays,
  ../tensor/datatypes

# Tools to manipulate deep nested containers

iterator flatIter*(s: string): string {.noSideEffect.} =
  yield s

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.}=
  ## Inline iterator on any-depth seq or array
  ## Returns values in order
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func getShape*(s: string, parent_shape = Metadata()): Metadata =
  ## Handle strings / avoid interpretation as openarray[char]
  const z = default(Metadata)
  if parent_shape == z:
    result = z
    result.len = 1
    result[0] = 1
  else: return parent_shape

func getShape*[T](s: openarray[T], parent_shape = Metadata()): Metadata =
  ## Get the shape of nested seqs/arrays
  ## Important ⚠: at each nesting level, only the length
  ##   of the first element is used for the shape.
  ##   Ensure before or after that seqs have the expected length
  ##   or that the total number of elements matches the product of the dimensions.

  result = parent_shape
  result.add(s.len)

  when (T is seq|array):
    result = getShape(s[0], result)
