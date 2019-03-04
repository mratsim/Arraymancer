# Copyright (c) 2019 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
    strutils,
    ../tensor/tensor

iterator whitespaceTokenizer*(input: Tensor[string]): seq[string] =
    for element in input:
        yield splitWhitespace(element)