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

import  ./nn_primitives/nnp_activation,
        ./nn_primitives/nnp_convolution,
        ./nn_primitives/nnp_linear,
        ./nn_primitives/nnp_sigmoid_cross_entropy,
        ./nn_primitives/nnp_softmax_cross_entropy,
        ./nn_primitives/nnp_maxpooling,
        ./nn_primitives/nnp_softmax,
        ./nn_primitives/nnp_numerical_gradient,
        ./nn_primitives/nnp_gru,
        ./nn_primitives/nnp_embedding

export  nnp_activation,
        nnp_convolution,
        nnp_linear,
        nnp_sigmoid_cross_entropy,
        nnp_softmax_cross_entropy,
        nnp_maxpooling,
        nnp_softmax,
        nnp_numerical_gradient,
        nnp_gru,
        nnp_embedding

import nn_primitives/private/p_nnp_types
export Size2D

when defined(cudnn) or defined(nimdoc) or defined(nimsuggest):
  import  ./nn_primitives/backend/cudnn,
          ./nn_primitives/nnp_conv2d_cudnn

  export nnp_conv2d_cudnn
