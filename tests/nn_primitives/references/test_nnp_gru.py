import torch
import torch.nn as nn

batch_size = 3
features = 5
hidden_size = 2 # weights and bias have 3x2 = 6 size
num_stacked_layers = 2
seq_len = 4

x = torch.tensor([
    [ [ 0.1,  0.2,  0.3,  0.4,  0.5],  # Sequence/timestep 1
      [-0.1, -0.2, -0.3, -0.4, -0.5],
      [ 0.5,  0.6,  0.7,  0.8,  0.9]],
    [ [-0.1, -0.2, -0.3, -0.4, -0.5],  # Sequence/timestep 2
      [-0.1, -0.2, -0.3, -0.4, -0.5],
      [ 0.5,  0.6,  0.7,  0.8,  0.9]],
    [ [ 0.1,  0.2,  0.3,  0.4,  0.5],  # Sequence/timestep 3
      [-0.1, -0.2, -0.3, -0.4, -0.5],
      [-0.1, -0.2, -0.3, -0.4, -0.5]],
    [ [-0.1, -0.2, -0.3, -0.4, -0.5],  # Sequence/timestep 4
      [-0.1, -0.2, -0.3, -0.4, -0.5],
      [-0.1, -0.2, -0.3, -0.4, -0.5]]
])

hidden = torch.tensor([
  [ [ -1.0, -1.0],  # Stacked layer 1
    [ -1.0, -1.0],
    [ -1.0, -1.0]],
  [ [  2.0,  3.0],  # Stacked layer 2
    [  2.0,  3.0],
    [  2.0,  3.0]]
])

w_input = [
  torch.tensor(
    [ [0.9, 0.8, 0.7, 0.6, 0.5], # Stacked layer 1
      [0.8, 0.7, 0.6, 0.5, 0.4],
      [0.7, 0.6, 0.5, 0.4, 0.3],
      [0.6, 0.5, 0.4, 0.3, 0.2],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.4, 0.3, 0.2, 0.1, 0.0]]),
  torch.tensor(
    [ [0.5, 0.6], # Stacked layer 2
      [0.4, 0.5],
      [0.3, 0.4],
      [0.2, 0.3],
      [0.1, 0.2],
      [0.0, 0.1]])
]

w_recur = torch.tensor([
  [ [-0.3, -0.1], # Stacked layer 1
    [-0.2,  0.0],
    [-0.3, -0.1],
    [-0.2,  0.0],
    [-0.3, -0.1],
    [-0.2,  0.0]],
  [ [-0.4, -0.5], # Stacked layer 2
    [-0.4,  0.5],
    [-0.4, -0.5],
    [-0.4,  0.5],
    [-0.4, -0.5],
    [-0.4,  0.5]]
])

b_input = torch.tensor([
  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], # Stacked layer 1
  [[0.2, 0.3, 0.4, 0.2, 0.3, 0.4]]  # Stacked layer 2
])

b_recur = torch.tensor([
  [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]], # Stacked layer 1
  [[ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6]], # Stacked layer 2
])

test_gru = nn.GRU(features, hidden_size, num_stacked_layers)

test_gru.weight_ih_l0.data = w_input[0]
test_gru.weight_ih_l1.data = w_input[1]
test_gru.weight_hh_l0.data = w_recur[0]
test_gru.weight_hh_l1.data = w_recur[1]
test_gru.bias_ih_l0.data   = b_input[0]
test_gru.bias_ih_l1.data   = b_input[1]
test_gru.bias_hh_l0.data   = b_recur[0]
test_gru.bias_hh_l1.data   = b_recur[1]

output, hiddenN = test_gru(x, hidden)

print(output)
print('\n###\n')
print(hiddenN)

# tensor([[[ 0.2564,  2.4324],
#          [ 0.3213,  2.4822],
#          [ 0.1984,  2.3850]],

#         [[ 0.1324,  2.1940],
#          [ 0.1591,  2.2447],
#          [ 0.0683,  2.1121]],

#         [[ 0.1234,  1.9977],
#          [ 0.1320,  2.0435],
#          [ 0.0904,  1.9112]],

#         [[ 0.1356,  1.8196],
#          [ 0.1370,  1.8613],
#          [ 0.1231,  1.7382]]])

# ###

# tensor([[[ 0.0155,  0.1680],
#          [-0.0039,  0.1854],
#          [-0.0549,  0.1189]],

#         [[ 0.1356,  1.8196],
#          [ 0.1370,  1.8613],
#          [ 0.1231,  1.7382]]])
