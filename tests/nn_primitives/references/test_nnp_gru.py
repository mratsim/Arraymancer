import torch
import torch.nn as nn

batch_size = 3
features = 5
hidden_size = 2 # weights and bias have 3x2 = 6 size
num_stacked_layers = 2
seq_len = 4

torch.set_printoptions(precision=10)

x = torch.DoubleTensor([
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

hidden = torch.DoubleTensor([
  [ [ -1.0, -1.0],  # Stacked layer 1
    [ -1.0, -1.0],
    [ -1.0, -1.0]],
  [ [  2.0,  3.0],  # Stacked layer 2
    [  2.0,  3.0],
    [  2.0,  3.0]]
])

w_input = [
  torch.DoubleTensor(
    [ [0.9, 0.8, 0.7, 0.6, 0.5], # Stacked layer 1
      [0.8, 0.7, 0.6, 0.5, 0.4],
      [0.7, 0.6, 0.5, 0.4, 0.3],
      [0.6, 0.5, 0.4, 0.3, 0.2],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.4, 0.3, 0.2, 0.1, 0.0]]),
  torch.DoubleTensor(
    [ [0.5, 0.6], # Stacked layer 2
      [0.4, 0.5],
      [0.3, 0.4],
      [0.2, 0.3],
      [0.1, 0.2],
      [0.0, 0.1]])
]

w_recur = torch.DoubleTensor([
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

b_input = torch.DoubleTensor([
  [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], # Stacked layer 1
  [[0.2, 0.3, 0.4, 0.2, 0.3, 0.4]]  # Stacked layer 2
])

b_recur = torch.DoubleTensor([
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

# tensor([[[0.2564464804, 2.4324064858],
#          [0.3212793315, 2.4821776260],
#          [0.1983539289, 2.3849941275]],

#         [[0.1323593874, 2.1939630024],
#          [0.1591080583, 2.2447442148],
#          [0.0683269257, 2.1121420346]],

#         [[0.1233993158, 1.9977140846],
#          [0.1320258443, 2.0434525564],
#          [0.0904040251, 1.9111566097]],

#         [[0.1355803839, 1.8195602154],
#          [0.1369919053, 1.8612790359],
#          [0.1230976350, 1.7381913793]]], dtype=torch.float64)

# ###

# tensor([[[0.0155439572, 0.1680427130],
#          [-0.0038861287, 0.1854366532],
#          [-0.0549487103, 0.1189245136]],

#         [[0.1355803839, 1.8195602154],
#          [0.1369919053, 1.8612790359],
#          [0.1230976350, 1.7381913793]]], dtype=torch.float64)
