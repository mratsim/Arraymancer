import torch
import torch.nn as nn

batch_size = 3
features = 4
hidden_size = 2 # weights and bias have 3x2 = 6 size

x = torch.tensor([[ 0.1,  0.2,  0.3,  0.4],
                  [-0.1, -0.2, -0.3, -0.4],
                  [ 0.5,  0.6,  0.7,  0.8]])

hidden = torch.tensor([
  [ -1.0, -1.0],
  [ -1.0, -1.0],
  [ -1.0, -1.0]])

w_input = torch.tensor([
  [0.9, 0.8, 0.7, 0.6],
  [0.8, 0.7, 0.6, 0.5],
  [0.7, 0.6, 0.5, 0.4],
  [0.6, 0.5, 0.4, 0.3],
  [0.5, 0.4, 0.3, 0.2],
  [0.4, 0.3, 0.2, 0.1]])

w_recur = torch.tensor([
  [-0.3, -0.1],
  [-0.2,  0.0],
  [-0.3, -0.1],
  [-0.2,  0.0],
  [-0.3, -0.1],
  [-0.2,  0.0],
])

b_input = torch.tensor([
  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
])

b_recur = torch.tensor([
  [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
])

test_gru = nn.GRUCell(4, 2)

test_gru.weight_ih.data = w_input
test_gru.weight_hh.data = w_recur
test_gru.bias_ih.data   = b_input
test_gru.bias_hh.data   = b_recur

print(test_gru(x, hidden))

# tensor([[-0.5317, -0.4753],
#         [-0.3930, -0.3210],
#         [-0.7325, -0.6430]])
