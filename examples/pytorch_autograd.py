
# Reference code

import torch
from torch.autograd import Variable

a = Variable(torch.arange(1,13).view(3,4), requires_grad=True)
b = Variable(torch.arange(2,14).view(3,4), requires_grad=True)
c = Variable(torch.arange(3,12).view(3,3), requires_grad=True)
x = Variable(torch.arange(4,16).view(4,3), requires_grad=True)
y = Variable(torch.arange(5,17).view(4,3), requires_grad=True)


def forwardNeuron(a,b,c,x,y):
  ax = a @ x
  by = b @ y
  axpby = ax + by
  axpbypc = axpby + c

  return axpbypc

s = forwardNeuron(a,b,c,x,y).mean()

print(s)
Variable containing:
 599
[torch.FloatTensor of size 1]

s.backward()

print(a.grad)
# Variable containing:
#  1.6667  2.6667  3.6667  4.6667
#  1.6667  2.6667  3.6667  4.6667
#  1.6667  2.6667  3.6667  4.6667
# [torch.FloatTensor of size 3x4]

print(b.grad)
# Variable containing:
#  2  3  4  5
#  2  3  4  5
#  2  3  4  5
# [torch.FloatTensor of size 3x4]

print(c.grad)
# Variable containing:
#  0.1111  0.1111  0.1111
#  0.1111  0.1111  0.1111
#  0.1111  0.1111  0.1111
# [torch.FloatTensor of size 3x3]

print(x.grad)
# Variable containing:
#  1.6667  1.6667  1.6667
#  2.0000  2.0000  2.0000
#  2.3333  2.3333  2.3333
#  2.6667  2.6667  2.6667
# [torch.FloatTensor of size 4x3]

print(y.grad)
# Variable containing:
#  2.0000  2.0000  2.0000
#  2.3333  2.3333  2.3333
#  2.6667  2.6667  2.6667
#  3.0000  3.0000  3.0000
# [torch.FloatTensor of size 4x3]