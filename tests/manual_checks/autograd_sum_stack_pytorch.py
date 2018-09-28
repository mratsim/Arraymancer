
# Reference code

import torch
from torch.autograd import Variable

a = Variable(torch.arange(1,13).view(3,4), requires_grad=True)
b = Variable(torch.arange(2,14).view(3,4), requires_grad=True)
c = Variable(torch.arange(3,15).view(3,4), requires_grad=True)
d = Variable(torch.arange(4,16).view(3,4), requires_grad=True)

def foo(a,b,c,d):
  return torch.stack([a, a+b, c-d], dim=0).sum()

s = foo(a,b,c,d)

print(s)
# tensor(234.)

s.backward()

print(a.grad)
# tensor([[ 2.,  2.,  2.,  2.],
#         [ 2.,  2.,  2.,  2.],
#         [ 2.,  2.,  2.,  2.]])

print(b.grad)
# tensor([[ 1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.]])

print(c.grad)
# tensor([[ 1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.]])

print(d.grad)
# tensor([[-1., -1., -1., -1.],
#         [-1., -1., -1., -1.],
#         [-1., -1., -1., -1.]])
