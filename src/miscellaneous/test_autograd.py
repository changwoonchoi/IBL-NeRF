import torch


def test_autograd():
	a = torch.tensor([[2., 3.], [4., 5.]])
	b = torch.tensor([[1., 1.], [1., 1.]])
	L = nn.Linear(2, 1)
	L.weight.data.fill_(1)
	L.bias.data.fill_(0)

	a.requires_grad = True
	Q = L(a*b)**2
	print("Q", Q)
	Q.backward(torch.ones_like(Q))
	print(a.grad)

	a = torch.tensor([[2., 5.], [3., 4.]])
	a.requires_grad = True
	Q = L(a*b) ** 2
	print("Q", Q)
	Q.backward(torch.ones_like(Q))
	print(a.grad)
	#print(b.grad)