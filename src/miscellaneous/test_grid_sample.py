import torch
import torch.nn.functional as F

def test_grid_sample():
	A = torch.tensor([
		[1, 1, 1],
		[2, 2, 2],
		[3, 3, 3]
	]).float()
	A = A[None, None, ...]
	uv = torch.tensor([
		[-1, -1],
		[0.5, 0],
		[1, 1],
	]).float()
	uv = uv[None, None, ...]
	#print(A)
	#print(A.shape)
	val = F.grid_sample(A, uv, align_corners=True)
	print(val)
	#print(A.unsqueeze(1).shape)

if __name__ == "__main__":
	test_grid_sample()
