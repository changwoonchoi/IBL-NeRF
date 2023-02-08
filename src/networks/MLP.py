import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionMLP(nn.Module):
	def __init__(self,D=8, W=256, input_ch=3, out_ch=3, skips=[4]):
		super().__init__()
		self.D = D
		self.W = W
		self.input_ch = input_ch
		self.skips = skips

		self.positions_linears = nn.ModuleList(
			[nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
										range(D - 1)]
		)
		self.out_linears = nn.Linear(W, out_ch)

	def forward(self, input_pts):
		h = input_pts
		# (1) position
		for i, l in enumerate(self.positions_linears):
			h = self.positions_linears[i](h)
			h = F.relu(h)
			if i in self.skips:
				h = torch.cat([input_pts, h], dim=-1)
		h = self.out_linears(h)
		return h


class PositionDirectionMLP(nn.Module):
	def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, out_ch=3, skips=[4]):
		super().__init__()
		self.D = D
		self.W = W
		self.input_ch = input_ch
		self.input_ch_views = input_ch_views
		self.skips = skips

		self.positions_linears = nn.ModuleList(
			[nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
										range(D - 1)]
		)
		self.feature_linear = nn.Linear(W, W)
		self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)] + [nn.Linear(W // 2, W // 2) for _ in range(D // 2 - 1)])

		self.final_linear = nn.Linear(W // 2, out_ch)

	def forward(self, x):
		if x.shape[-1] == self.input_ch + self.input_ch_views:
			input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
		else:
			input_pts = x
			input_views = None

		h = input_pts
		# (1) position
		for i, l in enumerate(self.positions_linears):
			h = self.positions_linears[i](h)
			h = F.relu(h)
			if i in self.skips:
				h = torch.cat([input_pts, h], dim=-1)

		# (2) position + direction
		feature = self.feature_linear(h)
		h = torch.cat([feature, input_views], dim=-1)
		for i, l in enumerate(self.views_linears):
			h = self.views_linears[i](h)
			h = F.relu(h)

		h = self.final_linear(h)

		return h
