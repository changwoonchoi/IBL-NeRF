from nerf_models.microfacet import *
import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def visualize_brdf():
	microfacet = Microfacet(f0=0.04)
	N = 200
	result = np.zeros((N, 3))
	thetas = np.linspace(-np.pi/2, np.pi/2, num=N)

	result[:,0] = np.sin(thetas)
	result[:,1] = np.cos(thetas)

	pts2l = torch.Tensor(result)
	pts2c = torch.Tensor([0, 1, 0])
	normal = torch.Tensor([0, 1, 0])
	albedo = torch.Tensor([1, 1, 1])
	pts2l = pts2l[None, ...]
	pts2c = pts2c[None, ...]
	normal = normal[None, ...]
	albedo = albedo[None, ...]
	l_dot_n = torch.sum(pts2l * normal, dim=-1, keepdim=True)
	M = 10
	roughnesses = np.linspace(0.2, 0.8, num=M)

	for r in range(M):
		roughness_value = roughnesses[r]
		roughness_value = np.sqrt(roughness_value)
		roughness = torch.Tensor([roughness_value])

		roughness = roughness[None, ...]
		brdf_d, brdf_s, l_dot_n = microfacet(pts2l, pts2c, normal, albedo, roughness)
		f, g, d, h = microfacet.get_decomposed_FGDs(pts2l, pts2c, normal, albedo, roughness)

		brdf = d * torch.sum(h * normal, dim=-1, keepdim=True)
		# brdf *= l_dot_n
		brdf = brdf[0]
		# print(brdf[:, 0])

		plt.plot(thetas, brdf[:,0])
	plt.show()


def visualize_gaussian():
	microfacet = Microfacet(f0=0.04)
	N = 200
	result = np.zeros((N, 3))
	thetas = np.linspace(-np.pi/2 + 0.001, np.pi/2 - 0.001, num=N)

	result[:,0] = np.sin(thetas)
	result[:,1] = np.cos(thetas)

	pts2l = torch.Tensor(result)
	pts2c = torch.Tensor([0, 1, 0])
	normal = torch.Tensor([0, 1, 0])
	albedo = torch.Tensor([1, 1, 1])
	pts2l = pts2l[None, ...]
	pts2c = pts2c[None, ...]
	normal = normal[None, ...]
	albedo = albedo[None, ...]
	l_dot_n = torch.sum(pts2l * normal, dim=-1, keepdim=True)
	M = 10
	roughnesses = np.linspace(0.2, 0.8, num=M)

	for r in range(M):
		roughness_value = roughnesses[r]
		roughness_value_sq = roughness_value * roughness_value

		h = pts2l + pts2c[:, None, :]  # NxLx3
		h = F.normalize(h, dim=2)
		h_dot_n = torch.sum(h * normal, dim=-1, keepdim=True)
		h_dot_n_sq = h_dot_n * h_dot_n
		h_dot_n_sq_2 = h_dot_n_sq * h_dot_n_sq
		h_dot_n_sin_sq = (1-h_dot_n_sq)
		h_dot_n_tan_sq = h_dot_n_sin_sq / h_dot_n

		pdf = 1 / (np.pi * roughness_value_sq * h_dot_n_sq_2) * torch.exp(-(h_dot_n_tan_sq / roughness_value_sq))
		pdf = pdf * h_dot_n
		pdf = pdf[0]
		#print(pdf)
		#brdf = pdf
		# brdf *= l_dot_n
		#brdf = brdf[0]
		# print(brdf[:, 0])

		plt.plot(thetas, pdf)
	plt.show()



def get_d(h, n, alpha, method='ggx'):
	if method == 'ggx':
		h_dot_n = np.sum(h * n, axis=-1)
		a2 = alpha * alpha
		t = 1.0 + (a2 - 1.0) * h_dot_n * h_dot_n
		return a2 / (np.pi * t * t)
	else:
		h_dot_n = np.sum(h * n, axis=-1)
		alphaSq = alpha * alpha
		cosThetaSq = h_dot_n * h_dot_n
		tanThetaSq = (1.0 - cosThetaSq) / cosThetaSq
		cosThetaQu = cosThetaSq * cosThetaSq
		return 1 / np.pi * np.exp(-tanThetaSq / alphaSq) / (alphaSq * cosThetaQu)


def plot_gaussian_kernel(N=100, roughness=0.2, epsilon=0.01):
	o = np.array([0, 0, 1])
	n = np.array([0, 0, 1])
	nx = np.linspace(-1, 1, N) * epsilon * N
	alpha = roughness

	i = np.stack([nx, np.zeros_like(nx), np.ones_like(nx)], axis=-1)
	h = (i + o)
	h /= np.linalg.norm(h, axis=-1, keepdims=True)
	kernel = 1 / (2 * np.pi * alpha * alpha) * np.exp(-0.5 * (nx * nx / (alpha * alpha)))

	plt.plot(nx, kernel)


def plot_kernel(N=100, roughness=0.2, epsilon=0.01):
	o = np.array([0, 0, 1])
	n = np.array([0, 0, 1])
	nx = np.linspace(-1, 1, N) * epsilon * N

	i = np.stack([nx, np.zeros_like(nx), np.ones_like(nx)], axis=-1)
	distance_sq = np.sum(i * i, axis=-1)
	i = i / np.linalg.norm(i, axis=-1, keepdims=True)
	h = (i + o)
	h /= np.linalg.norm(h, axis=-1, keepdims=True)
	h_dot_n = np.sum(h * n, axis=-1)
	h_dot_i = np.sum(h * i, axis=-1)
	i_dot_n = np.sum(i * n, axis=-1)
	d = get_d(h, n, roughness)
	h_pdf = d * h_dot_n
	pdf = h_pdf / (4 * h_dot_i)
	pdf_A = pdf * (i_dot_n / distance_sq)
	kernel = pdf_A / pdf_A.sum()
	# plt.figure()
	plt.plot(nx, kernel)
	# plt.show()


def plot_kernel_2(N=7, roughness=0.2, fov_x=45):
	camera_angle_x = fov_x / 180.0 * np.pi
	width = 640
	height = 360

	focal_length = .5 * width / np.tan(0.5 * camera_angle_x)
	# total_width = 2 * np.tan(0.5 * camera_angle_x)
	# epsilon = total_width / (N - 1)

	o = np.array([0, 0, 1])
	n = np.array([0, 0, 1])
	mid_n = N // 2
	nx = np.linspace(-(N-1)/2, (N-1)/2, N)
	ny = np.linspace(-(N-1)/2, (N-1)/2, N)
	xv, yv = np.meshgrid(nx, ny)
	i = np.stack([xv, yv, np.ones_like(xv) * focal_length], axis=-1)
	distance_sq = np.sum(i * i, axis=-1)
	i = i / np.linalg.norm(i, axis=-1, keepdims=True)
	h = (i + o)
	h /= np.linalg.norm(h, axis=-1, keepdims=True)
	h_dot_n = np.sum(h * n, axis=-1)
	h_dot_i = np.sum(h * i, axis=-1)
	i_dot_n = np.sum(i * n, axis=-1)
	alpha = roughness * roughness
	d = get_d(h, n, alpha)
	h_pdf = d * h_dot_n
	pdf = h_pdf / (4 * h_dot_i)
	pdf_A = pdf * (i_dot_n / distance_sq)
	kernel = pdf_A / pdf_A.sum()
	kernel = kernel[mid_n]
	plt.plot(nx, kernel, label= "%.2f" % (roughness))
	# print(kernel)

def visualize_kernel(N=21, roughness=0.2, epsilon=0.01, focal_length=1, visualize=True):
	o = np.array([0, 0, 1])
	n = np.array([0, 0, 1])
	mid_n = N // 2
	nx = np.linspace(-1,1,N) * epsilon * N
	ny = np.linspace(-1,1,N) * epsilon * N
	xv, yv = np.meshgrid(nx, ny)
	i = np.stack([xv, yv, np.ones_like(xv) * focal_length], axis=-1)
	distance_sq = np.sum(i * i, axis=-1)
	i = i / np.linalg.norm(i, axis=-1, keepdims=True)
	h = (i + o)
	h /= np.linalg.norm(h, axis=-1, keepdims=True)
	h_dot_n = np.sum(h * n, axis=-1)
	h_dot_i = np.sum(h * i, axis=-1)
	i_dot_n = np.sum(i * n, axis=-1)
	alpha = roughness * roughness
	d = get_d(h, n, alpha)
	h_pdf = d * h_dot_n
	pdf = h_pdf / (4 * h_dot_i)
	pdf_A = pdf * (i_dot_n / distance_sq)
	kernel = pdf_A / pdf_A.sum()
	kernel = kernel[mid_n]
	plt.plot(nx, kernel, label= "%.2f" % (roughness))
	#plt.imshow(kernel)
	#return kernel

def gkern(l=5, size=5, sig=1.):
	"""\
	creates gaussian kernel with side length `l` and a sigma of `sig`
	"""
	mid_n = l // 2
	ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l) / size
	gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
	kernel = np.outer(gauss, gauss)
	kernel = kernel / np.sum(kernel)
	kernel = kernel[mid_n]
	plt.plot(ax, kernel)

# visualize_brdf()
# visualize_gaussian()
for i in range(10):
	plt.figure()
	k = visualize_kernel(N=10, roughness=(i + 1) * 0.1)
	# print(k)
# for i in range(2, 10, 1):
# 	gkern(l=101, size=20, sig=(i+1) * 0.1)
# plt.figure()
# roughnesses = [0.1]
# for i in range(0, 9, 1):
# 	plot_kernel_2(N=101, roughness=roughness=(i+1) * 0.1)
plt.figure()
#for i in range(0, 9, 1):
#	plot_kernel_2(N=101, roughness=(i+1) * 0.1)

# for i in range(10):
# 	plot_gaussian_kernel(roughness=(i+1) * 0.1)
plt.xlabel("pixel position")
plt.yticks([])
plt.legend(title="roughness")
plt.show()
# visualize_kernel(N=5)
