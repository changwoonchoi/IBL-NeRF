import torch
import torch.nn.functional as F
import numpy as np

bias = 1e-5


def fresnel_schlick_roughness(cosTheta, F0, roughness):
	cosTheta = cosTheta[..., None]
	roughness = roughness[..., None]
	F1 = torch.maximum(1.0 - roughness, F0) - F0
	return F0 + F1 * torch.pow(torch.clip(1.0 - cosTheta, 0.0, 1.0), 5.0)


class Microfacet:
	"""As described in:
		Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
	"""
	def __init__(self, default_rough=0.3, lambert_only=False, f0=0.04):
		self.default_rough = default_rough
		self.lambert_only = lambert_only
		self.f0 = f0

	def __call__(self, pts2l, pts2c, normal, albedo=None, rough=None):
		"""All in the world coordinates.
		Too low roughness is OK in the forward pass, but may be numerically
		unstable in the backward pass
		pts2l: NxLx3
		pts2c: Nx3
		normal: Nx3
		albedo: Nx3
		rough: Nx1
		"""
		# print("-----------BRDF------------")
		# print(pts2l.shape, 'pts2l')
		# print(pts2c.shape, 'pts2c')
		# print(normal.shape, 'normal')
		# print(albedo.shape, 'albedo')
		# print(rough.shape, 'rough')

		if albedo is None:
			albedo = torch.ones((pts2c.shape[0], 3), dtype=torch.float32)
		if rough is None:
			rough = self.default_rough * torch.ones(
				(pts2c.shape[0], 1), dtype=torch.float32)
		# Normalize directions and normals
		pts2l = F.normalize(pts2l, dim=-1)
		pts2c = F.normalize(pts2c, dim=-1)
		normal = F.normalize(normal, dim=-1)
		# Glossy
		h = pts2l + pts2c[:, None, :] # NxLx3
		h = F.normalize(h, dim=2)
		metallic = 1 - rough
		f0 = self.f0 * (1-metallic) + albedo * metallic  #Nx3
		f = self._get_f(pts2l, h, f0) # NxLx3
		alpha = rough ** 2
		l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)
		v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)
		l_dot_n = torch.clip(l_dot_n, 0, 1)
		v_dot_n = torch.clip(v_dot_n, 0, 1)
		v_dot_n = v_dot_n[..., None]

		d = self._get_d(h, normal, alpha=alpha) # NxL
		g = self._get_g(v_dot_n, l_dot_n, alpha=alpha) # NxL

		denom = 4 * l_dot_n * v_dot_n
		g = g[...,None]
		d = d[..., None]
		denom = denom[...,None]

		if torch.any(d.isnan()).item() > 0:
			print("d_nan")
		if torch.any(g.isnan()).item() > 0:
			print("g_nan")
		if torch.any(f.isnan()).item() > 0:
			print("f_nan")
		if torch.any(denom.isnan()).item() > 0:
			print("denom_nan")

		brdf_glossy = (f * g * d / (denom + bias)) # NxLx3

		lambert = albedo / np.pi # Nx3
		brdf_diffuse = (1 - f) * lambert[:, None, :]
		brdf_diffuse *= (1-metallic[..., None])

		# Mix two shaders
		# brdf_glossy *= l_dot_n[..., None]
		# brdf_diffuse *= l_dot_n[..., None]

		#brdf = brdf_glossy + brdf_diffuse # TODO: energy conservation?
		return brdf_glossy, brdf_diffuse, l_dot_n[..., None] # NxLx3

	def get_decomposed_FGDs(self, pts2l, pts2c, normal, albedo=None, rough=None):
		"""All in the world coordinates.
		Too low roughness is OK in the forward pass, but may be numerically
		unstable in the backward pass
		pts2l: NxLx3
		pts2c: Nx3
		normal: Nx3
		albedo: Nx3
		rough: Nx1
		"""
		# print("-----------BRDF------------")
		# print(pts2l.shape, 'pts2l')
		# print(pts2c.shape, 'pts2c')
		# print(normal.shape, 'normal')
		# print(albedo.shape, 'albedo')
		# print(rough.shape, 'rough')

		if albedo is None:
			albedo = torch.ones((pts2c.shape[0], 3), dtype=torch.float32)
		if rough is None:
			rough = self.default_rough * torch.ones(
				(pts2c.shape[0], 1), dtype=torch.float32)
		# Normalize directions and normals
		pts2l = F.normalize(pts2l, dim=-1)
		pts2c = F.normalize(pts2c, dim=-1)
		normal = F.normalize(normal, dim=-1)
		# Glossy
		h = pts2l + pts2c[:, None, :]  # NxLx3
		h = F.normalize(h, dim=2)
		metallic = 1 - rough
		f0 = self.f0 * (1 - metallic) + albedo * metallic  # Nx3
		f = self._get_f(pts2l, h, f0)  # NxLx3
		alpha = rough ** 2
		l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)
		v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)
		l_dot_n = torch.clip(l_dot_n, 0, 1)
		v_dot_n = torch.clip(v_dot_n, 0, 1)
		v_dot_n = v_dot_n[..., None]

		d = self._get_d(h, normal, alpha=alpha)  # NxL
		g = self._get_g(v_dot_n, l_dot_n, alpha=alpha)  # NxL

		denom = 4 * l_dot_n * v_dot_n
		g = g[..., None]
		d = d[..., None]
		denom = denom[..., None]

		if torch.any(d.isnan()).item() > 0:
			print("d_nan")
		if torch.any(g.isnan()).item() > 0:
			print("g_nan")
		if torch.any(f.isnan()).item() > 0:
			print("f_nan")
		if torch.any(denom.isnan()).item() > 0:
			print("denom_nan")

		brdf_glossy = (f * g * d / (denom + bias))  # NxLx3

		lambert = albedo / np.pi  # Nx3
		brdf_diffuse = (1 - f) * lambert[:, None, :]
		brdf_diffuse *= (1 - metallic[..., None])

		# Mix two shaders
		# brdf_glossy *= l_dot_n[..., None]
		# brdf_diffuse *= l_dot_n[..., None]

		# brdf = brdf_glossy + brdf_diffuse # TODO: energy conservation?
		return f, g, d, h  # NxLx3

	@staticmethod
	def _get_g_ggx(n_dot_v, r):
		k = r * r / 2
		denom = n_dot_v * (1-k) + k
		return n_dot_v / (denom + bias)

	@staticmethod
	def _get_g(n_dot_v, n_dot_l, alpha=0.1):
		"""Geometric function (GGX).
		"""
		ggx2 = Microfacet._get_g_ggx(n_dot_v, alpha)
		ggx1 = Microfacet._get_g_ggx(n_dot_l, alpha)
		return ggx1 * ggx2

		# n_dot_v = torch.einsum('ij,ij->i', n, v)
		# cos_theta = torch.einsum('ijk,ik->ij', m, v)
		# denom = cos_theta_v[:, None]
		# div = torch.nan_to_num((cos_theta / (denom + bias)))
		# chi = torch.where(div > 0, 1., 0.)
		# cos_theta_v_sq = torch.square(cos_theta_v)
		# cos_theta_v_sq = torch.clip(cos_theta_v_sq, 0., 1.)
		# denom = cos_theta_v_sq
		# tan_theta_v_sq = torch.nan_to_num((1 - cos_theta_v_sq) / (denom + bias))
		# tan_theta_v_sq = torch.clip(tan_theta_v_sq, 0., np.inf)
		# denom = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
		# g = torch.nan_to_num(chi * 2 / (denom + bias))
		# return g # (n_pts, n_lights)

	@staticmethod
	def _get_d(m, n, alpha=0.1):
		"""Microfacet distribution (GGX).
		"""
		cos_theta_m = torch.einsum('ijk,ik->ij', m, n)
		cos_theta_m = torch.clip(cos_theta_m, 0, 1)
		cos_theta_m_sq = torch.square(cos_theta_m)

		a2 = alpha ** 2
		num = a2
		denom = np.pi * torch.square(cos_theta_m_sq * (a2 - 1) + 1)
		return num / (denom + bias)

		# chi = torch.where(cos_theta_m > 0, 1., 0.)
		# cos_theta_m_sq = torch.square(cos_theta_m)
		# denom = cos_theta_m_sq
		# tan_theta_m_sq = torch.nan_to_num((1 - cos_theta_m_sq) / (denom + bias))
		# denom = np.pi * torch.square(cos_theta_m_sq) * torch.square(
		# 	alpha ** 2 + tan_theta_m_sq)
		# d = torch.nan_to_num(alpha ** 2 * chi / (denom + bias))
		# return d # (n_pts, n_lights)

	def _get_f(self, l, m, f0):
		"""Fresnel (Schlick's approximation).
		"""
		cos_theta = torch.einsum('ijk,ijk->ij', l, m)
		cos_theta = torch.clip(cos_theta, 0, 1)
		cos_theta = torch.stack([cos_theta] * 3, dim=-1)
		f0 = f0[:, None,:]

		f = f0 + (1 - f0) * (1 - cos_theta) ** 5
		return f # (n_pts, n_lights, 3)
