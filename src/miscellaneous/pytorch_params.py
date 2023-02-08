from nerf_models.nerf import NeRF
model = NeRF()
for name, param in model.named_parameters():
    print(name, param.data.shape)