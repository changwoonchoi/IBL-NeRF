import matplotlib.pyplot as plt
import torch

def visualize_images_vertical(images, use_colorbar=True,\
    clim_val=None, horizontal=False, title=None):
    plt.figure()

    if title is not None:
        plt.suptitle(title)

    n = len(images)
    print("Image number", n)
    for i in range(n):
        image = images[i]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if horizontal:
            plt.subplot(1, n, i+1)
        else:
            plt.subplot(n, 1, i+1)

        plt.imshow(image)
        if isinstance(clim_val, list):
            plt.clim(0, clim_val[i])
        elif isinstance(clim_val, float):
            plt.clim(0, clim_val)
        if use_colorbar:
            plt.colorbar(label='color')

def visualize_8_channel_images(images, use_colorbar=True, clim_val = -1):
    plt.figure()
    indices = [0, 3, 6, 7, 8, 5, 2, 1, 4]
    for i in range(len(images)):
        image = images[i]
        plt.subplot(3, 3, indices[i] + 1)
        plt.imshow(image)
        if clim_val > 0:
            plt.clim(0, clim_val)
        if use_colorbar:
            plt.colorbar(label='color')