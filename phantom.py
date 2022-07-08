
from radon_transformation import *
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
import numpy as np
phantom = np.array(pil.open("phantom.png"))
phantom = torch.Tensor(phantom).unsqueeze(0).unsqueeze(0)
print(phantom.shape)
R = radon(256,512,"cpu")

sinogram = R(phantom)[0][0].cpu().numpy().astype(np.uint16)
print(sinogram.max())
sinogram = pil.fromarray(sinogram)#.convert("L")
sinogram.save("sinogram.png")
plt.imshow(sinogram)
plt.show()