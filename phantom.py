
from radon_transformation import *
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
import numpy as np
import glob

from decoder import *

d = Decoder(z_channels = 256,resolution=256, in_channels=3, out_ch = 1, ch = 128, ch_mult=[1,1,2,2,4],num_res_blocks = 2,attn_resolutions=[16])
d = d.eval()
a = torch.ones((16,256,16,16))
b = d(a)
print(b.shape)
'''
path = sorted(glob.glob("dynamic_phantom/*"))[:-1]

sinogram = torch.zeros((1,360,512))

for i in range(360):
    phantom = np.array(pil.open("dynamic_phantom/phantom_"+str(i)+".png"))
    phantom = torch.Tensor(phantom).unsqueeze(0).unsqueeze(0)
    thetas = torch.linspace(0, np.pi-(np.pi/360), 360).unsqueeze(0)
    R = radon(1,512,"cpu")
    print(i)
    projection = R(phantom,thetas[:,i:i+1])
    sinogram[:,i:i+1] = projection
sinogram = pil.fromarray(sinogram[0].detach().cpu().numpy().astype(np.uint16))#.convert("L")
sinogram.save("sinogram.png")
plt.imshow(sinogram)
plt.show()
'''

'''
phantom = np.array(pil.open("phantom.png"))
print(phantom.max())
phantom = torch.Tensor(phantom).unsqueeze(0).unsqueeze(0)/255
phantom = phantom.repeat(2,1,1,1)
thetas = torch.linspace(0, np.pi-(np.pi/360), 360).unsqueeze(0).repeat(2,1)-1
thetas[1,:180] = thetas[0,180:360]
thetas[1,180:360] = thetas[0,0:180]
print(thetas[1])

print(phantom.shape)
R = radon(360,512,"cpu")


sinogram = R(phantom,thetas)
#print(sinogram.max())
#sinogram = pil.fromarray(sinogram)#.convert("L")
#sinogram.save("sinogram.png")
plt.imshow(sinogram[0][0].cpu().numpy().astype(np.uint16))
plt.show()
plt.imshow(sinogram[1][0].cpu().numpy().astype(np.uint16))
plt.show()
'''