import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create3DsobelFilter():
    num_1 = np.array([[1., 2., 1.],
                      [2., 4., 2.],
                      [1., 2., 1.]])
    num_2 = np.zeros((3,3))
    num_3 = -num_1

    sobelFilter = np.zeros((3,1,3,3,3))

    sobelFilter[0,0,0,:,:] = num_1
    sobelFilter[0,0,1,:,:] = num_2
    sobelFilter[0,0,2,:,:] = num_3

    sobelFilter[1,0,:,0,:] = num_1
    sobelFilter[1,0,:,1,:] = num_2
    sobelFilter[1,0,:,2,:] = num_3

    sobelFilter[2,0,:,:,0] = num_1
    sobelFilter[2,0,:,:,1] = num_2
    sobelFilter[2,0,:,:,2] = num_3

    return torch.from_numpy(sobelFilter).cpu()


def sobelLayer(x):

    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x, dtype=torch.double)
    # Input expected: (N, C, H, W)
    if x.dim() == 4:
        x = x.unsqueeze(2)   # → (N, C, D=1, H, W)

    pad = nn.ConstantPad3d((1,1,1,1,1,1), 0)
    kernel = create3DsobelFilter()

    x = pad(x)
    sobel = F.conv3d(x, kernel, padding=0) / 4

    n, c, d, h, w = sobel.shape
    magnitude = torch.norm(sobel, p=2, dim=1, keepdim=True) / c * 3
    magnitude = torch.tanh(magnitude) * 2 - 1
    magnitude = magnitude.detach().cpu().numpy()
    magnitude = np.squeeze(magnitude).astype(np.float32)
    magnitude = np.expand_dims(magnitude, axis=0)
    magnitude = np.expand_dims(magnitude, axis=-1)
    return magnitude
