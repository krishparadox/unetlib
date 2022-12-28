# Unetlib
Unet is a library implementing UNETs with latest optimisations and SOTA method implementation. Unets have been an important part of medical computer vision systems for quite some time now. Modern techniques like stable diffusion use the UNET architecture as well.

# Usage

## Installation
`pip install git+https://github.com/krishparadox/unetlib.git@main`

## Example
```python
import torch
from unetlib import BuildUnet

unet = BuildUnet(
    dim = 64,
    nested_unet_depths = (7, 4, 2, 1)
)

img = torch.randn(1, 3, 256, 256)
out = unet(img)
```
