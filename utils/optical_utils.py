from PIL import Image
import torch.nn.functional as F
import numpy as np
from torchvision.utils import flow_to_image
import torch
from torchvision import transforms


DEVICE = 'cuda'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def flow_to_image_torch(flow):
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    img = np.transpose(flow_im.numpy(), [1, 2, 0])
    # print(img.shape)
    return img

def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def read_image_totorch(path):
    img = Image.open(path).convert("RGB")
    # print(type(decoded_img))
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img


def read_mask_totorch(path):
    img = Image.open(path).convert("L")
    # print(type(decoded_img))
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img



def gaussian_kernel1d(
        kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
        kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x

