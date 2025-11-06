from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from torch import nn

class GroupNorm2D(nn.GroupNorm):

    def forward(self, x):
        f = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        return x

class DummyConv3D(nn.Module):
    # just for ckpt compatibility
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride: Tuple[int, int, int] = (1, 1, 1), **kwargs
    ):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=(1, 1, 1), **kwargs)

    def forward(self, x):
        return self.conv(x)

# CausalConv3d
class CausalConv3d(nn.Module):
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride: Tuple[int, int, int] = (1, 1, 1), pad_mode="constant", **kwargs
    ):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.time_causal_padding = (0, 0, 0, 0, time_pad, 0)

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=(1, 1, 1), **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode='replicate')
        x = F.pad(x, self.spatial_padding, mode=self.pad_mode)
        return self.conv(x)


# CausalConv3dTrans
class CausalConv3dTrans(nn.Module):
    def __init__(
        self,
        chan_in,
        depth_to_space_block: Tuple[int, int, int],
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        pad_mode="constant",
        **kwargs
    ):
        super().__init__()
        self.frame_stride = depth_to_space_block[0]
        self.depth_to_space_block = depth_to_space_block
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        dilation = kwargs.pop("dilation", 1)

        self.pad_mode = pad_mode
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.time_causal_padding = (0, 0, 0, 0, time_pad, 0)

        dilation = (dilation, 1, 1)
        chan_out = chan_in * np.prod(depth_to_space_block)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.spatial_padding, mode=self.pad_mode)
        x = F.pad(x, self.time_causal_padding, mode='replicate')
        x = self.conv(x)
        time_factor, height_factor, width_factor = self.depth_to_space_block
        out_channels = x.shape[1] // (time_factor * height_factor * width_factor)
        x = rearrange(x, "b (c tf hf wf) t h w -> b c (t tf) (h hf) (w wf)", c=out_channels, tf=time_factor, hf=height_factor, wf=width_factor)

        if self.frame_stride > 1:
            avg = x[:, :, :self.frame_stride , :, :].mean(dim=2, keepdim=True)
            x = torch.cat((avg, x[:, :, self.frame_stride :, :, :]), dim=2)

        # x = x[:, :, (self.frame_stride - 1) :]
        return x


# ResBlock
class ResBlockX(nn.Module):
    def __init__(self, in_channels, num_groups, kernel_size, pad_mode="constant"):
        super(ResBlockX, self).__init__()
        self.group_norm = GroupNorm2D(num_groups=num_groups, num_channels=in_channels)
        self.swish = nn.SiLU()

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.pad_mode = pad_mode
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.time_causal_padding = (0, 0, 0, 0, time_pad, 0)


        self.t_causal_conv1 = DummyConv3D(in_channels, in_channels, kernel_size=kernel_size)
        self.t_causal_conv2 = DummyConv3D(in_channels, in_channels, kernel_size=kernel_size)

    def forward(self, x):
        residual = x

        out = self.group_norm(x)
        out = self.swish(out)
        out = F.pad(out, self.time_causal_padding, mode='replicate')
        out = F.pad(out, self.spatial_padding, mode=self.pad_mode)
        out = self.t_causal_conv1(out)

        out = self.group_norm(out)
        out = self.swish(out)
        out = F.pad(out, self.time_causal_padding, mode='replicate')
        out = F.pad(out, self.spatial_padding, mode=self.pad_mode)
        out = self.t_causal_conv2(out)

        out += residual
        return out


class ResBlockX2Y(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, kernel_size, pad_mode="constant"):
        super(ResBlockX2Y, self).__init__()
        self.group_norm1 = GroupNorm2D(num_groups=num_groups, num_channels=in_channels)
        self.group_norm2 = GroupNorm2D(num_groups=num_groups, num_channels=out_channels)
        self.swish = nn.SiLU()

        self.t_causal_conv1 = DummyConv3D(in_channels, out_channels, kernel_size=kernel_size)
        self.t_causal_conv2 = DummyConv3D(out_channels, out_channels, kernel_size=kernel_size)

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.pad_mode = pad_mode
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)
        self.time_causal_padding = (0, 0, 0, 0, time_pad, 0)

        if in_channels != out_channels:
            self.channel_match_conv = DummyConv3D(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.channel_match_conv = None

    def forward(self, x):
        residual = x
        out = self.group_norm1(x)
        out = self.swish(out)

        out = F.pad(out, self.time_causal_padding, mode='replicate')
        out = F.pad(out, self.spatial_padding, mode=self.pad_mode)
        out = self.t_causal_conv1(out)

        out = self.group_norm2(out)
        out = self.swish(out)
        out = F.pad(out, self.time_causal_padding, mode='replicate')
        out = F.pad(out, self.spatial_padding, mode=self.pad_mode)
        out = self.t_causal_conv2(out)

        if self.channel_match_conv is not None:
            residual = F.pad(residual, self.time_causal_padding, mode='replicate')
            residual = F.pad(residual, self.spatial_padding, mode=self.pad_mode)
            residual = self.channel_match_conv(residual)
        out += residual
        return out


# GroupNorm
class GroupNormSwishConv3d_in(nn.Module):
    def __init__(self, num_groups, num_channels_in, num_channels_out):
        super(GroupNormSwishConv3d_in, self).__init__()
        self.group_norm = GroupNorm2D(num_groups=num_groups, num_channels=num_channels_in)
        self.swish = nn.SiLU()
        self.conv = nn.Conv3d(num_channels_in, num_channels_out, kernel_size=(1, 1, 1), stride=1, padding=0)

    def forward(self, x):
        x = self.group_norm(x)
        x = self.swish(x)
        x = self.conv(x)
        return x


class GroupNormSwishConv3d_out(nn.Module):
    def __init__(self, num_groups, num_channels_in, num_channels_out, kernel_size,pad_mode="constant"):
        super(GroupNormSwishConv3d_out, self).__init__()
        self.group_norm = GroupNorm2D(num_groups=num_groups, num_channels=num_channels_in)
        self.swish = nn.SiLU()
        self.conv = CausalConv3d(num_channels_in, num_channels_out, kernel_size=kernel_size, pad_mode=pad_mode)

    def forward(self, x):
        x = self.group_norm(x)
        x = self.swish(x)
        x = self.conv(x)
        return x


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, latent_channels, eps=1e-5):
        super(AdaptiveGroupNorm, self).__init__()
        self.group_norm = GroupNorm2D(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=False)

        self.adaGN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent_channels, 2 * num_channels, bias=True))

    def forward(self, x, c):

        x = self.group_norm(x)
        c = c.mean(dim=(2, 3, 4))  # (b, c, t, h, w) ==> (b, c)
        shift, scale = self.adaGN_modulation(c).chunk(2, dim=1)
        scale += 1
        return x.mul_(scale[..., None, None, None]).add_(shift[..., None, None, None])


class DiagonalGaussianDistribution(DiagonalGaussianDistribution):
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=list(range(1, len(self.mean.shape))))
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=list(range(1, len(self.mean.shape))),
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(device=prediction.device, dtype=prediction.dtype)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv.to(dtype=real_data.dtype))
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
