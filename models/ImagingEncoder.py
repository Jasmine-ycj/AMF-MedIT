import torch
import torch.nn as nn
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from functools import partial


class ImagingEncoder(nn.Module):
  """
  Setting Imaging Encoders: Resnet50, Resnet18, Vit, Mamba.
  Also supports providing a checkpoint with trained weights to be loaded.
  """
  def __init__(self, args) -> None:
    super(ImagingEncoder, self).__init__()
    self.hparams = args

    # Build architecture
    self.encoder = self.build_encoder()

  def build_encoder(self) -> nn.modules:
    if self.hparams.model in ['resnet18', 'resnet50']:
      encoder_imaging = torchvision_ssl_encoder(self.hparams.model)
      if self.hparams.input_channel != 3:
        original_conv = encoder_imaging.conv1
        new_conv = nn.Conv2d(in_channels=self.hparams.input_channel,
                             out_channels=original_conv.out_channels,
                             kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride,
                             padding=original_conv.padding,
                             bias=original_conv.bias is not None)
        encoder_imaging.conv1 = new_conv
    else:
       raise Exception(f'Unknown imaging encoder {self.hparams.model}')
    return encoder_imaging

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Passes input through encoder.
    """
    x = self.encoder(x)
    return x