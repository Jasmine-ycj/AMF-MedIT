from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from lightly.models.modules import SimCLRProjectionHead

class FusionModule(nn.Module):
    """
    Fusion Module: Compress or amplify modal features for better fusion
    """

    def __init__(
        self,
        conf: float,
        img_dim: int,
        tab_dim: int,
        out_dim: int,
        device: str
    ) -> None:
        """
        Args:
            conf: confidence ratio of imaging modal and tabular modal, conf = conf_img / conf_tab
            img_dim: the length of the imaging feature (2048 for ResNet50, 512 for ResNet18)
            tab_dim: the length of the tabular feature (256 for MLP/ResNet, 64 for FTT/Mamba)
            out_dim: the length of the adjusted feature
            device: device to use
        """
        super().__init__()
        if conf >= 0:
            self.L_tab = int(out_dim / (1 + conf))
            self.L_img = out_dim - self.L_tab
        else:
            raise Exception(f'negative confidence: {conf} !')

        self.weights_img = torch.zeros(1, out_dim)
        self.weights_img[0, :self.L_img] = 1

        self.weights_tab = torch.zeros(1, out_dim)
        self.weights_tab[0, self.L_img:] = 1

        self.img_projector = SimCLRProjectionHead(img_dim, img_dim, out_dim)
        self.tab_projector = SimCLRProjectionHead(tab_dim, tab_dim, out_dim)

        if device is not None:
            self.weights_img = self.weights_img.to(device)
            self.weights_tab = self.weights_tab.to(device)
            self.L_tab = torch.tensor(self.L_img, dtype=torch.float, device=device)
            self.L_img = torch.tensor(self.L_tab, dtype=torch.float, device=device)


    def forward(self, x_im:Tensor, x_tab: Tensor) -> tuple[Any, Any, Any]:
        """Do the forward pass."""
        x_im = self.img_projector(x_im)
        x_tab = self.tab_projector(x_tab)
        x_fusion = x_im * self.weights_img + x_tab * self.weights_tab
        return x_fusion, x_im, x_tab

    def compute_density_loss(self, x_im, x_tab):
        density_img = torch.sum(torch.abs(x_im * self.weights_img), dim=1, keepdim=True) / self.L_img
        density_tab = torch.sum(torch.abs(x_tab * self.weights_tab), dim=1, keepdim=True) / self.L_tab
        delta = density_img - density_tab
        loss_density = torch.abs(delta).mean()

        return 5 * loss_density

    def compute_leakage_loss(self, x_im, x_tab):
        info_img = torch.sum(torch.abs(x_im * self.weights_tab), dim=1, keepdim=True) / self.L_tab
        info_tab = torch.sum(torch.abs(x_tab * self.weights_img), dim=1, keepdim=True) / self.L_img
        info_sum = (info_img + info_tab) / 2
        loss_leakage = torch.abs(info_sum).mean()

        return 5 * loss_leakage

    def compute_density_leakage_loss(self, x_im, x_tab):
        density_img = torch.sum(torch.abs(x_im * self.weights_img), dim=1, keepdim=True) / self.L_img
        density_tab = torch.sum(torch.abs(x_tab * self.weights_tab), dim=1, keepdim=True) / self.L_tab
        delta = density_img - density_tab
        loss_density = torch.abs(delta).mean()

        info_img = torch.sum(torch.abs(x_im * self.weights_tab), dim=1, keepdim=True) / self.L_tab
        info_tab = torch.sum(torch.abs(x_tab * self.weights_img), dim=1, keepdim=True) / self.L_img
        info_sum = (info_img + info_tab) / 2
        loss_leakage = torch.abs(info_sum).mean()

        return 5 * (loss_density + loss_leakage)