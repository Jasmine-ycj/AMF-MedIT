from pynndescent.optimal_transport import total_cost
from typing import Tuple
from models.Evaluator import Evaluator
from models.FusionMultiModalModel import FusionMultimodalModel
import torch

class FusionEvaluator(Evaluator):
    def __init__(self, hparams):
        super(FusionEvaluator, self).__init__(hparams)
        if self.hparams.eval_datatype == 'imaging_and_tabular':
            self.model = FusionMultimodalModel(self.hparams)

        self.lambda_density = self.hparams.density_loss_ratio
        if self.hparams.fusion_loss is not None:
            self.fusion_loss = self.hparams.fusion_loss
        else:
            self.fusion_loss = 'density-leakage'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a prediction from a data point
        """
        y_hat, x_im, x_tab = self.model(x)

        # Needed for gradcam
        if len(y_hat.shape) == 1:
            y_hat = torch.unsqueeze(y_hat, 0)
        return y_hat, x_im, x_tab


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch

        y_hat, x_im, x_tab = self.forward(x)

        ce_loss = self.criterion(y_hat, y)
        if self.fusion_loss == 'density-leakage':
            fusion_loss = self.model.fusion_model.compute_density_leakage_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'density':
            fusion_loss = self.model.fusion_model.compute_density_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'leakage':
            fusion_loss = self.model.fusion_model.compute_leakage_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'only_ce':
            fusion_loss = 0.0
        else:
            raise Exception(f'Unknown fusion loss {self.fusion_loss}')
        if self.fusion_loss == 'only_ce':
            total_loss = ce_loss
        else:
            total_loss = ce_loss + self.lambda_density * fusion_loss

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)

        self.log('eval.train.total_loss', total_loss, on_epoch=True, on_step=False)
        self.log('eval.train.bce_loss', ce_loss, on_epoch=True, on_step=False)
        self.log('eval.train.fusion_loss', fusion_loss, on_epoch=True, on_step=False)

        return total_loss


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Validate and log
        """
        x, y = batch

        y_hat, x_im, x_tab = self.forward(x)
        ce_loss = self.criterion(y_hat, y)
        if self.fusion_loss == 'density-leakage':
            fusion_loss = self.model.fusion_model.compute_density_leakage_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'density':
            fusion_loss = self.model.fusion_model.compute_density_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'leakage':
            fusion_loss = self.model.fusion_model.compute_leakage_loss(x_im=x_im, x_tab=x_tab)
        elif self.fusion_loss == 'only_ce':
            fusion_loss = 0.0
        else:
            raise Exception(f'Unknown fusion loss {self.fusion_loss}')
        if self.fusion_loss == 'only_ce':
            total_loss = ce_loss
        else:
            total_loss = ce_loss + self.lambda_density * fusion_loss

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

        self.log('eval.val.ce_loss', ce_loss, on_epoch=True, on_step=False)
        self.log('eval.val.fusion_loss', fusion_loss, on_epoch=True, on_step=False)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Runs test step
        """
        x, y = batch
        y_hat, _, _ = self.forward(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_test(y_hat, y)
        self.acc_test_topk(y_hat, y)
        self.auc_test(y_hat, y)