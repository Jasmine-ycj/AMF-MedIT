from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        device = inputs.device 
        targets = targets.to(device)
        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = pt.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        
        if self.alpha is not None:
            self.alpha = self.alpha.to(device) 
            alpha = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.eval_datatype == 'imaging':
      self.model = ImagingModel(self.hparams)
    if self.hparams.eval_datatype == 'tabular':
      self.model = TabularModel(self.hparams)
    if self.hparams.eval_datatype == 'imaging_and_tabular':
      self.model = MultimodalModel(self.hparams)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    
    self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_test_topk = torchmetrics.Accuracy(task=task, average="macro", num_classes=self.hparams.num_classes)

    self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

    if self.hparams.eval_loss == 'BCE':
      self.criterion = nn.CrossEntropyLoss()
    elif self.hparams.eval_loss == 'focal':
      self.criterion = FocalLoss(alpha=self.hparams.alpha)
    
    self.best_val_score = 0

    print(self.model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat = self.model(x)

    # Needed for gradcam
    if len(y_hat.shape)==1:
      y_hat = torch.unsqueeze(y_hat, 0)

    return y_hat

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    x, y = batch 
    y_hat = self.forward(x)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_test(y_hat, y)
    self.acc_test_topk(y_hat, y)
    self.auc_test(y_hat, y)

  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_acc = self.acc_test.compute()
    test_acc_topk = self.acc_test_topk.compute()
    test_auc = self.auc_test.compute()

    self.log('test.acc', test_acc)
    self.log('test.acc_topk', test_acc_topk)
    self.log('test.auc', test_auc)

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_train(y_hat, y)
    self.auc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)

    return loss

  def training_epoch_end(self, _) -> None:
    """
    Compute training epoch metrics and check for new best values
    """
    self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
    self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_val(y_hat, y)
    self.auc_val(y_hat, y)
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)
 
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_acc_val = self.acc_val.compute()
    epoch_auc_val = self.auc_val.compute()

    self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
    self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
  
    if self.hparams.target == 'dvm':
      self.best_val_score = max(self.best_val_score, epoch_acc_val)
    else:
      self.best_val_score = max(self.best_val_score, epoch_auc_val)

    self.acc_val.reset()
    self.auc_val.reset()

  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer
    
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )
