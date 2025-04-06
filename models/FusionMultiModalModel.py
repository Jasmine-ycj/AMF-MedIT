import torch
import torch.nn as nn
from collections import OrderedDict
from models.FusionModule import FusionModule
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel

class FusionMultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(FusionMultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)

    self.fusion_model = FusionModule(args.conf, self.imaging_model.get_pooled_dim(), args.embedding_dim, args.classify_dim, f'cuda:{args.gpus[0]}')

    self.head = nn.Linear(args.classify_dim, args.num_classes)

    # load checkpoints for classifier
    if args.checkpoint and args.test:
      checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpus[0]}')
      state_dict = checkpoint['state_dict']

      state_dict_classifier = {}
      prex = 'model.'
      for k in state_dict.keys():
        if not 'imaging_model' in k and not 'tabular_model' in k:
          state_dict_classifier[k[len(prex):]] = state_dict[k]
      self.load_state_dict(state_dict_classifier, strict=False)


  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x[0]: image, (batch_size, channels, h, w)
    # x[1]: table, (batch_size, feature_num)
    x_im = self.imaging_model.encoder(x[0]) # (batch_size, pooled_dim)
    if isinstance(x_im,list):
      x_im = x_im[0]
    x_tab = self.tabular_model.encoder(x[1]).squeeze() # (batch_size, embedding_dim)
    if len(x_tab.shape) == 1:
      x_tab = x_tab.unsqueeze(0)

    x_fusion, x_im, x_tab = self.fusion_model(x_im, x_tab)

    x = self.head(x_fusion)

    return x, x_im, x_tab

class FusionMultimodalModelForCAM(FusionMultimodalModel):
  def __init__(self, args):
    super(FusionMultimodalModelForCAM, self).__init__(args)

  def forward(self, x):
    x, _, _ = super().forward(x)
    return x
