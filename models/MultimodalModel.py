import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel

class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)

    input_features = args.embedding_dim + self.imaging_model.get_pooled_dim()
    in_dim = 256
    output_dim = args.num_classes
    # self.head = nn.Linear(input_features, output_dim)
    self.head1 = nn.Linear(input_features, in_dim)
    self.head2 = nn.Linear(in_dim, output_dim)
    self.layer_norm = nn.LayerNorm(in_dim)     # Layer Normalization
    self.relu = nn.ReLU()

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



  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x[0]: image, (batch_size, channels, h, w)
    # x[1]: table, (batch_size, feature_num)
    x_im = self.imaging_model.encoder(x[0]) # (batch_size, pooled_dim)
    if isinstance(x_im,list):
      x_im = x_im[0]
    x_tab = self.tabular_model.encoder(x[1]).squeeze() # (batch_size, embedding_dim)
    if len(x_tab.shape) == 1:
      x_tab = x_tab.unsqueeze(0)
    x = torch.cat([x_im, x_tab], dim=1) # (batch_size, pooled_dim + embedding_dim)

    # x = self.head(x)
    x = self.head1(x)
    x = self.layer_norm(x)
    x = self.relu(x)
    x = self.head2(x)
    return x