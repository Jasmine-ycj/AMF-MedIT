from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from models.TabularEncoder import TabularEncoder

class TabularModel(nn.Module):
  """
  Evaluation model for tabular trained with MLP backbone.
  """
  def __init__(self, args):
    super(TabularModel, self).__init__()

    self.encoder = TabularEncoder(args)
    self.encoder_name = 'encoder_tabular.'
    if args.test:
      self.encoder_name = 'model.tabular_model.encoder.'

    if args.checkpoint:
      loaded_chkpt = torch.load(args.checkpoint, map_location=f'cuda:{args.gpus[0]}')
      original_args = loaded_chkpt['hyper_parameters']
      state_dict = loaded_chkpt['state_dict']
      self.input_size = original_args['input_size']

      # Split weights
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith(self.encoder_name):
          # if args.model == 'resnet18':
            # state_dict_encoder['encoder.'+k[len(self.encoder_name):]] = state_dict[k]
          # else:
          state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]
      _ = self.encoder.load_state_dict(state_dict_encoder, strict=False)

      # Freeze if needed
      if args.finetune_strategy == 'frozen':
        for _, param in self.encoder.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        assert len(parameters)==0
    
    self.classifier = nn.Linear(args.embedding_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.classifier(x)
    return x
    