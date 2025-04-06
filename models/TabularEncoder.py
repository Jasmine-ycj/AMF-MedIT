from typing import Dict
from collections import OrderedDict
from models.rtdl_revisiting_models import ModifiedFTTransformer, ResNet
from models.FTMamba import FTMamba

import torch
import torch.nn as nn


class MLP(nn.Module):
  """
  Multiple-Layer Perception for tabular encoder.
  Original TabularEncoder code. 
  Main contrastive model used in SCARF. Consists of an encoder that takes the input and 
  creates an embedding of size {args.embedding_dim}.
  """
  def __init__(self, args):
    super(MLP, self).__init__()
    self.args = args
    self.encoder = self.build_encoder(args)
    self.encoder.apply(self.init_weights)
  
  def build_encoder(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(args.input_size, args['embedding_dim'])]
    for _ in range(args['encoder_num_layers']-1):
      modules.extend([nn.BatchNorm1d(args['embedding_dim']), nn.ReLU(), nn.Linear(args['embedding_dim'], args['embedding_dim'])])
    return nn.Sequential(*modules)
  
  def build_encoder_no_bn(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(args.input_size, args['embedding_dim'])]
    for _ in range(args['encoder_num_layers']-1):
      modules.extend([nn.ReLU(), nn.Linear(args['embedding_dim'], args['embedding_dim'])])
    return nn.Sequential(*modules)

  def build_encoder_bn_old(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(args.input_size, args.embedding_dim)]
    for _ in range(args.encoder_num_layers-1):
      modules.extend([nn.ReLU(), nn.BatchNorm1d(args.embedding_dim), nn.Linear(args.embedding_dim, args.embedding_dim)])
    return nn.Sequential(*modules)

  def init_weights(self, m: nn.Module, init_gain = 0.02) -> None:
    """
    Initializes weights according to desired strategy
    """
    if isinstance(m, nn.Linear):
      if self.args.init_strat == 'normal':
        nn.init.normal_(m.weight.data, 0, 0.001)
      elif self.args.init_strat == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=init_gain)
      elif self.args.init_strat == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif self.args.init_strat == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=init_gain)
      if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

  def forward(self, x):
      # 逐层传递输入，通过每一层
    x = self.encoder(x)
    return x


class TabularEncoder(nn.Module):
  """
  New TabularEncode.
  Supports selecting different encoders: MLP, transformer.
  Also supports providing a checkpoint with trained weights to be loaded.
  """
  def __init__(self, args) -> None:
    super(TabularEncoder, self).__init__()
    self.args = args
    self.input_size = args.input_size
    self.encoder = self.build_encoder(args)

  def build_encoder(self, args: Dict) -> nn.Module:
    if args.tabular_encoder == 'MLP':
      return MLP(self.args)
    elif args.tabular_encoder == 'FTTransformer':
      return ModifiedFTTransformer(
                n_cont_features=args.n_cont_features,
                cat_cardinalities=args.cat_cardinalities,
                n_categories=args.n_categories,
                d_out=args.embedding_dim,
                n_blocks=2,
                d_block=args.embedding_dim,
                attention_n_heads=8,
                attention_dropout=0.2,
                ffn_d_hidden=None,
                ffn_d_hidden_multiplier=4 / 3,
                ffn_dropout=0.1,
                residual_dropout=0.0,
            )
    elif args.tabular_encoder == 'ResNet':
      return ResNet(
        d_in=args.input_size, 
        d_out = None,  # 不要输出头
        n_blocks=2,
        d_block=args.embedding_dim,
        d_hidden=args.embedding_dim,
        d_hidden_multiplier=None,
        dropout1=0.25,
        dropout2=0.0
        )
    elif args.tabular_encoder == 'FT-Mamba':
      return FTMamba(
                n_cont_features=args.n_cont_features,
                cat_cardinalities=args.cat_cardinalities,
                n_categories=args.n_categories)
    else:
       raise Exception(f'Unknown tabular encoder {args.tabular_encoder}')
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Passes input through encoder and projector. 
    Output is ready for loss calculation.
    """
    x = self.encoder(x)
    return x