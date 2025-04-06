import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from models.ImagingEncoder import ImagingEncoder


class ImagingModel(nn.Module):
  """
  Evaluation model for imaging trained with ResNet encoder.
  """
  def __init__(self, args) -> None:
    super(ImagingModel, self).__init__()
    # encoder name
    encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
    self.encoder_name = encoder_name_dict[args['loss']]
    if args.test:
      self.encoder_name = 'model.imaging_model.encoder.'
    self.bolt_encoder = True

    # encoder
    self.encoder = ImagingEncoder(args=args)
    # pooled_dim
    if args.model =='resnet50':
      self.pooled_dim = 2048
    elif args.model =='resnet18':
      self.pooled_dim = 512
    else:
       raise Exception(f'Unknown imaging encoder {self.hparams.model}')
    # load checkpoint
    if args.checkpoint:
      # Load weights
      checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpus[0]}')
      original_args = checkpoint['hyper_parameters']
      state_dict = checkpoint['state_dict']

      # Remove prefix and fc layers
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith(self.encoder_name) and not 'projection_head' in k and not 'prototypes' in k:
          state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]
      log = self.encoder.load_state_dict(state_dict_encoder, strict=True)
      assert len(log.missing_keys) == 0

      # Freeze if needed
      if args.finetune_strategy == 'frozen':
        for _, param in self.encoder.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        assert len(parameters)==0
    # classifier
    self.classifier = nn.Linear(self.pooled_dim, args.num_classes)

  def get_pooled_dim(self):
    return self.pooled_dim
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.bolt_encoder:
      x = self.encoder(x)[0]
    else:
      x = self.encoder(x).squeeze()
    x = self.classifier(x)
    return x