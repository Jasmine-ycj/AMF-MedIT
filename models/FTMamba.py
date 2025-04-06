"""
This code is modified based on the papers:
Mambular: A Sequential Model for Tabular Deep Learning.
@article{thielmann2024mambular,
  title={Mambular: A Sequential Model for Tabular Deep Learning},
  author={Thielmann, Anton Frederik and Kumar, Manish and Weisser, Christoph and Reuter, Arik and S{\"a}fken, Benjamin and Samiee, Soheila},
  journal={arXiv preprint arXiv:2408.06291},
  year={2024}
}
Code: https://github.com/basf/mamba-tabular/tree/master

Revisiting deep learning models for tabular data.
@inproceedings{Gorishniy2024Revisiting,
  title = {Revisiting Deep Learning Models for Tabular Data},
  booktitle = {Proceedings of the 35th International Conference on Neural Information Processing Systems},
  author = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  year = {2024},
  month = jun,
  series = {NIPS '21},
  pages = {18932--18943},
}
Code: https://github.com/yandex-research/rtdl
"""
import torch
import torch.nn as nn
# from mambular.arch_utils.mamba_arch import Mamba
from mamba_ssm import Mamba
from mambular.arch_utils.mlp_utils import MLP
from mambular.arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from mambular.configs.mambular_config import DefaultMambularConfig
from mambular.base_models.basemodel import BaseModel
from mambular.arch_utils.embedding_layer import EmbeddingLayer
from typing import List
from torch import Tensor
from .rtdl_revisiting_models import LinearEmbeddings, CategoricalEmbeddings, _CLSEmbedding

_INTERNAL_ERROR = 'Internal error'

class FTMamba(BaseModel):
    """
    Modified based on Mambular:
    - the embedding layer is modified to be consistent with FT-Transformer, 
    no need for input Parameters cat_feature_info & num_feature_info.
    - the num_classes arg & tabular head is removed.

    Parameters
    ----------
    n_cont_features: the number of continuous features.
    cat_cardinalities: List[int], the cardinalities of categorical features.
                Pass en empty list if there are no categorical features.
    n_categories: List[int], the number of distinct values for each feature.
                Pass en empty list if there are no categorical features.
    config : DefaultMambularConfig, optional
        Configuration object containing default hyperparameters for the model (default is DefaultMambularConfig()).
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: List[int],
        n_categories: List[int],
        config: DefaultMambularConfig = DefaultMambularConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.cat_cardinalities = cat_cardinalities
        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.shuffle_embeddings = self.hparams.get(
            "shuffle_embeddings", config.shuffle_embeddings
        )

        n_layers = 2
        self.mamba = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba.append(Mamba(
            d_model=self.hparams.get("d_model", config.d_model),
            d_state=self.hparams.get("d_state", config.d_state),
            expand=self.hparams.get("expand_factor", config.expand_factor),
            d_conv=self.hparams.get("d_conv", config.d_conv),   
        ))
        
        self.norm_f = LayerNorm(
                self.hparams.get("d_model", config.d_model), eps=config.layer_norm_eps
            )
        
        # >>> Feature & cls embeddings in FT-Transformer.
        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, config.d_model) if n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(n_categories, config.d_model, True)
            if cat_cardinalities
            else None
        )
        self.cls_embedding = _CLSEmbedding(config.d_model)
        # <<<

        if self.pooling_method == "cls":
            self.use_cls = True
        else:
            self.use_cls = self.hparams.get("use_cls", config.use_cls)

        if self.shuffle_embeddings:
            self.perm = torch.randperm(self.embedding_layer.seq_len)

    def split_cont_cat_features(self, x) -> List[torch.Tensor]:
        """Split continuous and categorical features from x"""
        # x: (batch_size, num_feature)
        # select categorical features: (batch_size, num_categorical_feature)    
        if self.cat_cardinalities: 
            x_cat = x[:, self.cat_cardinalities].long()  

            # select continuous features
            mask = torch.ones(x.shape[1], dtype=torch.bool)
            mask[self.cat_cardinalities] = False
            x_cont = x[:, mask]  # (batch_size, num_continuous_feature)
        else:
            x_cat = None
            x_cont = x

        return x_cat, x_cont
    
    def forward(self, x):
        x_cat, x_cont = self.split_cont_cat_features(x)

        # cls embedding
        x_embeddings: List[Tensor] = []
        if self.cls_embedding is not None:
            x_embeddings.append(self.cls_embedding(x.shape[:-1]))

        # feature embedding
        # x_embeddings.append(self.cont_embeddings(x))
        for argname, argvalue, module in [
            ('x_cont', x_cont, self.cont_embeddings),
            ('x_cat', x_cat, self.cat_embeddings),
        ]:
            if module is not None:
                x_embeddings.append(module(argvalue))
        assert x_embeddings, _INTERNAL_ERROR
        x = torch.cat(x_embeddings, dim=1)
        
        if self.shuffle_embeddings:
            x = x[:, self.perm, :]

        for layer in self.mamba:
            x = layer(x)

        if self.pooling_method == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(x, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(x, dim=1)
        elif self.pooling_method == "cls_token":
            x = x[:, -1]
        elif self.pooling_method == "last":
            x = x[:, -1]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        x = self.norm_f(x)
        
        return x