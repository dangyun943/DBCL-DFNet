from typing import List, Optional

import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
from  gat-mamba-trans.Transformer import AdaptiveDimTransformer, CompressionLayer
from gat-mamba-trans.graph_data_loader import HeteroDataset
from raw_data_loader import MultiOmicsDataset
from  gat-mamba-trans.trainer import ModelTrainer
from  gat-mamba-trans.loss import ContrastiveLoss
from  gat-mamba-trans.base_models import HeteroGNN, VCDN,DynamicMultiheadAttentionFusion
from  gat-mamba-trans.HeteroGPSModel  import HeteroGPSModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewModel:
    def __init__(self,
                 dataset: HeteroDataset,
                 dataset1: MultiOmicsDataset,
                 num_modalities: int,
                 num_classes: int,
                 gat_num_layers: int,
                 gat_num_heads: int,
                 gat_hidden_dim: List[int],
                 gat_dropout_rate: float,
                 gat_lr_pretrain: float,
                 gat_lr: float,
                 gat_wd: float,
                 vcdn_lr: float,
                 trans_dropout_rate:float,
                 vcdn_wd: float,
                 contrastive_weight: float = 0.5, 
                 temperature: float = 0.5, 
                 weight: List[float]= [0.5,0.5],      
                 tune_hyperparameters: bool = False,
                 trial=None
                 ) -> None:
        self.dataset = dataset
        self.trial = trial
        self.dataset1 = dataset1
        self.unimodal_model: List[HeteroGPSModel] = []
        self.unimodal_model1: List[AdaptiveDimTransformer] = []
        self.multimodal_decoder: Optional[VCDN] = None
        self.loss_function = CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature) 
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.gat_num_layers = gat_num_layers
        self.gat_num_heads = gat_num_heads
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_dropout_rate = gat_dropout_rate
        self.gat_lr_pretrain = gat_lr_pretrain
        self.gat_lr = gat_lr
        self.gat_wd = gat_wd
        self.vcdn_lr = vcdn_lr
        self.weight = weight
        self.vcdn_wd = vcdn_wd
        self.trans_dropout_rate=trans_dropout_rate
        self.vcdn_hidden_dim = pow(self.num_classes, self.num_modalities)
        self.tune_hyperparameters = tune_hyperparameters
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_model()
    
    def _create_model(self) -> None:
        for modality in range(self.num_modalities):
            self.unimodal_model.append(
                HeteroGPSModel(
                    hidden_channels=self.gat_hidden_dim,
                    out_channels=self.num_classes,
                    num_layers=self.gat_num_layers,
                    num_heads=self.gat_num_heads,
                    dropout=self.gat_dropout_rate,
                )
            )

            self.unimodal_model1.append(
                AdaptiveDimTransformer(input_dim=self.dataset1.data[modality].num_features,
                 embed_dim=self.gat_hidden_dim[-1],out_channels=self.num_classes,dropout=self.trans_dropout_rate,
                )
            )
        # Initialize lazy modules
        with torch.no_grad():
            for modality in range(self.num_modalities):
                self.unimodal_model[modality](self.dataset.get(modality).x_dict,
                                              self.dataset.get(modality).edge_index_dict,
                                              self.dataset.get(modality).edge_attr_dict)
                self.unimodal_model1[modality](self.dataset1.get(modality).combined_data,stage="only")


        if self.num_modalities >= 1:
            self.multimodal_decoder = DynamicMultiheadAttentionFusion(
                num_modalities=self.num_modalities, num_classes=self.num_classes, hidden_dim=self.vcdn_hidden_dim,input_dim=self.gat_hidden_dim[-1]
            )
            self.multimodal_decoder

    def get_model(self, pretrain: bool = False) -> ModelTrainer:
        if pretrain:
            multimodal_model = None
            train_multimodal_decoder = False
            gat_lr = self.gat_lr_pretrain
        else:
            multimodal_model = self.multimodal_decoder
            train_multimodal_decoder = True
            gat_lr = self.gat_lr

        model = ModelTrainer(
            trial = self.trial,
            dataset=self.dataset,
            num_modalities=self.num_modalities,
            num_classes=self.num_classes,
            unimodal_model=self.unimodal_model,
            unimodal_model1=self.unimodal_model1,
            multimodal_decoder=multimodal_model,
            train_multimodal_decoder=train_multimodal_decoder,
            loss_fn=self.loss_function,
            gat_lr=gat_lr,
            gat_wd=self.gat_wd,
            vcdn_lr=self.vcdn_lr,
            vcdn_wd=self.vcdn_wd,
            tune_hyperparameters=self.tune_hyperparameters,
            contrastive_weight=self.contrastive_weight,
            weight=self.weight,
            temperature=self.temperature,
            contrastive_loss_fn=self.contrastive_loss
        )

        return model

    def __str__(self) -> str:
        r"""Returns a string representation of the model object.

        Returns:
            str: The string representation of the model object.
        """
        return self.get_model().__str__()
