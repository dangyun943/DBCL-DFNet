from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleList
from torch.optim.lr_scheduler import StepLR
from raw_data_loader_copy import MultiOmicsDataset
from gat.Transformer import AdaptiveDimTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
from gat.base_models import HeteroGNN, VCDN,DynamicMultiheadAttentionFusion
from utils import calculate_performance_metrics
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC,
    Specificity, NegativePredictiveValue, StatScores
)
from torch_geometric.data import HeteroData
from gat.loss import ContrastiveLoss
from gat.graph_data_loader_copy import HeteroDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer(pl.LightningModule):
    def __init__(
            self,
            dataset: HeteroDataset,
            num_modalities: int,
            num_classes: int,
            unimodal_model: List[HeteroGNN],
            unimodal_model1: List[AdaptiveDimTransformer],
            loss_fn: CrossEntropyLoss,
            contrastive_loss_fn: ContrastiveLoss,
            multimodal_decoder: Optional[DynamicMultiheadAttentionFusion] = None,
            train_multimodal_decoder: bool = True,
            gat_lr: float = 1e-3,
            gat_wd: float = 1e-3,
            vcdn_lr: float = 5e-2,
            vcdn_wd: float = 1e-3,
            weight:List[float]=[0.5,0.5],
            tune_hyperparameters: float = False,
            trial=None,
            contrastive_weight: float = 0.5,  # 新增：对比损失权重
            temperature: float = 0.5
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trial=trial
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.unimodal_model = ModuleList(unimodal_model)
        self.unimodal_model1 = ModuleList(unimodal_model1)
        self.multimodal_decoder = multimodal_decoder
        self.train_multimodal_decoder = train_multimodal_decoder
        self.loss_fn = loss_fn
        self.gat_lr = gat_lr
        self.gat_wd = gat_wd
        self.vcdn_lr = vcdn_lr
        self.vcdn_wd = vcdn_wd
        self.weight = weight
        self.tune_hyperparameters = tune_hyperparameters
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.contrastive_loss = contrastive_loss_fn
        # self.save_hyperparameters()
        self.log_metrics = {}

        # activate manual optimization
        self.automatic_optimization = False

    def get_log_metrics(self):
        return self.log_metrics


    def configure_optimizers(self):
        optimizers = []
        lr_schedulers = []
        for modality in range(self.num_modalities):
            optimizer = torch.optim.Adam(list(self.unimodal_model[modality].parameters()),
                                         lr=self.gat_lr,
                                         weight_decay=self.gat_wd)

            scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
            optimizers.append(optimizer)
            lr_schedulers.append(scheduler)
        
        for modality in range(self.num_modalities):
            optimizer = torch.optim.Adam(list(self.unimodal_model1[modality].parameters()),
                                         lr=self.gat_lr,
                                         weight_decay=self.gat_wd)

            scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
            optimizers.append(optimizer)
            lr_schedulers.append(scheduler)

        if self.multimodal_decoder is not None:
            optimizer = torch.optim.Adam(list(self.multimodal_decoder.parameters()),
                                         lr=self.vcdn_lr,
                                         weight_decay=self.vcdn_wd)

            scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
            optimizers.append(optimizer)
            lr_schedulers.append(scheduler)

        return optimizers, lr_schedulers

    def forward(self, data: HeteroDataset, multimodal: bool = False, phase: str = "train", stage: str = "only") -> Union[Tensor, List[Tensor]]:
        output = []
        output1 = []
        features = []  # 新增：存储HeteroGNN特征
        features1 = []  # 新增：存储Transformer特征
        for modality in range(self.num_modalities):
            # 修改：返回特征和输出
            feat, out = self.unimodal_model[modality](
                data[modality].x_dict, data[modality].edge_index_dict,
                data[modality].edge_attr_dict, phase=phase, stage=stage
            )
            features.append(feat)
            output.append(out)
            
        for modality in range(self.num_modalities):
            # 修改：返回特征和输出
            feat1, out1 = self.unimodal_model1[modality](
                data[modality]['patient'].x1, stage=stage
            )
            features1.append(feat1)
            output1.append(out1)
            
        if not multimodal:
            return output, output1, features, features1  # 修改：返回特征
        
        if self.multimodal_decoder is not None:
            return self.multimodal_decoder(features, features1)

        raise TypeError("multimodal_decoder must be defined for multiomics datasets.")

    def training_step(self, train_batch, batch_idx: int): 
        is_single_modality = self.num_modalities == 1
        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        # 修改：获取特征
        outputs, outputs1, features, features1 = self.forward(
            train_batch, multimodal=False, phase="train", stage="only"
        )
        
        # 新增：存储对比损失值
        total_contrastive_loss = 0.0
        
        for modality in range(self.num_modalities):
            if is_single_modality:
                opt = optimizers[0]
            else:
                opt = optimizers[modality]
            opt.zero_grad()
            
            mask = train_batch[modality]['patient'].train_mask
            # 原始分类损失
            loss = self.loss_fn(outputs[modality][mask], train_batch[modality]['patient'].y[mask])
            
            # 新增：对比损失
            contrast_loss = self.contrastive_loss(
                features[modality][mask], 
                features1[modality][mask].detach()
            )
            # 组合损失
            total_loss = self.weight[0]*loss + self.contrastive_weight * contrast_loss
            total_contrastive_loss += contrast_loss.item()
            
            self.log_metrics.setdefault(f"train_modality_loss_{modality + 1}", []).append(loss.detach().item())
            self.log_metrics.setdefault(f"train_contrast_loss_{modality + 1}", []).append(contrast_loss.item())
            
            self.manual_backward(total_loss)
            opt.step()
            
            if is_single_modality:
                lr_schedulers[0].step()
            else:
                lr_schedulers[modality].step()
        
        # 优化多组学模态模型 (unimodal_model1)
        for modality in range(self.num_modalities):
            if is_single_modality:
                opt = optimizers[1]
            else:
                opt = optimizers[modality + self.num_modalities] 
            
            # 原始分类损失
            omics_mask = train_batch[modality]['patient'].train_mask
            omics_loss = self.loss_fn(
                outputs1[modality][omics_mask],
                train_batch[modality]['patient'].y[omics_mask]
            )
            
            # 新增：对比损失
            contrast_loss = self.contrastive_loss(
                features[modality][omics_mask].detach(), 
                features1[modality][omics_mask]
            )
            
            # 组合损失
            total_loss = self.weight[1]*omics_loss + self.contrastive_weight * contrast_loss
            total_contrastive_loss += contrast_loss.item()
            
            self.log_metrics.setdefault(f"train_omics_loss_{modality}", []).append(omics_loss.item())
            self.log_metrics.setdefault(f"train_omics_contrast_{modality}", []).append(contrast_loss.item())
            
            self.manual_backward(total_loss)
            opt.step()
            
            if is_single_modality:
                lr_schedulers[1].step()
            else:
                lr_schedulers[modality + self.num_modalities].step()

        # 记录总对比损失
        self.log_metrics.setdefault("train_total_contrast_loss", []).append(total_contrastive_loss)
        
        if self.train_multimodal_decoder and self.multimodal_decoder is not None:
            optimizers[-1].zero_grad()
            mask = train_batch[0]['patient'].train_mask
            output = self.forward(train_batch, multimodal=True,phase="train",stage="multi")
            multi_loss = self.loss_fn(output[mask], train_batch[0]['patient'].y[mask])
            
            preds_multi = torch.argmax(output[mask], dim=1)
            acc_multi = (preds_multi == train_batch[0]['patient'].y[mask]).float().mean()
            self.log_metrics.setdefault(f"train_multi_loss", []).append(multi_loss.detach().item())
            self.manual_backward(multi_loss)
            optimizers[-1].step()
            lr_schedulers[-1].step()

    def on_validation_epoch_end(self):
        """每个验证周期结束时触发的剪枝逻辑"""
        if self.tune_hyperparameters:
            if self.tune_hyperparameters and hasattr(self, 'trial') and self.trial:
                # 安全获取聚合后的验证准确率
                current_acc = self.trainer.callback_metrics["val_acc"]
        
                if current_acc is not None:
                    # 向Optuna报告指标并触发剪枝
                    self.trial.report(current_acc.item(), self.current_epoch)
                    if self.trial.should_prune():
                        print(f"Trial {self.trial.number} pruned at epoch {self.current_epoch}")
                        self.trainer.should_stop = True  # 立即停止训练\

    def validation_step(self, validation_batch, batch_idx: int):
        if self.tune_hyperparameters:
        # 获取模型输出
            if self.multimodal_decoder is not None:
                output= self.forward(validation_batch, multimodal=True, phase="test", stage="multi") 
            else:
                output, output1, features, features1  = self.forward(validation_batch, multimodal=False, phase="test", stage="only")
                output = output[0]  # 假设我们主要关注第一个输出

        # 获取验证集掩码
            mask = validation_batch[0]['patient'].val_mask
        
        # 提取模型预测
            pred_val_data = output[mask]
        
        # 获取真实标签
            actual_labels = validation_batch[0]['patient'].y[mask]
        
        # 计算损失
            loss = self.loss_fn(pred_val_data, actual_labels)  # 使用与训练相同的损失函数
        
        # 记录验证损失
            self.log("val_loss", loss, batch_size=len(mask), on_epoch=True, prog_bar=True)
        
        # 转换为CPU用于指标计算
            pred_val_data = pred_val_data.detach().cpu()
            actual_output = actual_labels.detach().cpu()
        
        # 计算预测概率
            if self.num_classes == 2:
                final_output = F.softmax(pred_val_data, dim=1).numpy()
            else:
                final_output = pred_val_data.numpy()

        # 二分类情况
            if self.num_classes == 2:
                auc = roc_auc_score(actual_output, final_output[:, 1])
                acc = accuracy_score(actual_output, final_output.argmax(1))
                sensitivity, specificity, ppv, npv = calculate_performance_metrics(actual_output,
                                                                             final_output.argmax(1))

            # 使用 self.log 记录指标（自动聚合）
                self.log("val_acc", acc, batch_size=len(mask), on_epoch=True, prog_bar=True)
                self.log("val_auc", auc, batch_size=len(mask), on_epoch=True)
                self.log("val_recall", sensitivity, batch_size=len(mask), on_epoch=True)
                self.log("val_specificity", specificity, batch_size=len(mask), on_epoch=True)
                self.log("val_ppv", ppv, batch_size=len(mask), on_epoch=True)
                self.log("val_npv", npv, batch_size=len(mask), on_epoch=True)
        # 多分类情况
            else:
                acc = accuracy_score(actual_output, final_output.argmax(1))
                f1_macro = f1_score(actual_output, final_output.argmax(1), average="macro")
                f1_micro = f1_score(actual_output, final_output.argmax(1), average="micro")
                f1_weighted = f1_score(actual_output, final_output.argmax(1), average="weighted")
                precision = precision_score(actual_output, final_output.argmax(1), average="weighted")
                recall = recall_score(actual_output, final_output.argmax(1), average="weighted")

                # 使用 self.log 记录指标（自动聚合）
                self.log("val_acc", acc, batch_size=len(mask), on_epoch=True, prog_bar=True)
                self.log("val_f1_macro", f1_macro, batch_size=len(mask), on_epoch=True)
                self.log("val_f1_micro", f1_micro, batch_size=len(mask), on_epoch=True)
                self.log("val_f1_weighted", f1_weighted, batch_size=len(mask), on_epoch=True)
                self.log("val_precision", precision, batch_size=len(mask), on_epoch=True)
                self.log("val_recall", recall, batch_size=len(mask), on_epoch=True)

    def test_step(self, test_batch, batch_idx: int):
        if self.multimodal_decoder is not None:
            output = self.forward(test_batch, multimodal=True,phase="test",stage="mutli")
        else:
            output,output1 = self.forward(test_batch,multimodal=False,phase="test",stage="only")
            output =output[0]

        mask = test_batch[0]['patient'].test_mask
        pred_test_data = output[mask]
        torch.cuda.synchronize()

        final_output = F.softmax(pred_test_data, dim=1).detach().to('cpu')
        actual_output = test_batch[0]['patient'].y[mask].detach().to('cpu')
        predicted_labels = torch.argmax(final_output, dim=1)  # 形状变为 (batch_size,)

        from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
        if self.num_classes == 2:
            # 二分类任务
            auc = auroc(final_output[:, 1], actual_output, task="binary")
            acc = accuracy(predicted_labels, actual_output, task="binary")
            precision_val = precision(predicted_labels, actual_output, task="binary")  # PPV
            recall_val = recall(predicted_labels, actual_output, task="binary")  # Sensitivity
            f1 = f1_score(predicted_labels, actual_output, task="binary")
            specificity = Specificity(task="binary")(predicted_labels, actual_output)
            npv = NegativePredictiveValue(task="binary")(predicted_labels, actual_output)

            self.log_metrics.setdefault("test_AUROC", []).append(auc.item())
            self.log_metrics.setdefault("test_Accuracy", []).append(acc.item())
            self.log_metrics.setdefault("test_Precision", []).append(precision_val.item())
            self.log_metrics.setdefault("test_Recall", []).append(recall_val.item())
            self.log_metrics.setdefault("test_F1", []).append(f1.item())
            self.log_metrics.setdefault("test_Specificity", []).append(specificity.item())
            self.log_metrics.setdefault("test_NPV", []).append(npv.item())
        else:
            # 多分类任务
            acc = accuracy(predicted_labels, actual_output, task="multiclass", num_classes=self.num_classes)
            precision_val = precision(predicted_labels, actual_output, task="multiclass", average="weighted",
                                      num_classes=self.num_classes)
            recall_val = recall(predicted_labels, actual_output, task="multiclass", average="weighted",
                                num_classes=self.num_classes)
            weighted_f1 = f1_score(predicted_labels, actual_output, task="multiclass", average="weighted",
                          num_classes=self.num_classes)
            macro_f1 = f1_score(predicted_labels, actual_output, task="multiclass", average="macro",
                          num_classes=self.num_classes)
            micro_f1 = f1_score(predicted_labels, actual_output, task="multiclass", average="micro",
                          num_classes=self.num_classes)
            

            self.log_metrics.setdefault("test_Accuracy", []).append(acc.item())
            self.log_metrics.setdefault("test_Precision", []).append(precision_val.item())
            self.log_metrics.setdefault("test_Recall", []).append(recall_val.item())
            self.log_metrics.setdefault("test_weighted_F1", []).append(weighted_f1.item())
            self.log_metrics.setdefault("test_macro_F1", []).append(macro_f1.item())
            self.log_metrics.setdefault("test_micro_F1", []).append(micro_f1.item())
            
    def _custom_data_loader(self):
        return self.dataset

    def train_dataloader(self):
        return self._custom_data_loader()

    def val_dataloader(self):
        return self._custom_data_loader()

    def test_dataloader(self):
        return self._custom_data_loader()

    def __str__(self) -> str:
        r"""Returns a string representation of the multiomics trainer object.

        Returns:
            str: The string representation of the multiomics trainer object.
        """
        model_str = ["\nModel info:\n", "   Unimodal model:\n"]

        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.unimodal_model[modality]}")

        if self.multimodal_decoder is not None:
            model_str.append("\n\n  Multimodal decoder:\n")
            model_str.append(f"    {self.multimodal_decoder}")

        return "".join(model_str)

    
