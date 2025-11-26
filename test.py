import os
import time
import argparse
import warnings
import pickle
import gzip
import copy
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
from gat.graph_data_loader_copy import HeteroDataset
from gat.architecture import NewModel
from utils import seed_everything, create_file, save_output, sort_file, is_directory_empty, \
    load_dataset_indices
from configs import get_cfg_defaults
from raw_data_loader_copy import MultiOmicsDataset
import torch
import gc
from pytorch_lightning import Trainer
from feature_selection1 import select_feature_combined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gat.trainer import ModelTrainer

from pytorch_lightning import Trainer


def select_top_feats(x_train, y_train, feat_size, num_omics):
    subsets = []
    remaining = feat_size
    for omics_idx in range(num_omics):
        k = remaining // (num_omics - omics_idx) if (num_omics - omics_idx) !=0 else remaining
        selected = select_feature_combined(x_train[omics_idx], y_train, k)
        subsets.append(selected)
        remaining -= len(selected)
    return subsets

def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="HeteroGATomics for multiomics data integration")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()

    return args


def prepare_data(main_folder, fold_idx, multiomics, cfg):
    fold_dir = os.path.join(main_folder, f"{fold_idx + 1}")
    train_index, test_index = load_dataset_indices(fold_dir)
    multiomics_copy = copy.deepcopy(multiomics)
    train_index.sort()
    test_index.sort()
    multiomics_copy.set_train_test(train_index, test_index)
    multiomics_copy.config_components() 
    return multiomics_copy

def main():
    warnings.filterwarnings(action="ignore")
    # ---- setup configs ----
    gat_results = defaultdict(lambda: defaultdict(list))
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    for fold_idx in range(5):
        seed_everything(cfg.SOLVER.SEED)
        main_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        raw_file_paths = [(os.path.join(main_folder, f"{omics}.csv"), omics) for omics in cfg.DATASET.OMICS]
        raw_label_path = os.path.join(main_folder, f"ClinicalMatrix.csv")

        # ---- setup multiomics dataset ----
        multiomics = MultiOmicsDataset(
            dataset_name=cfg.DATASET.NAME,
            raw_file_paths=raw_file_paths,
            raw_label_path=raw_label_path,
            num_omics=len(cfg.DATASET.OMICS),
            num_classes=cfg.DATASET.NUM_CLASSES,
            sparsity_rates=cfg.DATASET.FEATURE_SPARSITY_RATES
        )
        
        print(f"==> Loading data from fold {fold_idx + 1}...")
        fold_multiomics = prepare_data(main_folder, fold_idx, multiomics, cfg)
        for feat_size in cfg.GAT.FINAL_FEAT_SIZES:
            multiomics_deepcopy1 = copy.deepcopy(fold_multiomics)
            multiomics_deepcopy = copy.deepcopy(fold_multiomics)
            x_train = [multiomics_deepcopy.get(i).train_data.values for i in range(multiomics_deepcopy.num_omics)]
            y_train = multiomics_deepcopy.get(0).train_label.values.ravel()
    
            final_feat_subset = select_top_feats(x_train, y_train, feat_size, multiomics_deepcopy.num_omics)
            multiomics_deepcopy.reduce_dimensionality(final_feat_subset)
            hetero_data = HeteroDataset(multiomics_deepcopy1,multiomics_deepcopy, sparsity_rate=cfg.DATASET.PATIENT_SPARSITY_RATES,
                                            tune_hyperparameters=cfg.SOLVER.TUNE_HYPER,
                                            seed=cfg.SOLVER.SEED)
            hetero_data.create_hetero_data()
            # ---- setup model ----

            print("\n   ==> Building model...")
            new_model = NewModel(
                    dataset=hetero_data,
                    dataset1=multiomics_deepcopy1,
                    num_modalities=multiomics_deepcopy1.num_omics,
                    num_classes=multiomics_deepcopy1.num_classes,
                    gat_num_layers=cfg.GAT.NUM_LAYERS,
                    gat_num_heads=cfg.GAT.NUM_HEADS,
                    weight=cfg.GAT.WEIGHT,
                    gat_hidden_dim=cfg.GAT.HIDDEN_DIM,
                    gat_dropout_rate=cfg.GAT.DROPOUT_RATE,
                    gat_lr_pretrain=cfg.GAT.LR_PRETRAIN,
                    gat_lr=cfg.GAT.LR,
                    gat_wd=cfg.GAT.WD,
                    vcdn_lr=cfg.VCDN.LR,
                    vcdn_wd=cfg.VCDN.WD,
                    trans_dropout_rate=cfg.GAT.TRANS_DROPOUT_RATE,
                    tune_hyperparameters=False
            )
            save_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(
                dataset_name=cfg.DATASET.NAME,
                fold_idx=fold_idx +1,
                feat_size=feat_size,
                seed=cfg.SOLVER.SEED
            )
            checkpoint_path = os.path.join(cfg.RESULT.SAVE_MODEL_DIR, save_model_name)
            checkpoint = torch.load(checkpoint_path)
            model = ModelTrainer.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                trial=None,
                dataset=new_model.dataset,
                dataset1=new_model.dataset1,
                num_modalities=new_model.num_modalities,
                num_classes=new_model.num_classes,
                weight=new_model.weight,
                unimodal_model=new_model.unimodal_model,
                unimodal_model1=new_model.unimodal_model1,
                multimodal_decoder=new_model.multimodal_decoder,
                loss_fn=new_model.loss_function,
                train_multimodal_decoder=True,
                gat_wd=new_model.gat_wd,
                gat_lr=new_model.gat_lr,
                vcdn_lr=new_model.vcdn_lr,
                vcdn_wd=new_model.vcdn_wd,
                trans_dropout_rate=new_model.trans_dropout_rate,
                contrastive_loss_fn=new_model.contrastive_loss,
                tune_hyperparameters=new_model.tune_hyperparameters,
                strict=True 
            )
            model.eval()
            model.requires_grad_(False)
            
            
            print("\n   ==> Testing model...")
            trainer = Trainer(
                accelerator="gpu",
                devices=1,
                logger=False,
                enable_checkpointing=False
            )
 
            with torch.no_grad():
                trainer.test(model)
            
            if hasattr(model, 'optimizers'):
                for opt in model.optimizers():
                    opt.state.clear() 

            for metric_key, metric_value in model.get_log_metrics().items():
                if metric_key.startswith("test_"):
                    gat_results[feat_size][metric_key.replace("test_", "")].append(metric_value[0])
            print(f"\n==> Showing results...")
            for feat_size, metrics in gat_results.items():
                for metric_name, values in metrics.items():
                    average = np.mean(values)
                    print(values)
                    std = np.std(values)
                    print(f"    - {metric_name}: {average:.3f}±{std:.3f}")
 
            
    for feat_size, metrics in gat_results.items():
        for metric_name, values in metrics.items():
            average = np.mean(values)
            std = np.std(values)
            print(f"    - {metric_name}: {average:.3f}±{std:.3f}")



if __name__ == '__main__':
    main()
