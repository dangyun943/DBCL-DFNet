import glob
import os
import re
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
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint
from feature_selection1 import select_feature_combined
from gat.trainer import ModelTrainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_top_feats(x_train, y_train, feat_size, num_omics):
    """基于F3分数的特征选择"""
    subsets = []
    remaining = feat_size
    for omics_idx in range(num_omics):
        # 修正：使用 num_omics 而不是未定义的 omimambacs_idx
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
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
  
    # if is_directory_empty(cfg.RESULT.SAVE_RICH_DATA_DIR):
    #     raise Exception("Perform feature selection first")
    # ---- setup folders and paths ----
    if not os.path.exists(cfg.RESULT.OUTPUT_DIR) and cfg.RESULT.SAVE_RESULT:
        os.makedirs(cfg.RESULT.OUTPUT_DIR)
    if not os.path.exists(cfg.RESULT.SAVE_MODEL_DIR) and cfg.RESULT.SAVE_MODEL:
        os.makedirs(cfg.RESULT.SAVE_MODEL_DIR)

    main_folder = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    raw_file_paths = [(os.path.join(main_folder, f"{omics}.csv"), omics) for omics in cfg.DATASET.OMICS]
    raw_label_path = os.path.join(main_folder, f"ClinicalMatrix.csv")
    if cfg.RESULT.SAVE_RESULT:
        output_gat_file = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}.csv')
        sorted_output_gat_file = os.path.join(cfg.RESULT.OUTPUT_DIR,
                                              f'HeteroGATomics_gat_{cfg.DATASET.NAME}_sorted.csv')
        output_gat_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR, f'HeteroGATomics_gat_{cfg.DATASET.NAME}_time.csv')
        sorted_output_gat_file_time = os.path.join(cfg.RESULT.OUTPUT_DIR,
                                                   f'HeteroGATomics_gat_{cfg.DATASET.NAME}_time_sorted.csv')
    create_file(file_dir=output_gat_file, header=cfg.RESULT.FILE_HEADER_GAT)
    create_file(file_dir=output_gat_file_time, header=cfg.RESULT.FILE_HEADER_GAT_TIME)

    # ---- setup multiomics dataset ----
    multiomics = MultiOmicsDataset(
        dataset_name=cfg.DATASET.NAME,
        raw_file_paths=raw_file_paths,
        raw_label_path=raw_label_path,
        num_omics=len(cfg.DATASET.OMICS),
        num_classes=cfg.DATASET.NUM_CLASSES,
        sparsity_rates=cfg.DATASET.FEATURE_SPARSITY_RATES
    )
    print(multiomics)


    final_gat_results = {}
    gat_results = defaultdict(lambda: defaultdict(list))
    time_results = defaultdict(list)        
    for fold_idx in range(5):
        seed_everything(cfg.SOLVER.SEED)
        print(f"==> Loading data from fold {fold_idx + 1}...")
        fold_multiomics = prepare_data(main_folder, fold_idx, multiomics, cfg)
        for feat_size in cfg.GAT.FINAL_FEAT_SIZES:
            multiomics_deepcopy1 = copy.deepcopy(fold_multiomics)
            multiomics_deepcopy = copy.deepcopy(fold_multiomics)
            x_train = [multiomics_deepcopy.get(i).train_data.values for i in range(multiomics_deepcopy.num_omics)]
            y_train = multiomics_deepcopy.get(0).train_label.values.ravel()
            # 执行新特征选择
            final_feat_subset = select_top_feats(x_train, y_train, feat_size, multiomics_deepcopy.num_omics)

            # 应用特征选择结果
            multiomics_deepcopy.reduce_dimensionality(final_feat_subset)

            start_time = time.time()
            hetero_data = HeteroDataset(multiomics_deepcopy1,multiomics_deepcopy, sparsity_rate=cfg.DATASET.PATIENT_SPARSITY_RATES,
                                                tune_hyperparameters=cfg.SOLVER.TUNE_HYPER,
                                                seed=cfg.SOLVER.SEED)
            hetero_data.create_hetero_data()
            # ---- setup model ----

            print("\n   ==> Building model...")
            new_model = NewModel(dataset=hetero_data,
                                dataset1=multiomics_deepcopy1,
                                num_modalities=multiomics_deepcopy.num_omics,
                                num_classes=multiomics_deepcopy.num_classes,
                                gat_num_layers=cfg.GAT.NUM_LAYERS,
                                gat_num_heads=cfg.GAT.NUM_HEADS,
                                gat_hidden_dim=cfg.GAT.HIDDEN_DIM,
                                weight=cfg.GAT.WEIGHT,
                                gat_dropout_rate=cfg.GAT.DROPOUT_RATE,
                                gat_lr_pretrain=cfg.GAT.LR_PRETRAIN,
                                gat_lr=cfg.GAT.LR,
                                gat_wd=cfg.GAT.WD,
                                vcdn_lr=cfg.VCDN.LR,
                                vcdn_wd=cfg.VCDN.WD,
                                trans_dropout_rate=0.2,
                                tune_hyperparameters=cfg.SOLVER.TUNE_HYPER,
                                )
            checkpoint_callback = None
            callbacks = []
            if cfg.SOLVER.TUNE_HYPER:
                filename=f"model-epoch={{epoch}}-val_acc={{val_acc:.3f}}-val_loss={{val_loss:.3f}}"
            # 创建检查点回调，基于验证准确率保存多个最佳模型
                checkpoint_callback = ModelCheckpoint(
                        monitor="val_acc",           # 监控验证集准确率
                        mode="max",                  # 最大化验证准确率
                        save_top_k=40,                # 保存验证准确率最高的3个模型
                        filename=filename,
                        dirpath="model13",
                        auto_insert_metric_name=False,
                        save_last=False
                    )
                callbacks.append(checkpoint_callback)
            model = new_model.get_model(pretrain=False).to(device)
                # for param in model.parameters():
                #     param.requires_grad_(True)
            # ---- setup training model and trainer ----
            print("\n   ==> Training model...")
            trainer = pl.Trainer(
                        max_epochs=cfg.SOLVER.MAX_EPOCHS,
                        default_root_dir=cfg.RESULT.LIGHTNING_LOG_DIR,
                        accelerator="auto", 
                        devices="auto",
                        enable_model_summary=False,
                        log_every_n_steps=1,
                        callbacks=callbacks, 
                        )
            print("Starting training...")
            trainer.fit(model)
            
            save_model_name = cfg.RESULT.SAVE_MODEL_TMPL.format(
                    dataset_name=cfg.DATASET.NAME,
                    fold_idx=fold_idx + 1,
                    feat_size=feat_size,
                    seed=cfg.SOLVER.SEED,
                )
            save_path = os.path.join("model13", save_model_name)
    
        if cfg.RESULT.SAVE_MODEL:   
                # 情况1：保存最佳模型（调参模式下）
            if cfg.SOLVER.TUNE_HYPER:
        # 获取所有保存的检查点
                saved_models = glob.glob(os.path.join("model13", "model-epoch=*.ckpt"))
        
        # 按照验证损失排序找到最佳模型
                best_model_path = None
                best_val_acc = float('-inf')
                best_val_loss = float('inf')  # 用于在val_acc相同时比较
                    
                for model_path in saved_models:
                        # 从文件名中提取验证准确率和验证损失值
                    val_acc_match = re.search(r"val_acc=([\d\.]+)", model_path)
                    val_loss_match = re.search(r"val_loss=([\d\.]+)", model_path)
                        
                        # 确保两者都能成功解析
                    if val_acc_match and val_loss_match:
                            
                        val_acc = float(val_acc_match.group(1))
                        val_loss = val_loss_match.group(1)
                        if val_loss.endswith('.'):
                            val_loss = val_loss.rstrip('.')

                        val_loss = float(val_loss)
                        print(f"模型: {model_path} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")
                            
                        # 首先比较准确率
                        if val_acc > best_val_acc:
                                # 找到更好的准确率
                            best_val_acc = val_acc
                            best_val_loss = val_loss  # 重置损失
                            best_model_path = model_path
                            
                            # 当准确率相同但损失更小
                        elif val_acc == best_val_acc and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_path = model_path
                        
                    
                if best_model_path:
                    print(f"最终选择模型: {best_model_path}")
                    print(f"验证准确率: {best_val_acc:.4f}, 验证损失: {best_val_loss:.4f}")
                if best_model_path:
                    print(f"找到最佳模型: {best_model_path}, 验证损失: {best_val_acc:.4f}")
                    shutil.copy(best_model_path, save_path)
                    print(f"已保存最佳模型到: {save_path}")
                # for model_path in saved_models:
                #     os.remove(model_path)
            else:
                trainer.save_checkpoint(save_path)
                # ---- test model ----
                print("\n   ==> Testing model...")

        if cfg.SOLVER.TUNE_HYPER:
                # 加载最佳模型（需传递模型初始化参数！）
            model = ModelTrainer.load_from_checkpoint(
                    checkpoint_path=save_path,
                    # 传递初始化参数（必须与训练时一致）
                    trial=None,
                    dataset=new_model.dataset,
                    dataset1=new_model.dataset1,
                    weight=new_model.weight,
                    num_modalities=new_model.num_modalities,
                    num_classes=new_model.num_classes,
                    unimodal_model=new_model.unimodal_model,
                    unimodal_model1=new_model.unimodal_model1,
                    multimodal_decoder=new_model.multimodal_decoder,
                    loss_fn=new_model.loss_function,
                    train_multimodal_decoder=True,  # 根据实际阶段设置
                    gat_wd=new_model.gat_wd,
                    gat_lr=new_model.gat_lr,
                    vcdn_lr=new_model.vcdn_lr,
                    vcdn_wd=new_model.vcdn_wd,
                    contrastive_loss_fn=new_model.contrastive_loss,
                    tune_hyperparameters=new_model.tune_hyperparameters,
                    strict=True  # 允许忽略缺失的键（如预训练时无多模态解码器）
                )

    
            model.eval()
            model.requires_grad_(False)
                
        trainer.test(model)

        end_time = time.time()
        running_time = end_time - start_time

        time_results[feat_size].append(running_time)
        if cfg.RESULT.SAVE_RESULT:
            time_result = [feat_size, fold_idx + 1, running_time]
            save_output(output_gat_file_time, time_result)

        final_gat_results.setdefault(feat_size, []).append(model.get_log_metrics())

        for metric_key, metric_value in model.get_log_metrics().items():
            if metric_key.startswith("test_"):
                gat_results[feat_size][metric_key.replace("test_", "")].append(metric_value[0])
                if cfg.RESULT.SAVE_RESULT:
                    result = [feat_size, metric_key.replace("test_", ""), fold_idx + 1, metric_value[0]]
                    save_output(output_gat_file, result)

        for feat_size, metrics in gat_results.items():
            for metric_name, values in metrics.items():
                average = np.mean(values)
                std = np.std(values)
                print(f"    - {metric_name}: {average:.3f}±{std:.3f}")

    if cfg.RESULT.SAVE_RESULT:
        sort_file(output_gat_file, sorted_output_gat_file, by=cfg.RESULT.FILE_HEADER_GAT[0:3])
        sort_file(output_gat_file_time, sorted_output_gat_file_time, by=cfg.RESULT.FILE_HEADER_GAT_TIME[0:2])

    print(f"\n==> Showing results...")
    for feat_size, metrics in gat_results.items():
        exe_time = round(np.mean(time_results[feat_size]))
        print(f"Feature size {feat_size} (execution time: {exe_time} seconds)")
        for metric_name, values in metrics.items():
            average = np.mean(values)
            std = np.std(values)
            print(f"    - {metric_name}: {average:.3f}±{std:.3f}")





if __name__ == '__main__':
    main()
