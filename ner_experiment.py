#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese Geographic NER Experiment Main Entry
Integrated with Geo Knowledge Graph and GeoGLUE Evaluation
"""
import os
# 必须在 import torch 之前设置，否则 CUDA 初始化后环境变量失效
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import csv
import json
import time
import gc
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 同步 ownthink 智能筛选参数到环境变量（供 geo_knowledge.py 读取，避免重新实例化 Config）
from config_ner import cfg_ner as _cfg_early
os.environ.setdefault('OWNTHINK_MAX_ORG', str(getattr(_cfg_early, 'ownthink_max_org', 200000)))
os.environ.setdefault('OWNTHINK_MAX_PER', str(getattr(_cfg_early, 'ownthink_max_per', 200000)))
os.environ.setdefault('OWNTHINK_MIN_LEN', str(getattr(_cfg_early, 'ownthink_min_entity_len', 2)))
os.environ.setdefault('OWNTHINK_MAX_LEN', str(getattr(_cfg_early, 'ownthink_max_entity_len', 15)))
del _cfg_early

import torch
import numpy as np

from config_ner import cfg_ner
from models import NERModel
from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, calculate_class_weights, create_lexicon_matcher
from geo_knowledge import GeoKnowledgeGraph, build_enhanced_lexicon_matcher
from geoglue_evaluator import GeoGLUEEvaluator, evaluate_long_tail_entities, save_geoglue_results, save_long_tail_results
from transformers import AutoTokenizer

# ========================== 实验模式配置 ==========================
# 只需要修改下面这个变量来选择运行模式！
#
#   "TEST"      -> 测试模式：快速验证代码能否运行（少量数据，1个epoch）
#   "ABLATION"  -> 消融实验：6个模型对比（BiLSTM组3个 + BiGRU组3个）
#   "FULL"      -> 完整实验：所有13个模型
#   "GEO_TEST"  -> 地理测试：对消融实验的6个模型进行GeoGLUE和长尾分析（BiLSTM组 + BiGRU组）
#   "STATICLEX" -> Ours-StaticLex 专项实验：消融变体训练 + 基准测试 + 网络文本错误分类
#
RUN_MODE = "STATICLEX"  # <-- 修改这里！！！

# 测试模式专用设置（当 RUN_MODE="TEST" 时生效）
TEST_MODE_CONFIG = {
    "sample_size": 50,        # 每数据集采样数量
    "num_epochs": 1,          # 训练轮数
    "batch_size": 8,           # 批次大小
    "eval_batch_size": 16,     # 评估批次大小
    "patience": 2,             # 早停耐心值
    "max_seq_length": 64,      # 最大序列长度
    "run_geoglue": False,      # 是否运行GeoGLUE评测
    "run_real_text": True,     # 是否运行真实文本测试
}

# ========================== Visualization Setup ==========================
def setup_visualization():
    """Setup matplotlib for English-only charts"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return plt
    except ImportError:
        return None

plt_lib = setup_visualization()

# ========================== Training Function ==========================
def train_single_ner_model(model_config, train_dataset, dev_dataset, tokenizer, cfg, history_save_path=None):
    model_name = model_config['name']
    use_crf = model_config['use_crf']
    use_lexicon = model_config['use_lexicon']
    lexicon_type = model_config.get('lexicon_type', 'static')
    ablation_config = model_config.get('ablation_config', None)
    adapter_only_finetune = model_config.get('adapter_only_finetune', False)
    full_finetune_from_ckpt = model_config.get('full_finetune_from_ckpt', False)
    num_epochs_override = model_config.get('num_epochs_override', None)

    # 全量微调模式：加载基线权重但不冻结任何层
    if full_finetune_from_ckpt:
        if cfg.staticlex_reuse_dynamiclex_ckpt:
            baseline_ckpt = os.path.join(cfg.model_save_dir, cfg.staticlex_dynamiclex_ckpt_name)
        else:
            baseline_ckpt = os.path.join(cfg.model_save_dir, f"ner_model_{cfg.roberta_model_name.replace('/', '_')}_Ours-StaticLex.pt")
        actual_num_epochs = num_epochs_override or cfg.staticlex_full_finetune_epochs
        actual_lr = cfg.ner_learning_rate
        actual_patience = cfg.staticlex_full_finetune_patience
    elif adapter_only_finetune:
        if cfg.staticlex_reuse_dynamiclex_ckpt:
            baseline_ckpt = os.path.join(cfg.model_save_dir, cfg.staticlex_dynamiclex_ckpt_name)
            print(f"\n[Adapter Fine-tune] 复用基线检查点: {baseline_ckpt}")
            if not os.path.exists(baseline_ckpt):
                raise FileNotFoundError(f"基线检查点不存在: {baseline_ckpt}，请先训练 Ours-StaticLex 基线")
        else:
            baseline_ckpt = os.path.join(cfg.model_save_dir, f"ner_model_{cfg.roberta_model_name.replace('/', '_')}_Ours-StaticLex.pt")
        # 适配器微调使用更短轮数和更大学习率
        actual_num_epochs = num_epochs_override or cfg.staticlex_adapter_epochs
        actual_lr = cfg.staticlex_adapter_lr
        actual_patience = cfg.staticlex_adapter_patience
    else:
        actual_num_epochs = num_epochs_override or cfg.ner_num_epochs
        actual_lr = cfg.ner_learning_rate
        actual_patience = cfg.ner_patience

    print(f"\n{'='*60}")
    print(f"Training Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Base Model: {model_config['model_name']}")
    print(f"  Use CRF: {use_crf}")
    print(f"  Use Lexicon: {use_lexicon}")
    print(f"  Lexicon Type: {lexicon_type}")
    print(f"  Ablation Config: {ablation_config}")
    if full_finetune_from_ckpt:
        print(f"  [Full Fine-tune from Ckpt] epochs={actual_num_epochs}, lr={actual_lr}, patience={actual_patience}")
    elif adapter_only_finetune:
        print(f"  [Adapter Fine-tune Mode] epochs={actual_num_epochs}, lr={actual_lr}, patience={actual_patience}")

    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU Memory Before Model] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        print(f"  GPU Memory Cleared")

    # 自动计算类别权重（基于训练数据分布）
    if cfg.use_class_weights and cfg.class_weight_method != 'manual':
        auto_weights = calculate_class_weights(
            train_dataset, 
            cfg.ner_label2id, 
            method=cfg.class_weight_method,
            min_weight=cfg.class_weight_min,
            max_weight=cfg.class_weight_max
        )
        if auto_weights is not None:
            cfg.ner_class_weights = auto_weights
            print(f"[类别权重] 使用自动计算的权重: {cfg.ner_class_weights}")

    model = NERModel(
        model_name=model_config['model_name'],
        num_labels=len(cfg.ner_label2id),
        model_variant=model_name,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=cfg.lexicon_size,
        lexicon_type=lexicon_type,
        use_reshaping=False,
        ablation_config=ablation_config,
        lexicon_config={
            'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels
        }
    )
    model.to(cfg.device)

    # 全量微调模式：加载基线权重但不冻结任何层（所有参数可训练）
    if full_finetune_from_ckpt:
        print(f"  [Full Fine-tune] 加载基线权重 (strict=False): {baseline_ckpt}")
        if not os.path.exists(baseline_ckpt):
            print(f"  [Full Fine-tune] WARNING: 基线检查点不存在，将从随机初始化训练")
        else:
            state_dict = torch.load(baseline_ckpt, map_location=cfg.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [Full Fine-tune] Missing keys (消融新增参数): {len(missing)} 个")
                for k in missing[:5]:
                    print(f"    - {k}")
            if unexpected:
                print(f"  [Full Fine-tune] Unexpected keys: {len(unexpected)} 个")
                for k in unexpected[:5]:
                    print(f"    - {k}")
        n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        print(f"  [Full Fine-tune] All params trainable: {n_trainable}")

    # 适配器微调模式：加载基线权重 + 冻结编码器（strict=False 允许适配器形状变化）
    elif adapter_only_finetune:
        print(f"  [Adapter Fine-tune] 加载基线权重 (strict=False): {baseline_ckpt}")
        state_dict = torch.load(baseline_ckpt, map_location=cfg.device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [Adapter Fine-tune] Missing keys (适配器新参数): {len(missing)} 个")
            for k in missing[:5]:
                print(f"    - {k}")
        if unexpected:
            print(f"  [Adapter Fine-tune] Unexpected keys (旧适配器替换): {len(unexpected)} 个")
            for k in unexpected[:5]:
                print(f"    - {k}")
        # 冻结编码器/CRF/中间层，仅微调适配器
        freeze_keys = cfg.staticlex_freeze_keys
        n_frozen = 0
        n_trainable = 0
        for n, p in model.named_parameters():
            if any(fk in n for fk in freeze_keys):
                p.requires_grad = False
                n_frozen += 1
            else:
                p.requires_grad = True
                n_trainable += 1
        print(f"  [Adapter Fine-tune] Frozen params: {n_frozen}, Trainable params: {n_trainable}")

    # 差异化学习率：RoBERTa编码器用小学习率，词典增强模块用大学习率
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    encoder_params = []
    encoder_params_no_decay = []
    lexicon_params = []
    lexicon_params_no_decay = []

    for n, p in param_optimizer:
        if not p.requires_grad:
            continue
        if 'lexicon_adapter' in n:
            if any(nd in n for nd in no_decay):
                lexicon_params_no_decay.append(p)
            else:
                lexicon_params.append(p)
        else:
            if any(nd in n for nd in no_decay):
                encoder_params_no_decay.append(p)
            else:
                encoder_params.append(p)

    if adapter_only_finetune:
        # 仅适配器可训练，使用统一学习率
        optimizer_grouped_parameters = []
        if lexicon_params:
            optimizer_grouped_parameters.append({'params': lexicon_params, 'weight_decay': cfg.weight_decay, 'lr': actual_lr})
        if lexicon_params_no_decay:
            optimizer_grouped_parameters.append({'params': lexicon_params_no_decay, 'weight_decay': 0.0, 'lr': actual_lr})
        # 编码器被冻结，不应出现在优化器组
        if encoder_params or encoder_params_no_decay:
            print(f"  [Adapter Fine-tune] Warning: 发现未冻结的编码器参数，但 adapter_only_finetune=True")
    else:
        optimizer_grouped_parameters = [
            {'params': encoder_params, 'weight_decay': cfg.weight_decay, 'lr': actual_lr},
            {'params': encoder_params_no_decay, 'weight_decay': 0.0, 'lr': actual_lr},
            {'params': lexicon_params, 'weight_decay': cfg.weight_decay, 'lr': cfg.ner_lexicon_learning_rate},
            {'params': lexicon_params_no_decay, 'weight_decay': 0.0, 'lr': cfg.ner_lexicon_learning_rate},
        ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g['params']]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_num_epochs)

    train_loader = create_dataloader(train_dataset, batch_size=cfg.ner_batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, batch_size=cfg.ner_eval_batch_size, shuffle=False)

    training_history = {
        'epoch': [],
        'train_loss': [],
        'dev_loss': [],
        'dev_micro_f1': [],
        'dev_macro_f1': []
    }

    best_micro_f1 = 0.0
    patience_counter = 0

    use_amp = cfg.device.type == 'cuda' and not cfg.use_fp32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"  Use AMP: {use_amp}")

    for epoch in range(actual_num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{actual_num_epochs}", leave=False, ncols=60)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)

            lexicon_indices = batch.get('lexicon_indices', None)
            lexicon_levels = batch.get('lexicon_levels', None)
            lexicon_types = batch.get('lexicon_types', None)
            lexicon_coords = batch.get('lexicon_coords', None)

            if lexicon_indices is not None:
                lexicon_indices = lexicon_indices.to(cfg.device)
            if lexicon_levels is not None:
                lexicon_levels = lexicon_levels.to(cfg.device)
            if lexicon_types is not None:
                lexicon_types = lexicon_types.to(cfg.device)
            if lexicon_coords is not None:
                lexicon_coords = lexicon_coords.to(cfg.device)

            if batch_idx == 0 and epoch == 0 and use_lexicon:
                print(f"\n[Debug] Lexicon Index Range Check:")
                if lexicon_indices is not None:
                    print(f"  lexicon_indices: max={lexicon_indices.max().item()}, cfg.lexicon_size={cfg.lexicon_size}")
                    match_ratio = (lexicon_indices != 0).float().mean().item()
                    print(f"  lexicon match ratio (first batch): {match_ratio:.4f}")
                if lexicon_levels is not None:
                    print(f"  lexicon_levels: max={lexicon_levels.max().item()}, num_levels={cfg.num_geo_levels}")
                if lexicon_types is not None:
                    print(f"  lexicon_types: max={lexicon_types.max().item()}")
                if hasattr(model, 'lexicon_adapter') and model.lexicon_adapter is not None:
                    print(f"  word_embed.num_embeddings={model.lexicon_adapter.word_embed.num_embeddings}")
                    if hasattr(model.lexicon_adapter, 'fusion_gate'):
                        print(f"  fusion_gate initial value: {model.lexicon_adapter.fusion_gate.item():.4f} (sigmoid={torch.sigmoid(model.lexicon_adapter.fusion_gate).item():.4f})")

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lexicon_indices=lexicon_indices,
                    lexicon_levels=lexicon_levels,
                    lexicon_types=lexicon_types,
                    lexicon_coords=lexicon_coords,
                    ner_tags=labels
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss (NaN/Inf), skip this batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            if batch_idx == 0 and epoch == 0 and use_lexicon and hasattr(model, 'lexicon_adapter') and model.lexicon_adapter is not None:
                print(f"\n[Debug] Gradient Check (first batch):")
                for name, param in model.lexicon_adapter.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"  {name}: grad_norm={grad_norm:.6f}, requires_grad={param.requires_grad}")
                    else:
                        print(f"  {name}: grad=None, requires_grad={param.requires_grad}")
                if hasattr(model.lexicon_adapter, 'fusion_gate'):
                    print(f"  fusion_gate before step: {model.lexicon_adapter.fusion_gate.item():.6f}")
            
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx == 0 and epoch == 0 and use_lexicon and hasattr(model, 'lexicon_adapter') and model.lexicon_adapter is not None:
                if hasattr(model.lexicon_adapter, 'fusion_gate'):
                    print(f"  fusion_gate after step: {model.lexicon_adapter.fusion_gate.item():.6f}")
                    print(f"  delta (per step): {model.lexicon_adapter.fusion_gate.item() - 0.0:.6f}")
            
            train_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0 and cfg.device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        scheduler.step()

        model.eval()
        dev_loss = 0.0
        num_dev_batches = 0
        all_preds = []
        all_labels = []
        total_lexicon_matches = 0
        total_lexicon_positions = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Dev Eval {epoch+1}/{actual_num_epochs}", leave=False, ncols=60):
                input_ids = batch['input_ids'].to(cfg.device)
                attention_mask = batch['attention_mask'].to(cfg.device)
                labels = batch['labels'].to(cfg.device)

                lexicon_indices = batch.get('lexicon_indices', None)
                lexicon_levels = batch.get('lexicon_levels', None)
                lexicon_types = batch.get('lexicon_types', None)
                lexicon_coords = batch.get('lexicon_coords', None)

                if lexicon_indices is not None:
                    lexicon_indices = lexicon_indices.to(cfg.device)
                if lexicon_levels is not None:
                    lexicon_levels = lexicon_levels.to(cfg.device)
                if lexicon_types is not None:
                    lexicon_types = lexicon_types.to(cfg.device)
                if lexicon_coords is not None:
                    lexicon_coords = lexicon_coords.to(cfg.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, logits, _, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        lexicon_indices=lexicon_indices,
                        lexicon_levels=lexicon_levels,
                        lexicon_types=lexicon_types,
                        lexicon_coords=lexicon_coords,
                        ner_tags=labels
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                dev_loss += loss.item()
                num_dev_batches += 1

                if lexicon_indices is not None:
                    total_lexicon_matches += (lexicon_indices != 0).float().sum().item()
                    total_lexicon_positions += lexicon_indices.numel()

                # CRF decode 在 fp32 下进行，避免 AMP fp16 精度问题
                logits_fp32 = logits.float() if logits.dtype != torch.float32 else logits
                predictions = model._decode(logits_fp32, attention_mask)

                # predictions 是列表，每个元素是不同长度的序列
                # 不能直接转换为 numpy 数组，需要逐个处理
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                    if predictions.ndim == 1:
                        predictions = [predictions]
                    else:
                        predictions = [predictions[i] for i in range(len(predictions))]
                elif isinstance(predictions, np.ndarray):
                    if predictions.ndim == 1:
                        predictions = [predictions]
                    else:
                        predictions = [predictions[i] for i in range(len(predictions))]
                # 如果已经是列表，保持不变

                labels_np = labels.cpu().numpy()

                # 遍历每个样本
                for i in range(len(predictions)):
                    pred_seq = predictions[i]
                    label_seq = labels_np[i]

                    # 确保是可迭代的
                    if isinstance(pred_seq, (int, np.integer)):
                        pred_seq = [pred_seq]

                    # 确保长度一致
                    min_len = min(len(pred_seq), len(label_seq))
                    for j in range(min_len):
                        if label_seq[j] != -100:
                            all_preds.append(int(pred_seq[j]))
                            all_labels.append(int(label_seq[j]))

        avg_dev_loss = dev_loss / num_dev_batches if num_dev_batches > 0 else 0.0

        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()

        from sklearn.metrics import f1_score
        if len(all_preds) == 0 or len(all_labels) == 0:
            print(f"  [WARN] Dev evaluation produced no predictions (all batches skipped?)")
            micro_f1 = 0.0
            macro_f1 = 0.0
        else:
            micro_f1 = f1_score(all_labels, all_preds, average='micro')
            macro_f1 = f1_score(all_labels, all_preds, average='macro')

        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['dev_loss'].append(avg_dev_loss)
        training_history['dev_micro_f1'].append(micro_f1)
        training_history['dev_macro_f1'].append(macro_f1)

        print(f"Epoch {epoch+1}/{actual_num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {avg_dev_loss:.4f}")
        print(f"  Dev Micro-F1: {micro_f1:.4f}")
        print(f"  Dev Macro-F1: {macro_f1:.4f}")
        
        if use_lexicon and hasattr(model, 'lexicon_adapter') and model.lexicon_adapter is not None:
            if hasattr(model.lexicon_adapter, 'fusion_gate'):
                gate_val = model.lexicon_adapter.fusion_gate.item()
                alpha = torch.sigmoid(model.lexicon_adapter.fusion_gate).item()
                print(f"  Fusion Gate: value={gate_val:.6f}, sigmoid(alpha)={alpha:.4f}")
            elif hasattr(model.lexicon_adapter, 'fusion_gate_net'):
                with torch.no_grad():
                    concat = torch.cat([char_hidden[:1], torch.zeros_like(char_hidden[:1])], dim=-1)
                    gate_logits = model.lexicon_adapter.fusion_gate_net(concat)
                    avg_alpha = torch.sigmoid(gate_logits).mean().item()
                print(f"  Avg Fusion Gate Alpha: {avg_alpha:.4f}")
        
        if total_lexicon_positions > 0:
            lexicon_ratio = total_lexicon_matches / total_lexicon_positions
            print(f"  Lexicon Match Ratio: {lexicon_ratio:.4f}")
        
        from collections import Counter
        pred_counter = Counter(all_preds)
        label_counter = Counter(all_labels)
        print(f"  Prediction Distribution: {dict(sorted(pred_counter.items(), key=lambda x: -x[1])[:5])}")
        print(f"  Label Distribution: {dict(sorted(label_counter.items(), key=lambda x: -x[1])[:5])}")

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            patience_counter = 0
            model_path = os.path.join(cfg.model_save_dir, f"ner_model_{model_name.replace('/', '_').replace('+', '_')}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= actual_patience:
                print(f"  Early stopping triggered")
                break

    if history_save_path:
        with open(history_save_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_history.keys())
            writer.writeheader()
            for i in range(len(training_history['epoch'])):
                row = {k: training_history[k][i] for k in training_history.keys()}
                writer.writerow(row)

    print(f"\nTraining completed, best validation Micro-F1: {best_micro_f1:.4f}")
    return model, best_micro_f1, training_history

# ========================== Evaluation Function ==========================
def evaluate_ner_model(model, test_loader, id2label, device, use_crf=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            lexicon_indices = batch.get('lexicon_indices', None)
            lexicon_levels = batch.get('lexicon_levels', None)
            lexicon_types = batch.get('lexicon_types', None)
            lexicon_coords = batch.get('lexicon_coords', None)

            if lexicon_indices is not None:
                lexicon_indices = lexicon_indices.to(device)
            if lexicon_levels is not None:
                lexicon_levels = lexicon_levels.to(device)
            if lexicon_types is not None:
                lexicon_types = lexicon_types.to(device)
            if lexicon_coords is not None:
                lexicon_coords = lexicon_coords.to(device)

            predictions, _, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lexicon_indices=lexicon_indices,
                lexicon_levels=lexicon_levels,
                lexicon_types=lexicon_types,
                lexicon_coords=lexicon_coords
            )

            pred_tags = predictions
            
            # pred_tags 是列表，每个元素是不同长度的序列
            # 不能直接转换为 numpy 数组，需要逐个处理
            if isinstance(pred_tags, torch.Tensor):
                pred_tags = pred_tags.cpu().numpy()
                if pred_tags.ndim == 1:
                    pred_tags = [pred_tags]
                else:
                    pred_tags = [pred_tags[i] for i in range(len(pred_tags))]
            elif isinstance(pred_tags, np.ndarray):
                if pred_tags.ndim == 1:
                    pred_tags = [pred_tags]
                else:
                    pred_tags = [pred_tags[i] for i in range(len(pred_tags))]
            # 如果已经是列表，保持不变
            
            labels_np = labels.cpu().numpy()
            
            # 遍历每个样本
            for i in range(len(pred_tags)):
                pred_seq = pred_tags[i]
                label_seq = labels_np[i]
                
                # 确保是可迭代的
                if isinstance(pred_seq, (int, np.integer)):
                    pred_seq = [pred_seq]
                
                # 确保长度一致
                min_len = min(len(pred_seq), len(label_seq))
                for j in range(min_len):
                    if label_seq[j] != -100:
                        all_preds.append(int(pred_seq[j]))
                        all_labels.append(int(label_seq[j]))

    from sklearn.metrics import f1_score
    return {
        'micro_f1': f1_score(all_labels, all_preds, average='micro'),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'predictions': all_preds,
        'labels': all_labels
    }

# ========================== Result Saving ==========================
def save_results_to_csv(results, csv_path):
    if not results:
        return

    fieldnames = ['model_name', 'best_dev_f1', 'avg_micro_f1']
    dataset_names = set()

    for result in results:
        for key in result.keys():
            if key.endswith('_micro_f1') or key.endswith('_macro_f1'):
                dataset_names.add(key.replace('_micro_f1', '').replace('_macro_f1', ''))

    for dataset in sorted(dataset_names):
        fieldnames.append(f'{dataset}_micro_f1')
        fieldnames.append(f'{dataset}_macro_f1')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_path}")

# ========================== Visualization Functions ==========================
def plot_model_comparison_bar(results, save_dir):
    if plt_lib is None or not results:
        return
    plt = plt_lib
    dataset_names = sorted([k.replace('_micro_f1', '') for r in results for k in r.keys() if k.endswith('_micro_f1')])
    model_names = [r['model_name'] for r in results]
    x = np.arange(len(model_names))
    width = 0.18
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, dataset in enumerate(dataset_names):
        values = [r.get(f'{dataset}_micro_f1', 0) * 100 for r in results]
        offset = (i - len(dataset_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dataset, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Micro-F1 (%)', fontsize=13)
    ax.set_title('NER Model Comparison Across Test Sets', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, ncol=len(dataset_names))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f'model_comparison_bar.{ext}'), dpi=300, bbox_inches='tight', format=ext)
    plt.close()


def plot_ablation_study(results, save_dir):
    if plt_lib is None or not results:
        return
    plt = plt_lib
    dataset_names = sorted([k.replace('_micro_f1', '') for r in results for k in r.keys() if k.endswith('_micro_f1')])
    model_names = [r['model_name'] for r in results]
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, metric in enumerate(['micro_f1', 'macro_f1']):
        ax = axes[idx]
        for i, dataset in enumerate(dataset_names):
            values = [r.get(f'{dataset}_{metric}', 0) * 100 for r in results]
            offset = (i - len(dataset_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=dataset, color=colors[i % len(colors)])
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 添加分组分隔线
        if len(model_names) == 6:
            ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(1.5, ax.get_ylim()[1] * 0.95, 'BiLSTM Group', ha='center', fontsize=11, fontweight='bold', color='#2980b9')
            ax.text(4.5, ax.get_ylim()[1] * 0.95, 'BiGRU Group', ha='center', fontsize=11, fontweight='bold', color='#c0392b')
        
        ax.set_xlabel('Model Variant', fontsize=12)
        ax.set_ylabel(f'{metric.upper().replace("_", "-")} (%)', fontsize=12)
        ax.set_title(f'Ablation Study - {metric.upper().replace("_", "-")}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f'ablation_study.{ext}'), dpi=300, bbox_inches='tight', format=ext)
    plt.close()


def plot_geoglue_results(geoglue_results, save_dir):
    if plt_lib is None or not geoglue_results:
        return
    plt = plt_lib

    f1_tasks = []
    micro_f1s = []
    macro_f1s = []
    acc_tasks = []
    accuracies = []
    recall_tasks = []
    recalls = []
    rerank_tasks = []
    mrrs = []

    for task, result in geoglue_results.items():
        if 'micro_f1' in result:
            f1_tasks.append(task)
            micro_f1s.append(result['micro_f1'] * 100)
            macro_f1s.append(result['macro_f1'] * 100)
        elif 'accuracy' in result and 'mrr' not in result:
            acc_tasks.append(task)
            accuracies.append(result['accuracy'] * 100)
        elif 'entity_recall' in result:
            recall_tasks.append(task)
            recalls.append(result['entity_recall'] * 100)
        elif 'mrr' in result:
            rerank_tasks.append(task)
            mrrs.append(result['mrr'] * 100)

    num_plots = sum([len(f1_tasks) > 0, len(acc_tasks) > 0, len(recall_tasks) > 0, len(rerank_tasks) > 0])
    if num_plots == 0:
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    if len(f1_tasks) > 0:
        ax = axes[plot_idx]
        x = np.arange(len(f1_tasks))
        width = 0.35
        bars1 = ax.bar(x - width/2, micro_f1s, width, label='Micro-F1', color='#1f77b4')
        bars2 = ax.bar(x + width/2, macro_f1s, width, label='Macro-F1', color='#ff7f0e')
        for bar, val in zip(bars1, micro_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        for bar, val in zip(bars2, macro_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('GeoGLUE Task', fontsize=12)
        ax.set_ylabel('F1 Score (%)', fontsize=12)
        ax.set_title('GeoGLUE F1-based Tasks', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(f1_tasks, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
        plot_idx += 1

    if len(acc_tasks) > 0:
        ax = axes[plot_idx]
        x = np.arange(len(acc_tasks))
        bars = ax.bar(x, accuracies, color='#2ecc71', alpha=0.8)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('GeoGLUE Task', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('GeoGLUE Accuracy-based Tasks', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(acc_tasks, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
        plot_idx += 1

    if len(recall_tasks) > 0:
        ax = axes[plot_idx]
        x = np.arange(len(recall_tasks))
        bars = ax.bar(x, recalls, color='#9b59b6', alpha=0.8)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('GeoGLUE Task', fontsize=12)
        ax.set_ylabel('Entity Recall (%)', fontsize=12)
        ax.set_title('GeoGLUE Recall-based Tasks', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(recall_tasks, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
        plot_idx += 1

    if len(rerank_tasks) > 0:
        ax = axes[plot_idx]
        x = np.arange(len(rerank_tasks))
        bars = ax.bar(x, mrrs, color='#e67e22', alpha=0.8)
        for bar, val in zip(bars, mrrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('GeoGLUE Task', fontsize=12)
        ax.set_ylabel('MRR (%)', fontsize=12)
        ax.set_title('GeoGLUE Reranking Tasks', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rerank_tasks, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
        plot_idx += 1

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f'geoglue_results.{ext}'), dpi=300, bbox_inches='tight', format=ext)
    plt.close()

# ========================== Real-world Text Test ==========================
def test_on_real_text(model, tokenizer, lexicon_matcher, id2label, label2id, cfg, model_name, output_dir):
    print(f"\n{'='*80}")
    print(f"Real-world Text Inference Test - {model_name}")
    print(f"{'='*80}")

    test_cases = [
        {'category': 'News Headlines', 'texts': [
            "Beijing hosted a major international summit yesterday.",
            "Typhoon made landfall in Jinmen, Fujian Province.",
            "The World AI Conference was held in Shanghai Pudong."
        ]},
        {'category': 'Social Media', 'texts': [
            "Just landed in Chengdu, going to visit the Panda Base.",
            "Guangzhou dinner with friends tonight.",
            "Hangzhou West Lake was beautiful this weekend."
        ]},
        {'category': 'Travel Reviews', 'texts': [
            "We traveled from Xi'an to Lhasa by train.",
            "Three days in Sanya, the beach was amazing.",
            "Beijing-Hong Kong high-speed rail is convenient."
        ]},
        {'category': 'Emergency Reports', 'texts': [
            "Earthquake struck Jishishan County in Qinghai.",
            "Heavy rainfall caused flooding in Zhengzhou.",
            "Forest fire broke out in Liangshan, Sichuan."
        ]}
    ]

    all_entities = []
    inference_log = []

    for case in test_cases:
        print(f"\n--- Category: {case['category']} ---")
        for text in case['texts']:
            print(f"\nInput: {text}")
            start_time = time.time()

            encoding = tokenizer(text, max_length=cfg.max_seq_length, padding='max_length', truncation=True, return_offsets_mapping=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to(cfg.device)
            attention_mask = encoding['attention_mask'].to(cfg.device)

            with torch.no_grad():
                predictions, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # predictions 可能是列表（CRF模式）或 tensor（非CRF模式）
            if isinstance(predictions, list):
                # CRF 模式返回列表
                pred_tags = predictions[0] if len(predictions) > 0 else []
                if isinstance(pred_tags, torch.Tensor):
                    pred_tags = pred_tags.cpu().numpy()
            else:
                # 非 CRF 模式返回 tensor
                pred_tags = predictions[0].cpu().numpy()
            
            tags = [id2label.get(p, 'O') for p in pred_tags]
            
            entities = []
            current = None
            for i, tag in enumerate(tags):
                if tag.startswith('B-'):
                    if current:
                        entities.append(current)
                    current = {'type': tag[2:], 'start': i, 'text': ''}
                elif tag.startswith('I-') and current and tag[2:] == current['type']:
                    continue
                else:
                    if current:
                        entities.append(current)
                        current = None

            inference_time = time.time() - start_time

            if entities:
                print(f"  Detected Entities: {entities}")
                for ent in entities:
                    all_entities.append({'category': case['category'], 'text': text, 'entity': ent})
            else:
                print(f"  No entities detected")
            print(f"  Time: {inference_time*1000:.1f}ms")

            inference_log.append({
                'category': case['category'],
                'text': text,
                'num_entities': len(entities),
                'time_ms': inference_time * 1000
            })

    # Save log
    with open(os.path.join(output_dir, 'real_text_inference.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'text', 'num_entities', 'time_ms'])
        writer.writeheader()
        writer.writerows(inference_log)

    return all_entities, inference_log

def plot_long_tail_results(long_tail_results, save_dir):
    """绘制长尾分析结果图表（英文）"""
    if plt_lib is None or not long_tail_results:
        return
    plt = plt_lib
    
    # 图1: 正则分类结果
    if 'regex_based' in long_tail_results and long_tail_results['regex_based'].get('results'):
        regex_data = long_tail_results['regex_based']['results']
        categories = list(regex_data.keys())
        precisions = [regex_data[c]['precision'] * 100 for c in categories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(categories, precisions, color='#3498db', alpha=0.8)
        
        for bar, val in zip(bars, precisions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Geographic Entity Category', fontsize=12)
        ax.set_ylabel('Recognition Precision (%)', fontsize=12)
        ax.set_title('Long-tail Entity Recognition - Regex-based Classification', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=25, ha='right')
        
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(save_dir, f'longtail_regex.{ext}'), dpi=300, bbox_inches='tight', format=ext)
        plt.close()
    
    # 图2: 频率分组结果
    if 'frequency_based' in long_tail_results and long_tail_results['frequency_based'].get('results'):
        freq_data = long_tail_results['frequency_based']['results']
        levels = list(freq_data.keys())
        precisions = [freq_data[l]['precision'] * 100 for l in levels]
        counts = [freq_data[l]['total_entities'] for l in levels]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 精确率对比
        bars1 = axes[0].bar(levels, precisions, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        for bar, val in zip(bars1, precisions):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        axes[0].set_xlabel('Entity Frequency Level', fontsize=12)
        axes[0].set_ylabel('Recognition Precision (%)', fontsize=12)
        axes[0].set_title('Recognition by Training Frequency', fontsize=13, fontweight='bold')
        axes[0].set_ylim(0, 105)
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        
        # 实体数量分布
        bars2 = axes[1].bar(levels, counts, color=['#27ae60', '#f1c40f', '#c0392b'], alpha=0.8)
        for bar, val in zip(bars2, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                         f'{val}', ha='center', va='bottom', fontsize=10)
        axes[1].set_xlabel('Entity Frequency Level', fontsize=12)
        axes[1].set_ylabel('Number of Entities', fontsize=12)
        axes[1].set_title('Entity Distribution by Frequency', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(os.path.join(save_dir, f'longtail_frequency.{ext}'), dpi=300, bbox_inches='tight', format=ext)
        plt.close()


def test_real_world_text(model, tokenizer, id2label, label2id, cfg, model_name):
    """
    测试真实网络文本的地理实体识别能力
    
    Args:
        model: 训练好的NER模型
        tokenizer: 分词器
        id2label: 标签ID到标签名的映射
        label2id: 标签名到ID的映射
        cfg: 配置对象
        model_name: 模型名称
    
    Returns:
        list of dict: 每个文本的识别结果
    """
    # 真实网络文本测试用例（来自新闻、社交媒体等）
    test_texts = [
        {
            'category': 'News',
            'text': '据新华社报道，2024年5月1日上午，习近平总书记来到河北省雄安新区考察调研，强调要高标准高质量推进雄安新区建设。'
        },
        {
            'category': 'Social Media',
            'text': '刚从上海浦东国际机场落地，准备坐地铁去外滩看夜景，听说陆家嘴的灯光秀特别漂亮！'
        },
        {
            'category': 'Travel Blog',
            'text': '云南自驾游攻略：从昆明出发，途经大理古城、丽江玉龙雪山，最后到达香格里拉，全程风景如画！'
        },
        {
            'category': 'Weather Report',
            'text': '中央气象台发布暴雨预警：未来三天，广东珠三角地区、广西南宁、海南海口等地将出现大到暴雨。'
        },
        {
            'category': 'News',
            'text': '港珠澳大桥开通五年来，累计通行车辆超过3000万辆次，成为连接香港、珠海、澳门的重要纽带。'
        },
        {
            'category': 'Social Media',
            'text': '周末去了一趟杭州西湖，在断桥残雪拍了很多照片，还去了灵隐寺祈福，人真的太多了！'
        },
        {
            'category': 'News',
            'text': '川藏铁路雅安至林芝段即将通车，这条穿越青藏高原的天路将极大促进西藏地区的经济发展。'
        },
        {
            'category': 'Travel Blog',
            'text': '新疆美食之旅：乌鲁木齐的大盘鸡、喀什的烤包子、吐鲁番的葡萄，每一口都是满满的幸福感！'
        },
        {
            'category': 'News',
            'text': '粤港澳大湾区建设持续推进，深圳前海自贸区、广州南沙新区、珠海横琴新区协同发展成效显著。'
        },
        {
            'category': 'Social Media',
            'text': '今天从成都双流机场出发，经停西安咸阳机场，最终到达北京首都机场，长途飞行真的好累啊！'
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"Real-world Text Test - {model_name}")
    print(f"{'='*60}")
    
    results = []
    
    for item in test_texts:
        text = item['text']
        category = item['category']
        
        print(f"\n[{category}]")
        print(f"Text: {text}")
        
        start_time = time.time()
        
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=cfg.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(cfg.device)
        attention_mask = encoding['attention_mask'].to(cfg.device)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()
        
        # Inference
        with torch.no_grad():
            predictions, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 处理预测结果
        if isinstance(predictions, list):
            pred_tags = predictions[0] if len(predictions) > 0 else []
            if isinstance(pred_tags, torch.Tensor):
                pred_tags = pred_tags.cpu().numpy()
        else:
            pred_tags = predictions[0].cpu().numpy()
        
        tags = [id2label.get(p, 'O') if isinstance(p, (int, np.integer)) else 'O' for p in pred_tags]
        
        # 提取实体
        entities = []
        current_entity = None
        
        def extract_entity_text(entity):
            start_char = offset_mapping[entity['start']][0]
            end_idx = entity['end'] - 1
            while end_idx >= entity['start'] and offset_mapping[end_idx][1] == 0:
                end_idx -= 1
            if end_idx >= entity['start']:
                end_char = offset_mapping[end_idx][1]
            else:
                end_char = start_char + 1
            return text[start_char:end_char]
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_entity:
                    current_entity['text'] = extract_entity_text(current_entity)
                    entities.append(current_entity)
                current_entity = {
                    'text': '',
                    'type': tag[2:],
                    'start': i,
                    'end': i + 1
                }
            elif tag.startswith('I-') and current_entity:
                current_entity['end'] = i + 1
            else:
                if current_entity:
                    current_entity['text'] = extract_entity_text(current_entity)
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            current_entity['text'] = extract_entity_text(current_entity)
            entities.append(current_entity)
        
        inference_time = (time.time() - start_time) * 1000
        
        # 打印识别结果
        if entities:
            print(f"Detected Entities:")
            for ent in entities:
                print(f"  - {ent['text']} ({ent['type']})")
        else:
            print(f"Detected Entities: None")
        print(f"Inference Time: {inference_time:.1f}ms")
        
        results.append({
            'category': category,
            'text': text,
            'entities': entities,
            'time_ms': inference_time
        })
    
    return results


def run_geographic_tests(cfg, ablation_models, train_data, lexicon, device, lexicon_matcher=None):
    """
    运行地理理解测试模块
    对消融实验的模型进行GeoGLUE评测和长尾分析（支持BiLSTM组和BiGRU组）
    
    Args:
        lexicon_matcher: 已有的 LexiconMatcher 实例，用于复用避免重复加载
    """
    print(f"\n{'#'*80}")
    print("# Geographic Understanding Tests Module")
    print(f"{'#'*80}")
    
    geoglue_dir = os.path.join(cfg.data_dir, 'GeoGLUE')
    
    # 检查GeoGLUE数据是否存在
    if not os.path.exists(geoglue_dir):
        print(f"Warning: GeoGLUE data directory not found: {geoglue_dir}")
        print("Skipping GeoGLUE evaluation...")
        geoglue_available = False
    else:
        geoglue_available = True
    
    # 初始化评测器
    if geoglue_available:
        geoglue_evaluator = GeoGLUEEvaluator(geoglue_dir)
    
    all_geoglue_results = {}
    all_longtail_results = {}
    
    # 分组处理
    groups = {}
    for model_config in ablation_models:
        group = model_config.get('group', 'Unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(model_config)
    
    # 对每个组的每个模型进行测试
    for group_name, group_models in groups.items():
        print(f"\n{'='*70}")
        print(f"Group: {group_name}")
        print(f"{'='*70}")
        
        for model_config in group_models:
            model_name = model_config['name']
            model_save_path = os.path.join(cfg.model_save_dir, 
                                            f"ner_model_{model_name.replace('/', '_').replace('+', '_')}.pt")
            
            print(f"\n{'-'*60}")
            print(f"Testing Model: {model_name}")
            print(f"{'-'*60}")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_save_path):
                print(f"Warning: Model file not found: {model_save_path}")
                print("Please train the model first using ABLATION mode.")
                continue
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], local_files_only=True)
            
            # 初始化模型
            use_lexicon = model_config['use_lexicon']
            lexicon_type = model_config.get('lexicon_type', 'static')
            model_lexicon_matcher = lexicon_matcher if (use_lexicon and lexicon_matcher) else None
            if use_lexicon and not lexicon_matcher:
                print(f"  Creating new LexiconMatcher for {model_name}")
                model_lexicon_matcher = create_lexicon_matcher(cfg)
            
            model = NERModel(
                model_name=model_config['model_name'],
                num_labels=len(cfg.ner_label2id),
                model_variant=model_name,
                use_crf=model_config['use_crf'],
                use_lexicon=use_lexicon,
                lexicon_size=cfg.lexicon_size,
                lexicon_type=lexicon_type,
                use_reshaping=False,
                lexicon_config={
                    'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout,
                    'num_levels': cfg.num_geo_levels
                }
            )
            model.to(device)
            
            # 加载训练好的权重（strict=False允许加载不匹配的参数，兼容不同版本的模型结构）
            state_dict = torch.load(model_save_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"  Warning: Unexpected keys in state_dict: {unexpected_keys}")
            model.eval()
            print(f"Loaded model from: {model_save_path}")
            
            # ==================== GeoGLUE评测 ====================
            if geoglue_available:
                print("\n--- GeoGLUE Evaluation ---")
                try:
                    geoglue_results = geoglue_evaluator.evaluate_all(model, tokenizer, model_lexicon_matcher, cfg)
                    all_geoglue_results[model_name] = geoglue_results
                    
                    # 保存CSV
                    csv_path = os.path.join(cfg.result_dir, f'geoglue_{model_name.replace("+", "_").replace("/", "_")}.csv')
                    save_geoglue_results(geoglue_results, csv_path)
                except Exception as e:
                    print(f"GeoGLUE evaluation failed: {e}")
                    all_geoglue_results[model_name] = {}
            
            # ==================== 长尾地名分析 ====================
            print("\n--- Long-tail Entity Analysis ---")
            try:
                longtail_results = evaluate_long_tail_entities(
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    train_data=train_data,
                    lexicon=lexicon,
                    id2label=cfg.ner_id2label
                )
                all_longtail_results[model_name] = longtail_results
                
                # 保存CSV
                csv_path = os.path.join(cfg.result_dir, f'longtail_{model_name.replace("+", "_").replace("/", "_")}.csv')
                save_long_tail_results(longtail_results, csv_path)
                
                # 绘制图表
                plot_long_tail_results(longtail_results, cfg.figure_dir)
            except Exception as e:
                print(f"Long-tail analysis failed: {e}")
                all_longtail_results[model_name] = {}
            
            # ==================== 网络文本识别测试 ====================
            print("\n--- Real-world Text Geo-Entity Recognition Test ---")
            try:
                real_text_results = test_real_world_text(
                    model=model,
                    tokenizer=tokenizer,
                    id2label=cfg.ner_id2label,
                    label2id=cfg.ner_label2id,
                    cfg=cfg,
                    model_name=model_name
                )
                
                # 保存识别结果
                text_results_path = os.path.join(cfg.result_dir, f'real_text_test_{model_name.replace("+", "_").replace("/", "_")}.csv')
                with open(text_results_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Text', 'Detected_Entities', 'Inference_Time_ms'])
                    for item in real_text_results:
                        entities_str = '; '.join([f"{e['text']}({e['type']})" for e in item['entities']])
                        writer.writerow([item['text'], entities_str, item['time_ms']])
                print(f"Real-text recognition results saved to: {text_results_path}")
                
            except Exception as e:
                print(f"Real-world text test failed: {e}")
            
            # 清理显存
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # ==================== 汇总结果 ====================
    print(f"\n{'#'*80}")
    print("# Geographic Tests Summary")
    print(f"{'#'*80}")
    
    # GeoGLUE汇总
    if all_geoglue_results:
        print("\n" + "="*90)
        print("GeoGLUE 地名识别评测汇总 (百分制 / 100分制)")
        print("="*90)
        
        # 汇总表
        summary_data = []
        for model_name, results in all_geoglue_results.items():
            row = {'Model': model_name}
            
            if 'GeoETA' in results:
                row['GeoETA_MicroF1'] = results['GeoETA'].get('micro_f1', 0)
                row['GeoETA_MacroF1'] = results['GeoETA'].get('macro_f1', 0)
            
            if 'GeoCPA' in results:
                row['GeoCPA_MicroF1'] = results['GeoCPA'].get('micro_f1', 0)
                row['GeoCPA_MacroF1'] = results['GeoCPA'].get('macro_f1', 0)
            
            if 'GeoWWC' in results:
                row['GeoWWC_MicroF1'] = results['GeoWWC'].get('micro_f1', 0)
                row['GeoWWC_Accuracy'] = results['GeoWWC'].get('accuracy', 0)
            
            if '_summary' in results:
                row['Avg_MicroF1'] = results['_summary'].get('avg_micro_f1', 0)
                row['Overall_Score'] = results['_summary'].get('overall_score', 0)
            
            summary_data.append(row)
        
        # 按分组打印（百分制显示）
        for group_name in ['BiLSTM', 'BiGRU']:
            group_rows = [r for r in summary_data if group_name in r['Model']]
            if group_rows:
                print(f"\n[{group_name} Group]")
                print(f"{'Model':<42} {'GeoETA':>8} {'GeoCPA':>8} {'GeoWWC':>8} {'综合分':>8} {'等级':>6}")
                print(f"{'':<42} {'(Micro)':>8} {'(Micro)':>8} {'(Micro)':>8} {'(/100)':>8} {'':>6}")
                print("-" * 90)
                
                for row in group_rows:
                    model_short = row['Model'][:39] + "..." if len(row['Model']) > 39 else row['Model']
                    eta_micro = row.get('GeoETA_MicroF1', 0) * 100
                    cpa_micro = row.get('GeoCPA_MicroF1', 0) * 100
                    wwc_micro = row.get('GeoWWC_MicroF1', 0) * 100
                    overall = row.get('Overall_Score', row.get('Avg_MicroF1', 0) * 100)
                    
                    # 评分等级
                    if overall >= 90:
                        grade = "A"
                    elif overall >= 80:
                        grade = "B"
                    elif overall >= 70:
                        grade = "C"
                    elif overall >= 60:
                        grade = "D"
                    else:
                        grade = "F"
                    
                    print(f"{model_short:<42} {eta_micro:>7.1f} {cpa_micro:>8.1f} {wwc_micro:>8.1f} {overall:>7.1f} {grade:>6}")
        
        # 总体排名（按综合分排序）
        print(f"\n{'='*90}")
        print("综合得分排名（按百分制综合分降序）:")
        print("-" * 90)
        ranked = sorted(summary_data, key=lambda x: x.get('Overall_Score', x.get('Avg_MicroF1', 0)*100), reverse=True)
        for rank, row in enumerate(ranked, 1):
            overall = row.get('Overall_Score', row.get('Avg_MicroF1', 0) * 100)
            if overall >= 90:
                grade = "A"
            elif overall >= 80:
                grade = "B"
            elif overall >= 70:
                grade = "C"
            elif overall >= 60:
                grade = "D"
            else:
                grade = "F"
            model_short = row['Model'][:50] + "..." if len(row['Model']) > 50 else row['Model']
            print(f"  #{rank} {model_short:<55} {overall:>6.1f}分  [{grade}]")
        print("="*90)
        
        # 保存汇总CSV（同时保存小数和百分制）
        summary_path = os.path.join(cfg.result_dir, 'geographic_tests_summary.csv')
        if summary_data:
            import pandas as pd
            # 构建百分制版本的汇总数据
            summary_pct = []
            for row in summary_data:
                pct_row = {'Model': row['Model']}
                if 'GeoETA_MicroF1' in row:
                    pct_row['GeoETA_MicroF1(%)'] = round(row['GeoETA_MicroF1'] * 100, 2)
                    pct_row['GeoETA_MacroF1(%)'] = round(row['GeoETA_MacroF1'] * 100, 2)
                if 'GeoCPA_MicroF1' in row:
                    pct_row['GeoCPA_MicroF1(%)'] = round(row['GeoCPA_MicroF1'] * 100, 2)
                    pct_row['GeoCPA_MacroF1(%)'] = round(row['GeoCPA_MacroF1'] * 100, 2)
                if 'GeoWWC_MicroF1' in row:
                    pct_row['GeoWWC_MicroF1(%)'] = round(row['GeoWWC_MicroF1'] * 100, 2)
                    pct_row['GeoWWC_Accuracy(%)'] = round(row['GeoWWC_Accuracy'] * 100, 2)
                if 'Overall_Score' in row:
                    pct_row['Overall_Score(%)'] = round(row['Overall_Score'], 2)
                elif 'Avg_MicroF1' in row:
                    pct_row['Overall_Score(%)'] = round(row['Avg_MicroF1'] * 100, 2)
                summary_pct.append(pct_row)
            
            df = pd.DataFrame(summary_pct)
            df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f"\nSummary saved to: {summary_path}")
        
        # 绘制汇总图表
        if plt_lib is not None and len(summary_data) >= 2:
            plt = plt_lib
            
            # 图1: GeoGLUE任务对比（两组对比）
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 准备数据
            models = [row['Model'] for row in summary_data]
            model_short = []
            for m in models:
                short = m.replace('RoBERTa', '')
                short = short.replace('AttentionCRF', '')
                short = short.replace('+Lexicon', '+Lex')
                short = short.replace('+DynamicLexicon', '+DLex')
                model_short.append(short)
            
            eta_micros = [row.get('GeoETA_MicroF1', 0) * 100 for row in summary_data]
            eta_macros = [row.get('GeoETA_MacroF1', 0) * 100 for row in summary_data]
            tes_recalls = [row.get('TESrecall_Entity', 0) * 100 for row in summary_data]
            rerank_mrrs = [row.get('TESrerank_MRR', 0) * 100 for row in summary_data]
            
            x = np.arange(len(models))
            width = 0.35
            
            # GeoETA对比
            axes[0].bar(x - width/2, eta_micros, width, label='Micro-F1', color='#3498db')
            axes[0].bar(x + width/2, eta_macros, width, label='Macro-F1', color='#e74c3c')
            axes[0].set_xlabel('Model', fontsize=11)
            axes[0].set_ylabel('F1 Score (%)', fontsize=11)
            axes[0].set_title('GeoETA Performance Comparison', fontsize=13, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_short, rotation=25, ha='right', fontsize=8)
            axes[0].legend()
            axes[0].grid(axis='y', linestyle='--', alpha=0.5)
            axes[0].set_ylim(0, 100)
            
            # 添加分组分隔线
            if len(models) == 6:
                axes[0].axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
                axes[1].axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # TES对比
            axes[1].bar(x - width/2, tes_recalls, width, label='TES-Recall', color='#27ae60')
            axes[1].bar(x + width/2, rerank_mrrs, width, label='TES-Rerank MRR', color='#9b59b6')
            axes[1].set_xlabel('Model', fontsize=11)
            axes[1].set_ylabel('Score (%)', fontsize=11)
            axes[1].set_title('GeoGeneralization Performance', fontsize=13, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(model_short, rotation=25, ha='right', fontsize=8)
            axes[1].legend()
            axes[1].grid(axis='y', linestyle='--', alpha=0.5)
            axes[1].set_ylim(0, 100)
            
            plt.tight_layout()
            for ext in ['pdf', 'png']:
                plt.savefig(os.path.join(cfg.figure_dir, f'geographic_tests_summary.{ext}'), 
                           dpi=300, bbox_inches='tight', format=ext)
            plt.close()
            print(f"Summary charts saved to: {cfg.figure_dir}")
    
    # 长尾分析汇总
    if all_longtail_results:
        print("\nLong-tail Entity Analysis Summary:")
        print("-" * 70)
        
        for group_name in ['BiLSTM', 'BiGRU']:
            group_models = [m for m in all_longtail_results.keys() if group_name in m]
            if group_models:
                print(f"\n[{group_name} Group]")
                for model_name in group_models:
                    results = all_longtail_results[model_name]
                    print(f"  Model: {model_name}")
                    
                    # 频率分组结果
                    if 'frequency_based' in results and 'results' in results['frequency_based']:
                        print("    By Training Frequency:")
                        freq_results = results['frequency_based']['results']
                        for level, data in freq_results.items():
                            level_name = level.replace('_', ' ').title()
                            precision = data['precision'] * 100
                            total = data['total_entities']
                            print(f"      - {level_name}: {precision:.1f}% (Total: {total} entities)")
    
    print(f"\n{'#'*80}")
    print("Geographic Tests Completed!")
    print(f"Results saved to: {cfg.result_dir}")
    print(f"Charts saved to: {cfg.figure_dir}")
    print(f"{'#'*80}")
    
    return all_geoglue_results, all_longtail_results


# ========================== Main Experiment Function ==========================
def run_ner_experiments(cfg, model_configs=None, run_ablation=False, do_real_text_test=True, do_geoglue=True, sample_size=None):
    print(f"\n{'='*80}")
    print("Starting Enhanced NER Experiment with Geo Knowledge")
    print(f"{'='*80}")

    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n[Pre-Experiment] GPU Memory Cleared")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  - Allocated: {allocated:.1f} MB")
        print(f"  - Reserved: {reserved:.1f} MB")

    # 根据模式选择模型列表
    if run_ablation:
        model_configs = cfg.ablation_models
        print("\n[Ablation Mode] Running 6 model variants (2 groups: BiLSTM + BiGRU)")
    else:
        model_configs = model_configs or cfg.ner_experiment_models
        print(f"\n[Experiment Mode] Running {len(model_configs)} model variants")

    # Load Geo Knowledge Graph (from cache for faster loading)
    # Note: Run preprocess_data.py first to generate/update the cache
    print("\nLoading Geographic Knowledge Graph (from cache)...")
    load_ownthink = getattr(cfg, 'load_ownthink', False)
    print(f"  - load_ownthink: {load_ownthink}")
    kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink, skip_cache=False)
    print(f"  - Lexicon: {len(kg.lexicon)} entries")
    print(f"  - Aliases: {len(kg.alias_map)} entries")
    print(f"  - Hierarchy: {len(kg.hierarchy_map)} entries")
    print(f"  - Coordinates: {len(kg.coord_map)} entries")

    # Create global LexiconMatcher for all models (reuse kg instance)
    print("\nCreating global LexiconMatcher...")
    global_lexicon_matcher = create_lexicon_matcher(cfg, load_ownthink=load_ownthink, kg_instance=kg)

    # Load NER datasets (with optional sampling for test mode)
    if sample_size:
        print(f"\n[TEST MODE] Sampling {sample_size} entries per dataset...")
        train_data, dev_data, test_datasets = load_ner_datasets(sample_size=sample_size, use_dev_as_test=False)
    else:
        train_data, dev_data, test_datasets = load_ner_datasets(sample_size=None, use_dev_as_test=False)
    
    print(f"\nDataset Statistics:")
    print(f"  Training: {len(train_data)}, Dev: {len(dev_data)}")
    for name, data in test_datasets.items():
        print(f"  {name}: {len(data)}")

    lexicon = load_lexicon(cfg.geo_lexicon_path)
    all_results = []
    best_model_info = None

    for model_config in model_configs:
        model_name = model_config['name']
        use_lexicon = model_config['use_lexicon']
        lexicon_matcher = global_lexicon_matcher if use_lexicon else None

        print(f"\n{'#'*80}")
        print(f"### Model: {model_name} ###")
        print(f"{'#'*80}")

        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], local_files_only=True)

        train_dataset = MultiTaskDataset(train_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)

        history_path = os.path.join(cfg.log_dir, f'history_{model_name.replace("/", "_").replace("+", "_")}.csv')
        trained_model, best_dev_f1, _ = train_single_ner_model(model_config, train_dataset, dev_dataset, tokenizer, cfg, history_path)

        results = {'model_name': model_name, 'best_dev_f1': best_dev_f1}
        all_test_f1 = []

        for dataset_name, test_data in test_datasets.items():
            test_dataset = MultiTaskDataset(test_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.ner_eval_batch_size, shuffle=False)
            eval_results = evaluate_ner_model(trained_model, test_loader, cfg.ner_id2label, cfg.device, use_crf=model_config['use_crf'])

            results[f'{dataset_name}_micro_f1'] = eval_results['micro_f1']
            results[f'{dataset_name}_macro_f1'] = eval_results['macro_f1']
            all_test_f1.append(eval_results['micro_f1'])

            print(f"\n{dataset_name} Results:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}, Macro-F1: {eval_results['macro_f1']:.4f}")

            del test_dataset, test_loader

        results['avg_micro_f1'] = np.mean(all_test_f1)
        all_results.append(results)

        if not best_model_info or results['avg_micro_f1'] > best_model_info['avg_f1']:
            # 释放旧的最佳模型
            if best_model_info and 'model' in best_model_info:
                old_model = best_model_info.pop('model')
                del old_model

            # 新的最佳模型移至CPU保存，释放GPU显存
            trained_model.cpu()
            best_model_info = {
                'model': trained_model,
                'tokenizer': tokenizer,
                'lexicon_matcher': lexicon_matcher,
                'name': model_name,
                'avg_f1': results['avg_micro_f1']
            }
            print(f"  [Best model saved to CPU: {model_name}]")
        else:
            del trained_model

        # 清理数据集和数据加载器
        del train_dataset, dev_dataset

        # 强制垃圾回收和GPU缓存清理
        gc.collect()
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 打印GPU内存状态
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Save results
    if run_ablation:
        csv_path = os.path.join(cfg.result_dir, 'ablation_ner_results.csv')
    else:
        csv_path = os.path.join(cfg.result_dir, 'ner_results.csv')
    save_results_to_csv(all_results, csv_path)

    # Plot charts
    plot_model_comparison_bar(all_results, cfg.figure_dir)
    if run_ablation:
        plot_ablation_study(all_results, cfg.figure_dir)

    # GeoGLUE evaluation
    if do_geoglue and best_model_info:
        print(f"\n{'='*80}")
        print("GeoGLUE Evaluation")
        print(f"{'='*80}")
        
        # 将最佳模型移回GPU进行评估
        best_model_info['model'].to(cfg.device)
        gc.collect()
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        geoglue_dir = os.path.join(cfg.data_dir, 'GeoGLUE')
        if os.path.exists(geoglue_dir):
            try:
                geoglue_evaluator = GeoGLUEEvaluator(geoglue_dir)
                geoglue_results = geoglue_evaluator.evaluate_all(
                    best_model_info['model'],
                    best_model_info['tokenizer'],
                    best_model_info['lexicon_matcher'],
                    cfg
                )
                
                save_geoglue_results(geoglue_results, os.path.join(cfg.result_dir, 'geoglue_results.csv'))
                plot_geoglue_results(geoglue_results, cfg.figure_dir)
            except Exception as e:
                print(f"GeoGLUE evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"GeoGLUE data directory not found: {geoglue_dir}")
            print("Skipping GeoGLUE evaluation.")

    # Real-world text test
    if do_real_text_test and best_model_info:
        try:
            test_on_real_text(
                best_model_info['model'],
                best_model_info['tokenizer'],
                best_model_info['lexicon_matcher'],
                cfg.ner_id2label,
                cfg.ner_label2id,
                cfg,
                best_model_info['name'],
                cfg.result_dir
            )
        except Exception as e:
            print(f"Real-world text test failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")
    print(f"{'Model':<38} {'CLUENER':^10} {'MSRA':^10} {'Weibo':^10} {'CMNER':^10} {'Avg':^10}")
    print(f"{'-'*90}")
    for result in all_results:
        cluener = result.get('cluener_micro_f1', 0) * 100
        msra = result.get('msra_micro_f1', 0) * 100
        weibo = result.get('weibo_ner_micro_f1', 0) * 100
        cmner = result.get('cmner_micro_f1', 0) * 100
        avg = result.get('avg_micro_f1', 0) * 100
        print(f"{result['model_name']:<38} {cluener:^10.2f} {msra:^10.2f} {weibo:^10.2f} {cmner:^10.2f} {avg:^10.2f}")

    return all_results

# ========================== Ours-StaticLex 消融实验编排 ==========================

def _load_per_dataset_dev(cfg):
    """加载各数据集的验证集（用于按数据集评估 F1）"""
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    datasets = ['cluener', 'msra', 'weibo_ner', 'cmner']
    dev_datasets = {}
    for ds_name in datasets:
        dev_path = os.path.join(processed_dir, ds_name, 'dev.json')
        if os.path.exists(dev_path):
            with open(dev_path, 'r', encoding='utf-8') as f:
                dev_data = [json.loads(line) for line in f if line.strip()]
            dev_datasets[ds_name] = dev_data
            print(f"  {ds_name} dev: {len(dev_data)} samples")
    return dev_datasets


def _create_staticlex_model(cfg, model_config):
    """根据 model_config 创建 StaticLex 模型（用于加载检查点评估）"""
    model = NERModel(
        model_name=model_config['model_name'],
        num_labels=len(cfg.ner_label2id),
        model_variant=model_config['name'],
        use_crf=model_config['use_crf'],
        use_lexicon=model_config['use_lexicon'],
        lexicon_size=cfg.lexicon_size,
        lexicon_type=model_config.get('lexicon_type', 'static'),
        use_reshaping=False,
        ablation_config=model_config.get('ablation_config'),
        lexicon_config={
            'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels
        }
    )
    model.to(cfg.device)
    return model


def _evaluate_model_on_datasets(model, cfg, tokenizer, lexicon_matcher, dev_datasets):
    """在多个数据集的验证集上评估模型，返回各数据集的 Micro-F1"""
    from sklearn.metrics import f1_score
    results = {}
    f1_values = []

    model.eval()
    with torch.no_grad():
        for ds_name, dev_data in dev_datasets.items():
            dev_dataset = MultiTaskDataset(dev_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)
            dev_loader = create_dataloader(dev_dataset, batch_size=cfg.ner_eval_batch_size, shuffle=False)

            all_preds = []
            all_labels = []

            for batch in tqdm(dev_loader, desc=f"Eval {ds_name}", leave=False, ncols=60):
                input_ids = batch['input_ids'].to(cfg.device)
                attention_mask = batch['attention_mask'].to(cfg.device)
                labels = batch['labels'].to(cfg.device)

                lexicon_indices = batch.get('lexicon_indices', None)
                lexicon_levels = batch.get('lexicon_levels', None)
                lexicon_types = batch.get('lexicon_types', None)
                lexicon_coords = batch.get('lexicon_coords', None)

                if lexicon_indices is not None:
                    lexicon_indices = lexicon_indices.to(cfg.device)
                if lexicon_levels is not None:
                    lexicon_levels = lexicon_levels.to(cfg.device)
                if lexicon_types is not None:
                    lexicon_types = lexicon_types.to(cfg.device)
                if lexicon_coords is not None:
                    lexicon_coords = lexicon_coords.to(cfg.device)

                # fp32 forward + decode（评估不用 AMP，避免 CRF 精度问题）
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lexicon_indices=lexicon_indices,
                    lexicon_levels=lexicon_levels,
                    lexicon_types=lexicon_types,
                    lexicon_coords=lexicon_coords,
                )

                # 统一 predictions 格式为 list of array
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                if isinstance(predictions, np.ndarray):
                    if predictions.ndim == 1:
                        predictions = [predictions]
                    else:
                        predictions = [predictions[i] for i in range(len(predictions))]

                labels_np = labels.cpu().numpy()
                for i in range(len(predictions)):
                    pred_seq = predictions[i]
                    label_seq = labels_np[i]
                    if isinstance(pred_seq, (int, np.integer)):
                        pred_seq = [pred_seq]
                    min_len = min(len(pred_seq), len(label_seq))
                    for j in range(min_len):
                        if label_seq[j] != -100:
                            all_preds.append(int(pred_seq[j]))
                            all_labels.append(int(label_seq[j]))

            if len(all_preds) > 0 and len(all_labels) > 0:
                micro_f1 = f1_score(all_labels, all_preds, average='micro')
            else:
                micro_f1 = 0.0
            results[f'{ds_name}_micro_f1'] = micro_f1
            f1_values.append(micro_f1)
            print(f"    {ds_name}: Micro-F1 = {micro_f1:.4f}")

            del dev_dataset, dev_loader

    avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    results['avg_micro_f1'] = avg_f1
    print(f"    Average: Micro-F1 = {avg_f1:.4f}")
    return results


def train_staticlex_ablation_variants(cfg, train_data, dev_data, tokenizer, lexicon_matcher):
    """
    训练 6 个 Ours-StaticLex 消融变体并按数据集评估 F1。
    - baseline (Ours-StaticLex): 复用现有检查点，不重训，仅加载评估
    - 5 个变体: 从头训练 30 轮（不加载检查点），训练后按数据集评估
    """
    print(f"\n{'#'*80}")
    print(f"# Ours-StaticLex Ablation Experiment (6 models)")
    print(f"{'#'*80}")

    # 加载各数据集验证集
    print("\n[StaticLex] Loading per-dataset dev sets for evaluation...")
    dev_datasets = _load_per_dataset_dev(cfg)

    all_results = []
    for model_config in cfg.ablation_models_staticlex:
        model_name = model_config['name']

        # 主模型 Ours-StaticLex：如果检查点已存在则跳过训练，仅加载评估
        ckpt_path = os.path.join(cfg.model_save_dir, f'ner_model_{model_name.replace("/", "_").replace("+", "_")}.pt')
        if model_name == 'Ours-StaticLex' and os.path.exists(ckpt_path):
            print(f"\n{'='*70}")
            print(f"[StaticLex] SKIP TRAINING {model_name} (checkpoint exists)")
            print(f"[StaticLex] Loading checkpoint and evaluating on per-dataset dev sets...")
            print(f"{'='*70}")

            try:
                model = _create_staticlex_model(cfg, model_config)
                state = torch.load(ckpt_path, map_location=cfg.device)
                model.load_state_dict(state, strict=False)
                model.eval()

                # 按数据集评估
                ds_results = _evaluate_model_on_datasets(model, cfg, tokenizer, lexicon_matcher, dev_datasets)
                result = {'model_name': model_name, 'best_dev_f1': ds_results.get('avg_micro_f1', 0.0)}
                result.update(ds_results)
                all_results.append(result)
                print(f"[StaticLex] {model_name} evaluation done, avg_f1={ds_results.get('avg_micro_f1', 0.0):.4f}")

                del model
                gc.collect()
                if cfg.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"[StaticLex] ERROR evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({'model_name': model_name, 'best_dev_f1': 0.0, 'error': str(e)})
            continue

        # 5 个消融变体：从头训练
        print(f"\n{'='*70}")
        print(f"[StaticLex] Training from scratch: {model_name}")
        print(f"  ablation_config={model_config.get('ablation_config')}")
        print(f"  epochs={model_config.get('num_epochs_override') or cfg.ner_num_epochs}")
        print(f"{'='*70}")

        train_dataset = MultiTaskDataset(train_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer, max_seq_length=cfg.max_seq_length, lexicon_matcher=lexicon_matcher)
        history_path = os.path.join(cfg.log_dir, f'history_staticlex_{model_name.replace("/", "_").replace("+", "_")}.csv')

        try:
            trained_model, best_dev_f1, _ = train_single_ner_model(
                model_config, train_dataset, dev_dataset, tokenizer, cfg, history_path
            )

            # 训练完成后，按数据集评估
            print(f"\n  [StaticLex] Evaluating {model_name} on per-dataset dev sets...")
            ds_results = _evaluate_model_on_datasets(trained_model, cfg, tokenizer, lexicon_matcher, dev_datasets)

            result = {'model_name': model_name, 'best_dev_f1': best_dev_f1}
            result.update(ds_results)
            all_results.append(result)
            print(f"[StaticLex] {model_name} done, dev_f1={best_dev_f1:.4f}, avg_ds_f1={ds_results.get('avg_micro_f1', 0.0):.4f}")
        except FileNotFoundError as e:
            print(f"[StaticLex] SKIP {model_name}: {e}")
            all_results.append({'model_name': model_name, 'best_dev_f1': 0.0, 'error': str(e)})
        except Exception as e:
            print(f"[StaticLex] ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'model_name': model_name, 'best_dev_f1': 0.0, 'error': str(e)})

        # 清理
        try:
            del trained_model
        except NameError:
            pass
        del train_dataset, dev_dataset
        gc.collect()
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # 保存训练结果汇总
    csv_path = os.path.join(cfg.result_dir, 'staticlex_ablation_results.csv')
    save_results_to_csv(all_results, csv_path)

    # 按数据集显示 F1 汇总
    print(f"\n{'='*90}")
    print(f"[StaticLex] Ablation Results Summary (Per-Dataset Micro-F1)")
    print(f"{'Model':<35} {'CLUENER':^10} {'MSRA':^10} {'Weibo':^10} {'CMNER':^10} {'Avg':^10}")
    print(f"{'-'*90}")
    for r in all_results:
        cluener = r.get('cluener_micro_f1', 0) * 100
        msra = r.get('msra_micro_f1', 0) * 100
        weibo = r.get('weibo_ner_micro_f1', 0) * 100
        cmner = r.get('cmner_micro_f1', 0) * 100
        avg = r.get('avg_micro_f1', 0) * 100
        print(f"{r['model_name']:<35} {cluener:^10.2f} {msra:^10.2f} {weibo:^10.2f} {cmner:^10.2f} {avg:^10.2f}")
    print(f"{'='*90}")

    return all_results


def _load_network_text_cases(cfg):
    """加载 25 条网络文本测试用例"""
    with open(cfg.network_text_cases_path, 'r', encoding='utf-8-sig') as f:
        cases = json.load(f)
    return cases


def _load_network_text_ground_truth(cfg):
    """加载网络文本真值标注"""
    with open(cfg.network_text_ground_truth_path, 'r', encoding='utf-8-sig') as f:
        gt = json.load(f)
    # 去除 _meta 字段
    gt.pop('_meta', None)
    return gt


def _extract_entities_from_pred(tags, text, offset_mapping, id2label):
    """从 BIO 标签序列提取实体（带 offset 映射）"""
    entities = []
    current = None
    for i, tag_id in enumerate(tags):
        tag = id2label.get(int(tag_id), 'O') if isinstance(tag_id, (int, np.integer, np.ndarray)) else str(tag_id)
        if tag.startswith('B-'):
            if current:
                entities.append(current)
            current = {'type': tag[2:], 'start': int(offset_mapping[i][0]), 'end': int(offset_mapping[i][1])}
        elif tag.startswith('I-') and current and current['type'] == tag[2:]:
            current['end'] = int(offset_mapping[i][1])
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    # 提取文本
    for e in entities:
        e['text'] = text[e['start']:e['end']]
    return [e for e in entities if e['text']]


def test_real_world_text_expanded(model, tokenizer, id2label, label2id, cfg, model_name):
    """
    在扩充的 25 条网络文本上测试 Ours-StaticLex 模型，并基于 ground_truth 计算 P/R/F1 + 错误分类。

    Returns:
        dict: {
            'per_case': [...], 'aggregate_prf': {...}, 'error_categories': {...}
        }
    """
    cases = _load_network_text_cases(cfg)
    ground_truth = _load_network_text_ground_truth(cfg)

    print(f"\n{'='*70}")
    print(f"[Real-World Text Test (Expanded)] {model_name}")
    print(f"  Cases: {len(cases)}, Ground Truth: {len(ground_truth)} cases")
    print(f"{'='*70}")

    model.eval()
    per_case = []
    all_pred_set = []  # (text_id, entity_text, entity_type)
    all_gt_set = []    # (text_id, entity_text, entity_type)

    for item in cases:
        case_id = item['id']
        text = item['text']
        category = item.get('category', 'Unknown')
        gt_entry = ground_truth.get(case_id, {'entities': [], 'nested': []})
        gt_entities = [(e['text'], e['type']) for e in gt_entry.get('entities', [])]
        gt_nested = [(e['text'], e['type']) for e in gt_entry.get('nested', [])]
        for gt in gt_entities:
            all_gt_set.append((case_id, gt[0], gt[1]))

        # 推理
        t0 = time.time()
        encoding = tokenizer(text, max_length=cfg.max_seq_length, padding='max_length',
                             truncation=True, return_offsets_mapping=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(cfg.device)
        attention_mask = encoding['attention_mask'].to(cfg.device)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()
        with torch.no_grad():
            predictions, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(predictions, list):
            pred_tags = predictions[0] if predictions else []
            if isinstance(pred_tags, torch.Tensor):
                pred_tags = pred_tags.cpu().numpy()
        else:
            pred_tags = predictions[0].cpu().numpy()
        inference_ms = (time.time() - t0) * 1000

        # 提取预测实体
        pred_entities_list = _extract_entities_from_pred(pred_tags, text, offset_mapping, id2label)
        for pe in pred_entities_list:
            all_pred_set.append((case_id, pe['text'], pe['type']))

        # case 级别 P/R/F1
        pred_set = set([(pe['text'], pe['type']) for pe in pred_entities_list])
        gt_set = set(gt_entities)
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_case.append({
            'id': case_id, 'category': category, 'text': text,
            'pred_entities': pred_entities_list, 'gt_entities': gt_entities,
            'gt_nested': gt_nested,
            'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1,
            'inference_ms': inference_ms,
        })
        print(f"  [{case_id}|{category}] P={p:.2f} R={r:.2f} F1={f1:.2f} t={inference_ms:.1f}ms")

    # 聚合 P/R/F1
    pred_set_all = set(all_pred_set)
    gt_set_all = set(all_gt_set)
    tp_all = len(pred_set_all & gt_set_all)
    fp_all = len(pred_set_all - gt_set_all)
    fn_all = len(gt_set_all - pred_set_all)
    P_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    R_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    F1_all = 2 * P_all * R_all / (P_all + R_all) if (P_all + R_all) > 0 else 0.0
    aggregate_prf = {'precision': P_all, 'recall': R_all, 'f1': F1_all,
                     'tp': tp_all, 'fp': fp_all, 'fn': fn_all,
                     'total_pred': len(pred_set_all), 'total_gt': len(gt_set_all)}

    # 错误分类
    error_categories = categorize_realtext_errors(per_case)

    # 保存结果
    result_path = os.path.join(cfg.result_dir, f'staticlex_realtext_{model_name}.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'aggregate_prf': aggregate_prf,
            'error_categories': error_categories,
            'per_case': per_case,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[StaticLex] Real-text results saved: {result_path}")
    print(f"  Aggregate: P={P_all:.4f} R={R_all:.4f} F1={F1_all:.4f}")
    print(f"  TP={tp_all} FP={fp_all} FN={fn_all}")

    return {'per_case': per_case, 'aggregate_prf': aggregate_prf, 'error_categories': error_categories}


def categorize_realtext_errors(per_case_results):
    """
    对预测错误进行分类统计：
    - BE (Boundary Error): 实体边界切分不对
    - TE (Type Error): 边界对但类型错
    - OV (OOV Error): 词典未覆盖
    - NE (Nested Entity Error): 嵌套实体错误

    Returns:
        dict: {'BE': {'count': N, 'cases': [...]}, 'TE': {...}, 'OV': {...}, 'NE': {...}}
    """
    categories = {
        'BE': {'count': 0, 'cases': []},  # Boundary Error
        'TE': {'count': 0, 'cases': []},  # Type Error
        'OV': {'count': 0, 'cases': []},  # OOV (词典未覆盖)
        'NE': {'count': 0, 'cases': []},  # Nested Entity Error
    }

    for case in per_case_results:
        case_id = case['id']
        text = case['text']
        pred_set = set([(e['text'], e['type']) for e in case.get('pred_entities', [])])
        gt_set = set(case.get('gt_entities', []))
        gt_nested_set = set(case.get('gt_nested', []))

        # 1) BE/TE: 检查 FN（漏召）+ FP（误召）
        # FN: 真值有但预测无 —— 可能是 BE 或 TE
        for gt_text, gt_type in gt_set:
            if (gt_text, gt_type) not in pred_set:
                # 检查是否有相同文本不同类型 -> TE
                if any(p[0] == gt_text for p in pred_set):
                    pred_type_for_text = next(p[1] for p in pred_set if p[0] == gt_text)
                    categories['TE']['count'] += 1
                    categories['TE']['cases'].append({
                        'case_id': case_id, 'text': text,
                        'entity': gt_text, 'gt_type': gt_type, 'pred_type': pred_type_for_text,
                        'error': f'Type mismatch: gt={gt_type} pred={pred_type_for_text}'
                    })
                else:
                    # 检查是否预测了一个重叠但边界不同的实体 -> BE
                    overlap_preds = [p for p in pred_set if p[0] in gt_text or gt_text in p[0]]
                    if overlap_preds:
                        categories['BE']['count'] += 1
                        categories['BE']['cases'].append({
                            'case_id': case_id, 'text': text,
                            'gt_entity': gt_text, 'gt_type': gt_type,
                            'pred_entity': overlap_preds[0][0], 'pred_type': overlap_preds[0][1],
                            'error': f'Boundary mismatch: gt="{gt_text}" pred="{overlap_preds[0][0]}"'
                        })
                    else:
                        # 既无边界重叠也无类型匹配 -> 可能是 OV 或纯漏召
                        # 这里归类为 OV（词典未覆盖）
                        categories['OV']['count'] += 1
                        categories['OV']['cases'].append({
                            'case_id': case_id, 'text': text,
                            'entity': gt_text, 'gt_type': gt_type,
                            'error': f'Missed entity (likely OOV): "{gt_text}"'
                        })

        # 2) NE: 检查嵌套实体——真值的 nested 是否被错误外延覆盖（应为 nested 但被合并成外层）
        for nested_text, nested_type in gt_nested_set:
            # 如果 nested entity 应该被独立抽取但模型只抽了外层
            parent_text = None
            for nested_item in per_case_results:
                if nested_item['id'] == case_id:
                    for n in nested_item.get('gt_nested', []):
                        if n[0] == nested_text:
                            parent_text = None  # 真值 nested 元组只有 (text, type)，无 parent
                            break
            # 检查预测：nested 是否被抽取
            if (nested_text, nested_type) not in pred_set:
                # 是否外层被抽但 nested 没抽？=> 嵌套实体错误
                # 找外层（包含 nested_text 的真值实体）
                outer_in_pred = False
                for p_text, p_type in pred_set:
                    if nested_text in p_text and p_text != nested_text:
                        outer_in_pred = True
                        break
                if outer_in_pred:
                    categories['NE']['count'] += 1
                    categories['NE']['cases'].append({
                        'case_id': case_id, 'text': text,
                        'nested_entity': nested_text, 'nested_type': nested_type,
                        'error': f'Nested entity missed (only outer extracted): "{nested_text}"'
                    })

        # 3) FP（误召）部分：检查预测有但真值没有的实体
        for p_text, p_type in pred_set - gt_set:
            # 是否在嵌套真值里
            if (p_text, p_type) in gt_nested_set:
                continue  # 这是 nested 真值，模型抽出来了，不算错误（实际上是对的）
            # 是否有真值文本相同但类型不同
            if any(g[0] == p_text for g in gt_set):
                continue  # 已在 TE 中处理
            # 是否有边界重叠
            overlap_gts = [g for g in gt_set if g[0] in p_text or p_text in g[0]]
            if overlap_gts:
                continue  # 已在 BE 中处理
            # 纯误召 -> OV（词典噪声注入）
            categories['OV']['count'] += 1
            categories['OV']['cases'].append({
                'case_id': case_id, 'text': text,
                'entity': p_text, 'pred_type': p_type,
                'error': f'False positive (lexicon noise): "{p_text}"'
            })

    # 打印汇总
    print(f"\n  [Error Categorization]")
    for cat, info in categories.items():
        print(f"    {cat}: {info['count']} cases")

    return categories


def _count_parameters(model):
    """分桶统计模型参数量（按组件）"""
    buckets = {
        'encoder': 0,           # RoBERTa 编码器
        'bilstm': 0,             # BiLSTM/Attention 中间层
        'crf': 0,                # CRF 转移矩阵
        'classifier': 0,        # 分类头
        'lexicon_adapter': 0,    # 词汇适配器
        'other': 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if 'encoder' in name or 'roberta' in name.lower() or 'bert' in name.lower():
            buckets['encoder'] += n
        elif any(k in name for k in ['layers', 'bilstm', 'bigru', 'attention', 'middle']):
            buckets['bilstm'] += n
        elif 'crf' in name.lower():
            buckets['crf'] += n
        elif 'classifier' in name or 'fc' in name.lower():
            buckets['classifier'] += n
        elif 'lexicon_adapter' in name:
            buckets['lexicon_adapter'] += n
        else:
            buckets['other'] += n
    total = sum(buckets.values())
    return {'buckets': buckets, 'total': total}


def run_staticlex_benchmark(cfg, tokenizer, lexicon_matcher):
    """
    基准测试：参数量 / 推理延迟 / 词典匹配耗时占比 / P/R/F1
    对比 3 组配置：Base / Ours-StaticLex (Full) / Ours-StaticLex-LexEmbOnly
    """
    print(f"\n{'#'*80}")
    print(f"# StaticLex Benchmark: Params / Latency / Lexicon Ratio / P-R-F1")
    print(f"{'#'*80}")

    # 加载 25 条网络文本与真值
    cases = _load_network_text_cases(cfg)
    ground_truth = _load_network_text_ground_truth(cfg)

    results = []
    for bench_cfg in cfg.staticlex_benchmark_configs:
        name = bench_cfg['name']
        print(f"\n{'='*70}")
        print(f"[Benchmark] {name}")
        print(f"{'='*70}")

        # 构造模型
        model = NERModel(
            model_name=bench_cfg['model_name'],
            num_labels=len(cfg.ner_label2id),
            model_variant=name,
            use_crf=bench_cfg['use_crf'],
            use_lexicon=bench_cfg['use_lexicon'],
            lexicon_size=cfg.lexicon_size,
            lexicon_type=bench_cfg.get('lexicon_type', 'static'),
            use_reshaping=False,
            ablation_config=bench_cfg.get('ablation_config', None),
            lexicon_config={
                'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
                'fusion_type': cfg.lexicon_fusion_type,
                'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                'dropout': cfg.lexicon_dropout,
                'num_levels': cfg.num_geo_levels,
            }
        )
        model.to(cfg.device)

        # 加载检查点
        ckpt_path = os.path.join(cfg.model_save_dir, bench_cfg['ckpt'])
        if os.path.exists(ckpt_path):
            print(f"  Loading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=cfg.device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(f"  [WARN] Checkpoint not found: {ckpt_path}, using random init")

        model.eval()

        # 参数量统计
        param_info = _count_parameters(model)
        print(f"  Total params: {param_info['total']:,}")
        for k, v in param_info['buckets'].items():
            print(f"    - {k}: {v:,}")

        # 推理延迟（25 条文本 × 4 次循环取平均，模拟 100 条）
        N_RUNS = 4
        latency_list = []
        lexicon_match_time_total = 0.0
        all_pred_set = []
        all_gt_set = []

        for run_idx in range(N_RUNS):
            for item in cases:
                case_id = item['id']
                text = item['text']
                gt_entry = ground_truth.get(case_id, {'entities': [], 'nested': []})
                for e in gt_entry.get('entities', []):
                    all_gt_set.append((case_id, e['text'], e['type']))

                # 词典匹配耗时（仅第一轮统计）
                if run_idx == 0 and lexicon_matcher is not None and bench_cfg['use_lexicon']:
                    t_lx = time.time()
                    try:
                        if hasattr(lexicon_matcher, 'match_with_attributes'):
                            lexicon_matcher.match_with_attributes(text)
                    except Exception:
                        try:
                            lexicon_matcher.match(text)
                        except Exception:
                            pass
                    lexicon_match_time_total += (time.time() - t_lx) * 1000

                # 推理
                t0 = time.time()
                encoding = tokenizer(text, max_length=cfg.max_seq_length, padding='max_length',
                                     truncation=True, return_offsets_mapping=True, return_tensors='pt')
                input_ids = encoding['input_ids'].to(cfg.device)
                attention_mask = encoding['attention_mask'].to(cfg.device)
                offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()
                with torch.no_grad():
                    predictions, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(predictions, list):
                    pred_tags = predictions[0] if predictions else []
                    if isinstance(pred_tags, torch.Tensor):
                        pred_tags = pred_tags.cpu().numpy()
                else:
                    pred_tags = predictions[0].cpu().numpy()
                if run_idx == 0:
                    pred_ents = _extract_entities_from_pred(pred_tags, text, offset_mapping, cfg.ner_id2label)
                    for pe in pred_ents:
                        all_pred_set.append((case_id, pe['text'], pe['type']))
                latency_ms = (time.time() - t0) * 1000
                if run_idx > 0:  # 跳过第一次（warmup）
                    latency_list.append(latency_ms)

        # 平均延迟
        avg_latency = float(np.mean(latency_list)) if latency_list else 0.0
        # 词典匹配耗时占比 = lexicon_match_time / (lexicon_match_time + avg_inference_per_case)
        total_inference_avg = avg_latency * len(cases)
        lexicon_ratio = lexicon_match_time_total / (lexicon_match_time_total + total_inference_avg) if (lexicon_match_time_total + total_inference_avg) > 0 else 0.0

        # P/R/F1
        pred_set_all = set(all_pred_set)
        gt_set_all = set(all_gt_set)
        tp_all = len(pred_set_all & gt_set_all)
        fp_all = len(pred_set_all - gt_set_all)
        fn_all = len(gt_set_all - pred_set_all)
        P = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        R = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

        result = {
            'name': name,
            'total_params': param_info['total'],
            'adapter_params': param_info['buckets']['lexicon_adapter'],
            'encoder_params': param_info['buckets']['encoder'],
            'avg_latency_ms': avg_latency,
            'lexicon_match_ms': lexicon_match_time_total,
            'lexicon_ratio': lexicon_ratio,
            'precision': P, 'recall': R, 'f1': F1,
            'tp': tp_all, 'fp': fp_all, 'fn': fn_all,
        }
        results.append(result)
        print(f"  Avg latency: {avg_latency:.2f} ms/case ({len(latency_list)} samples)")
        print(f"  Lexicon match time: {lexicon_match_time_total:.2f} ms ({lexicon_ratio*100:.1f}% of total)")
        print(f"  P={P:.4f} R={R:.4f} F1={F1:.4f}")

        # 清理
        del model
        gc.collect()
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # 保存 CSV
    csv_path = os.path.join(cfg.result_dir, 'staticlex_benchmark.csv')
    fieldnames = ['name', 'total_params', 'adapter_params', 'encoder_params',
                  'avg_latency_ms', 'lexicon_match_ms', 'lexicon_ratio',
                  'precision', 'recall', 'f1', 'tp', 'fp', 'fn']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[StaticLex Benchmark] Saved to: {csv_path}")

    # 绘图：参数量 + 延迟 + P/R/F1 三联柱状图
    if plt_lib is not None:
        _plot_staticlex_benchmark(results, cfg.figure_dir)

    return results


def _plot_staticlex_benchmark(results, save_dir):
    """三联柱状图：参数量、延迟、P/R/F1"""
    plt = plt_lib
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 参数量（百万）
    total_p = [r['total_params'] / 1e6 for r in results]
    adapter_p = [r['adapter_params'] / 1e6 for r in results]
    width = 0.35
    axes[0].bar(x - width/2, total_p, width, label='Total (M)', color=colors[0])
    axes[0].bar(x + width/2, adapter_p, width, label='Adapter (M)', color=colors[1])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('Parameters (M)')
    axes[0].set_title('Parameter Count')
    axes[0].legend()
    for i, v in enumerate(total_p):
        axes[0].text(i - width/2, v + 1, f'{v:.1f}M', ha='center', fontsize=8)

    # 延迟
    latency = [r['avg_latency_ms'] for r in results]
    bars = axes[1].bar(x, latency, width=0.5, color=colors[2])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha='right')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Inference Latency (per case)')
    for bar, v in zip(bars, latency):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}ms', ha='center', fontsize=8)

    # P/R/F1
    p_vals = [r['precision'] * 100 for r in results]
    r_vals = [r['recall'] * 100 for r in results]
    f1_vals = [r['f1'] * 100 for r in results]
    width2 = 0.25
    axes[2].bar(x - width2, p_vals, width2, label='P', color='#1f77b4')
    axes[2].bar(x, r_vals, width2, label='R', color='#ff7f0e')
    axes[2].bar(x + width2, f1_vals, width2, label='F1', color='#2ca02c')
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=20, ha='right')
    axes[2].set_ylabel('Score (%)')
    axes[2].set_title('Precision / Recall / F1 on 25 Network Texts')
    axes[2].legend()
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'staticlex_benchmark.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[StaticLex Benchmark] Plot saved: {save_path}")


def run_staticlex_experiment(cfg, do_train=True, do_benchmark=True, do_realtext=True):
    """
    Ours-StaticLex 完整实验编排：
    1) 训练 6 个消融变体（baseline + 5 ablation）
    2) 基准测试（参数量/延迟/词典占比/P/R/F1）
    3) 真实网络文本测试 + 错误分类统计
    """
    print(f"\n{'#'*80}")
    print(f"# Ours-StaticLex Complete Experiment")
    print(f"{'#'*80}")

    # 加载共享资源
    load_ownthink = getattr(cfg, 'load_ownthink', False)
    print("\n[StaticLex] Loading GeoKnowledgeGraph...")
    kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink, skip_cache=False)
    print(f"  Lexicon: {len(kg.lexicon)} entries")
    global_lexicon_matcher = create_lexicon_matcher(cfg, load_ownthink=load_ownthink, kg_instance=kg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.roberta_model_name, local_files_only=True)

    # 加载训练/验证数据
    print("\n[StaticLex] Loading NER datasets...")
    train_data, dev_data, _ = load_ner_datasets(sample_size=None, use_dev_as_test=False)
    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}")

    # 阶段 1：训练消融变体
    if do_train:
        print(f"\n{'='*70}")
        print(f"[StaticLex Stage 1] Training 6 ablation variants")
        print(f"{'='*70}")
        train_staticlex_ablation_variants(cfg, train_data, dev_data, tokenizer, global_lexicon_matcher)

    # 阶段 2：基准测试（参数量 / 延迟 / 词典占比 / P/R/F1）
    if do_benchmark:
        print(f"\n{'='*70}")
        print(f"[StaticLex Stage 2] Benchmark (Params/Latency/Lexicon Ratio/P-R-F1)")
        print(f"{'='*70}")
        benchmark_results = run_staticlex_benchmark(cfg, tokenizer, global_lexicon_matcher)

    # 阶段 3：真实网络文本测试 + 错误分类（对 Ours-StaticLex 完整模型）
    if do_realtext:
        print(f"\n{'='*70}")
        print(f"[StaticLex Stage 3] Real-world text test + error categorization")
        print(f"{'='*70}")
        # 加载 Ours-StaticLex 完整模型
        model = NERModel(
            model_name=cfg.roberta_model_name,
            num_labels=len(cfg.ner_label2id),
            model_variant='Ours-StaticLex',
            use_crf=True, use_lexicon=True,
            lexicon_size=cfg.lexicon_size, lexicon_type='static',
            use_reshaping=False,
            ablation_config=None,  # 默认全开
            lexicon_config={
                'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
                'fusion_type': cfg.lexicon_fusion_type,
                'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                'dropout': cfg.lexicon_dropout,
                'num_levels': cfg.num_geo_levels,
            }
        )
        model.to(cfg.device)
        ckpt_path = os.path.join(cfg.model_save_dir, 'ner_model_Ours-StaticLex.pt')
        if not os.path.exists(ckpt_path):
            # 回退到 DynamicLexicon 检查点
            ckpt_path = os.path.join(cfg.model_save_dir, cfg.staticlex_dynamiclex_ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"  Loading model checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=cfg.device)
            model.load_state_dict(state, strict=False)
        else:
            print(f"  [WARN] No checkpoint found at {ckpt_path}, using random init")
        model.eval()

        test_real_world_text_expanded(
            model, tokenizer, cfg.ner_id2label, cfg.ner_label2id, cfg, 'Ours-StaticLex'
        )

        del model
        gc.collect()
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'='*70}")
    print(f"[StaticLex] All stages completed!")
    print(f"  Results: {cfg.result_dir}")
    print(f"  Figures: {cfg.figure_dir}")
    print(f"{'='*70}")


def main():
    """
    主入口：根据顶部的 RUN_MODE 变量自动选择运行模式
    """
    print(f"\n[MAIN] Starting experiment with RUN_MODE: {RUN_MODE}")
    
    # ==================== 模式选择（根据 RUN_MODE 变量） ====================
    
    if RUN_MODE == "TEST":
        print("=" * 60)
        print("TEST MODE: Quick verification with minimal settings")
        print("=" * 60)
        
        # 切换到测试模式专用输出目录（避免影响正式结果）
        cfg_ner.switch_to_test_mode()
        
        # 应用测试模式配置
        cfg_ner.max_seq_length = TEST_MODE_CONFIG["max_seq_length"]
        cfg_ner.ner_num_epochs = TEST_MODE_CONFIG["num_epochs"]
        cfg_ner.ner_batch_size = TEST_MODE_CONFIG["batch_size"]
        cfg_ner.ner_eval_batch_size = TEST_MODE_CONFIG["eval_batch_size"]
        cfg_ner.ner_patience = TEST_MODE_CONFIG["patience"]
        
        run_ner_experiments(
            cfg_ner,
            run_ablation=False,  # 测试所有13个模型，确保都能正常运行
            do_real_text_test=TEST_MODE_CONFIG["run_real_text"],
            do_geoglue=TEST_MODE_CONFIG["run_geoglue"],
            sample_size=TEST_MODE_CONFIG["sample_size"]  # 关键：减少数据量
        )
        
    elif RUN_MODE == "ABLATION":
        print("=" * 60)
        print("ABLATION MODE: Comparing 6 model variants (2 groups)")
        print("  Group 1 - BiLSTM:")
        print("    - RoBERTaBiLSTMAttentionCRF (Baseline)")
        print("    - RoBERTaBiLSTMAttentionCRF + Lexicon")
        print("    - RoBERTaBiLSTMAttentionCRF + DynamicLexicon")
        print("  Group 2 - BiGRU:")
        print("    - RoBERTaBiGRUAttentionCRF (Baseline)")
        print("    - RoBERTaBiGRUAttentionCRF + Lexicon")
        print("    - RoBERTaBiGRUAttentionCRF + DynamicLexicon")
        print("=" * 60)
        
        run_ner_experiments(
            cfg_ner,
            run_ablation=True,
            do_real_text_test=True,
            do_geoglue=True,
            sample_size=None  # 全量数据
        )
        
    elif RUN_MODE == "FULL":
        print("=" * 60)
        print("FULL EXPERIMENT MODE: Running ablation (6 models) + remaining 7 models")
        print("=" * 60)
        
        # 第一阶段：消融实验（两组共6个模型）
        print("\n" + "=" * 60)
        print("Stage 1: Ablation Study (6 models, 2 groups)")
        print("  Group 1: BiLSTM (3 models)")
        print("  Group 2: BiGRU (3 models)")
        print("=" * 60)
        ablation_results = run_ner_experiments(
            cfg_ner,
            run_ablation=True,
            do_real_text_test=False,  # 先不测试，最后一起测
            do_geoglue=False,
            sample_size=None
        )
        
        # 第二阶段：只训练剩余的7个模型（13个完整模型 - 6个消融模型）
        # 获取消融实验中已训练的模型名称
        ablation_model_names = set([m['name'] for m in cfg_ner.ablation_models])
        
        # 筛选出未在消融实验中训练的模型
        remaining_models = [
            m for m in cfg_ner.ner_experiment_models 
            if m['name'] not in ablation_model_names
        ]
        
        print("\n" + "=" * 60)
        print(f"Stage 2: Remaining Models Study ({len(remaining_models)} models)")
        print(f"  Skipping {len(ablation_model_names)} ablation models (already trained)")
        print("=" * 60)
        
        full_results = run_ner_experiments(
            cfg_ner,
            model_configs=remaining_models,  # 只训练剩余模型
            run_ablation=False,
            do_real_text_test=True,  # 最后测试最优模型
            do_geoglue=True,
            sample_size=None
        )
        
        # 合并消融实验结果和剩余模型结果
        all_results = ablation_results + full_results
        
        # 更新结果文件（包含所有13个模型）
        csv_path = os.path.join(cfg_ner.result_dir, 'ner_results.csv')
        save_results_to_csv(all_results, csv_path)
        print(f"\nMerged results saved to: {csv_path}")
        
        # 第三阶段：地理测试（自动运行）
        print("\n" + "=" * 60)
        print("Stage 3: Geographic Understanding Tests")
        print("=" * 60)
        print("Running GeoGLUE evaluation and Long-tail analysis on 6 ablation models (2 groups)...")
        
        train_data, dev_data, _ = load_ner_datasets(sample_size=None, use_dev_as_test=False)
        lexicon = load_lexicon(cfg_ner.geo_lexicon_path)
        
        # 创建全局 LexiconMatcher（从缓存加载，避免重复处理）
        print("\nLoading geographic knowledge (from cache)...")
        load_ownthink = getattr(cfg_ner, 'load_ownthink', False)
        kg = GeoKnowledgeGraph(cfg_ner.data_dir, load_ownthink=load_ownthink, skip_cache=False)
        global_lexicon_matcher = create_lexicon_matcher(cfg_ner, load_ownthink=load_ownthink, kg_instance=kg)

        # 运行地理测试
        run_geographic_tests(
            cfg=cfg_ner,
            ablation_models=cfg_ner.ablation_models,
            train_data=train_data,
            lexicon=lexicon,
            device=cfg_ner.device,
            lexicon_matcher=global_lexicon_matcher
        )
        
        print("\n" + "=" * 60)
        print("FULL EXPERIMENT COMPLETED!")
        print(f"Results saved to {cfg_ner.result_dir}/")
        print("=" * 60)
        
    elif RUN_MODE == "GEO_TEST":
        print("=" * 60)
        print("GEO_TEST MODE: Geographic Understanding Tests")
        print("  Testing 6 ablation models on GeoGLUE and Long-tail Analysis")
        print("  Group 1 - BiLSTM:")
        print("    - RoBERTaBiLSTMAttentionCRF (Baseline)")
        print("    - RoBERTaBiLSTMAttentionCRF + Lexicon")
        print("    - RoBERTaBiLSTMAttentionCRF + DynamicLexicon")
        print("  Group 2 - BiGRU:")
        print("    - RoBERTaBiGRUAttentionCRF (Baseline)")
        print("    - RoBERTaBiGRUAttentionCRF + Lexicon")
        print("    - RoBERTaBiGRUAttentionCRF + DynamicLexicon")
        print("=" * 60)
        
        # 加载训练数据和词典
        train_data, dev_data, _ = load_ner_datasets(sample_size=None, use_dev_as_test=False)
        lexicon = load_lexicon(cfg_ner.geo_lexicon_path)
        
        # 创建全局 LexiconMatcher（从缓存加载，避免重复处理）
        print("\nLoading geographic knowledge (from cache)...")
        load_ownthink = getattr(cfg_ner, 'load_ownthink', False)
        kg = GeoKnowledgeGraph(cfg_ner.data_dir, load_ownthink=load_ownthink, skip_cache=False)
        global_lexicon_matcher = create_lexicon_matcher(cfg_ner, load_ownthink=load_ownthink, kg_instance=kg)

        # 运行地理测试
        run_geographic_tests(
            cfg=cfg_ner,
            ablation_models=cfg_ner.ablation_models,
            train_data=train_data,
            lexicon=lexicon,
            device=cfg_ner.device,
            lexicon_matcher=global_lexicon_matcher
        )
        
        print("\n" + "=" * 60)
        print("GEO_TEST COMPLETED!")
        print(f"Results saved to {cfg_ner.result_dir}/")
        print(f"Charts saved to {cfg_ner.figure_dir}/")
        print("=" * 60)
        
    elif RUN_MODE == "STATICLEX":
        print("=" * 60)
        print("STATICLEX MODE: Ours-StaticLex Complete Experiment")
        print("  Stage 1: Ablation training + per-dataset evaluation")
        print("    - Ours-StaticLex (reuse existing checkpoint, eval only)")
        print("    - 5 ablation variants (train from scratch, 30 epochs)")
        print("    - Per-dataset F1: CLUENER / MSRA / Weibo / CMNER + Average")
        print("    - Ours-StaticLex_woLexEmb (only word_emb, remove level/type/coord)")
        print("    - Ours-StaticLex_woCoord (remove coord projection)")
        print("    - Ours-StaticLex_woMask (no position mask, inject knowledge everywhere)")
        print("    - Ours-StaticLex_woGate (no fusion gate, direct residual)")
        print("    - Ours-StaticLex_TokenGate (token-level gate vs global scalar)")
        print("  Stage 2: Benchmark (Params/Latency/Lexicon Ratio/P-R-F1)")
        print("    - Base (no lexicon)")
        print("    - Ours-StaticLex (full multi-feature)")
        print("    - Ours-StaticLex-LexEmbOnly (word_emb only)")
        print("  Stage 3: Real-world text test (25 cases) + error categorization")
        print("    - BE (Boundary Error), TE (Type Error)")
        print("    - OV (OOV Error), NE (Nested Entity Error)")
        print("=" * 60)

        run_staticlex_experiment(
            cfg_ner,
            do_train=True,
            do_benchmark=True,
            do_realtext=True,
        )

        print("\n" + "=" * 60)
        print("STATICLEX EXPERIMENT COMPLETED!")
        print(f"  Results: {cfg_ner.result_dir}")
        print(f"  Figures: {cfg_ner.figure_dir}")
        print("=" * 60)

    else:
        print(f"ERROR: Unknown RUN_MODE '{RUN_MODE}'")
        print("Please set RUN_MODE to one of: TEST, ABLATION, FULL, GEO_TEST, STATICLEX")

if __name__ == '__main__':
    main()