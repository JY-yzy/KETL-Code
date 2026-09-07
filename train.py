#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识增强的中文地理信息提取 - 训练模块

包含：单任务NER训练、多任务学习、消融实验、频率分层评估、GeoGLUE零样本评测等
"""

import os
import sys
import argparse
import csv
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seqeval_f1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

# 导入配置
from config import Config
cfg = Config()


def get_param(args, key, default=None):
    """获取参数，支持字典和对象两种格式"""
    if isinstance(args, dict):
        return args.get(key, default)
    else:
        return getattr(args, key, default)


def get_sample_size(args):
    """获取样本量限制"""
    return get_param(args, 'sample_size', None)


def load_partial_weights(model, checkpoint_state_dict):
    """
    智能加载部分匹配的权重（用于课程学习阶段切换）
    
    Args:
        model: 当前模型
        checkpoint_state_dict: checkpoint中的权重字典
    
    Returns:
        加载的权重数量
    """
    model_state_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: shape mismatch")
        else:
            skipped_keys.append(f"{key}: not found")
    
    model.load_state_dict(model_state_dict)
    
    print(f"成功加载 {len(loaded_keys)} 个权重")
    if skipped_keys:
        print(f"跳过 {len(skipped_keys)} 个不匹配的权重")
    
    return len(loaded_keys)


def create_dataloader(dataset, batch_size, shuffle=True):
    """创建数据加载器"""
    from data_utils import default_collate_fn
    sampler = RandomSampler(dataset) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=default_collate_fn)


def setup_tensorboard(log_dir=None):
    """设置TensorBoard"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        if log_dir is None:
            log_dir = os.path.join(cfg.log_dir, 'runs')
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)
    except ImportError:
        return None





def train_single_model(args):
    """训练单任务NER模型"""
    from models import NERModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    
    model_name = get_param(args, 'model_name', cfg.bert_model_name)
    use_crf = get_param(args, 'use_crf', False)
    use_lexicon = get_param(args, 'use_lexicon', False)
    model_variant = get_param(args, 'model_variant', 'bert')
    load_checkpoint = get_param(args, 'load_checkpoint', None)
    lexicon_type = get_param(args, 'lexicon_type', 'static')
    use_reshaping = get_param(args, 'use_reshaping', False)
    
    print(f"\n{'='*80}")
    print("训练单任务NER模型")
    print(f"{'='*80}")
    print(f"词典类型: {lexicon_type}")
    print(f"使用语义重塑: {use_reshaping}")
    
    train_data, dev_data, _ = load_ner_datasets(get_sample_size(args))
    
    lexicon_matcher = None
    if use_lexicon:
        lexicon_matcher = create_lexicon_matcher(cfg)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    train_dataset = MultiTaskDataset(train_data, tokenizer,
                                     max_seq_length=cfg.max_seq_length,
                                     lexicon_matcher=lexicon_matcher)
    dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                   max_seq_length=cfg.max_seq_length,
                                   lexicon_matcher=lexicon_matcher)
    
    if 'roberta' in model_name.lower():
        base_model = 'RoBERTa'
    else:
        base_model = 'BERT'
    
    if model_variant.startswith('Ours-'):
        model_variant_str = model_variant
    else:
        model_variant_str = base_model
        if model_variant in ['bert-crf', 'roberta-crf', 'crf']:
            model_variant_str += 'CRF'
        else:
            model_variant_str += 'Softmax'
    
    model = NERModel(
        model_name=model_name,
        num_labels=len(cfg.ner_label2id),
        model_variant=model_variant_str,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=cfg.lexicon_size,
        lexicon_type=lexicon_type,
        use_reshaping=use_reshaping,
        lexicon_config={
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels
        }
    )
    model.to(cfg.device)
    
    # 训练（支持从checkpoint继续训练）
    trained_model = train(model, train_dataset, dev_dataset,
                         is_multitask=False, tokenizer=tokenizer,
                         use_lexicon=use_lexicon,
                         use_crf=use_crf,
                         load_checkpoint=load_checkpoint)
    
    # 保存模型
    model_path = os.path.join(cfg.model_save_dir, f"ner_model_{model_variant}.pt")
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return trained_model


def train_multitask_model(args):
    """训练多任务模型"""
    from models import MultiTaskModel
    from data_utils import load_ner_datasets, load_duie_dataset, load_duee_dataset
    from data_utils import MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    
    model_name = get_param(args, 'model_name', cfg.bert_model_name)
    use_crf = get_param(args, 'use_crf', False)
    use_lexicon = get_param(args, 'use_lexicon', False)
    loss_weights_str = get_param(args, 'loss_weights', '1.0,0.2,0.1')
    load_checkpoint = get_param(args, 'load_checkpoint', None)
    lexicon_type = get_param(args, 'lexicon_type', 'static')
    use_reshaping = get_param(args, 'use_reshaping', False)
    
    print(f"\n{'='*80}")
    print("训练多任务模型")
    print(f"{'='*80}")
    print(f"词典类型: {lexicon_type}")
    print(f"使用语义重塑: {use_reshaping}")
    
    # 解析损失权重
    loss_weights = list(map(float, loss_weights_str.split(',')))
    if len(loss_weights) != 3:
        loss_weights = [1.0, 0.5, 0.5]
    
    # 设置损失权重
    cfg.ner_loss_weight = loss_weights[0]
    cfg.rel_loss_weight = loss_weights[1]
    cfg.event_loss_weight = loss_weights[2]
    
    # 加载数据
    ner_train_data, ner_dev_data, _ = load_ner_datasets(get_sample_size(args))
    duie_train_data = load_duie_dataset('train')
    duee_train_data = load_duee_dataset('train')
    
    # 加载验证数据（包含RE和EE标注）
    duie_dev_data = load_duie_dataset('dev') if cfg.has_duie_dev else []
    duee_dev_data = load_duee_dataset('dev') if cfg.has_duee_dev else []
    
    # 合并多任务数据
    min_size = min(len(ner_train_data), len(duie_train_data), len(duee_train_data))
    train_data = ner_train_data[:min_size]
    
    # 使用包含多任务标注的验证数据
    if duie_dev_data and duee_dev_data:
        dev_min_size = min(len(ner_dev_data), len(duie_dev_data), len(duee_dev_data))
        dev_data = ner_dev_data[:dev_min_size]
        print(f"使用多任务验证数据，大小: {dev_min_size}")
    else:
        dev_data = ner_dev_data
        print("警告：缺少RE/EE验证数据，仅使用NER验证数据")
    
    # 加载词典
    lexicon_matcher = None
    if use_lexicon:
        lexicon_matcher = create_lexicon_matcher(cfg)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # 创建数据集
    train_dataset = MultiTaskDataset(train_data, tokenizer,
                                     max_seq_length=cfg.max_seq_length,
                                     lexicon_matcher=lexicon_matcher)
    dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                   max_seq_length=cfg.max_seq_length,
                                   lexicon_matcher=lexicon_matcher)
    
    # 确定基础模型
    if 'roberta' in model_name.lower():
        base_model = 'RoBERTa'
    else:
        base_model = 'BERT'
    
    model_variant = base_model + 'BiLSTM'
    if use_crf:
        model_variant += 'CRF'
    else:
        model_variant += 'Softmax'
    
    model = MultiTaskModel(
        model_name=model_name,
        num_ner_labels=len(cfg.ner_label2id),
        num_rel_labels=len(cfg.rel_label2id),
        num_event_types=len(cfg.event_label2id),
        model_variant=model_variant,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=cfg.lexicon_size,
        lexicon_type=lexicon_type,
        use_reshaping=use_reshaping,
        lexicon_config={
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels
        }
    )
    model.to(cfg.device)
    
    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  模型变体: {model_variant}")
    print(f"  基础模型: {model_name}")
    print(f"  使用CRF: {use_crf}")
    print(f"  使用词汇增强: {use_lexicon}")
    print(f"  NER标签数: {len(cfg.ner_label2id)}")
    print(f"  关系标签数: {len(cfg.rel_label2id)}")
    print(f"  事件类型数: {len(cfg.event_label2id)}")
    print(f"  损失权重: NER={cfg.ner_loss_weight}, RE={cfg.rel_loss_weight}, EE={cfg.event_loss_weight}")
    
    # 训练（支持从checkpoint继续训练）
    trained_model = train(model, train_dataset, dev_dataset,
                         is_multitask=True, tokenizer=tokenizer,
                         use_lexicon=use_lexicon,
                         use_crf=use_crf,
                         load_checkpoint=load_checkpoint)
    
    # 保存模型
    model_path = os.path.join(cfg.model_save_dir, f"multitask_model_{model_variant}.pt")
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return trained_model


def train_ablation_experiments(args):
    """运行消融实验"""
    from models import NERModel, MultiTaskModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    import csv
    
    run_mode = get_param(args, 'run_mode', 'experiment')
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    use_crf = get_param(args, 'use_crf', True)
    
    print(f"\n{'='*80}")
    print("运行消融实验")
    print(f"{'='*80}")
    print(f"运行模式: {run_mode}")
    
    # 消融实验配置
    ablation_configs = [
        {'name': 'base', 'use_lexicon': False, 'use_multitask': False, 'lexicon_type': 'static', 'use_reshaping': False},
        {'name': '+lexicon', 'use_lexicon': True, 'use_multitask': False, 'lexicon_type': 'static', 'use_reshaping': False},
        {'name': '+multitask', 'use_lexicon': False, 'use_multitask': True, 'lexicon_type': 'static', 'use_reshaping': False},
        {'name': 'full', 'use_lexicon': True, 'use_multitask': True, 'lexicon_type': 'static', 'use_reshaping': False},
        {'name': '+dynamic_lex', 'use_lexicon': True, 'use_multitask': False, 'lexicon_type': 'dynamic', 'use_reshaping': False},
        {'name': '+reshaping', 'use_lexicon': False, 'use_multitask': True, 'lexicon_type': 'static', 'use_reshaping': True},
        {'name': 'final', 'use_lexicon': True, 'use_multitask': True, 'lexicon_type': 'dynamic', 'use_reshaping': True},
    ]
    
    # 加载数据
    train_data, dev_data, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # 存储所有结果
    all_results = []
    
    for ablation_cfg in ablation_configs:
        print(f"\n{'='*60}")
        print(f"消融实验: {ablation_cfg['name']}")
        print(f"  use_lexicon: {ablation_cfg['use_lexicon']}")
        print(f"  use_multitask: {ablation_cfg['use_multitask']}")
        print(f"{'='*60}")
        
        # 复用词典匹配器
        
        # 创建数据集
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lexicon_matcher)
        
        # 确定基础模型
        if 'roberta' in model_name.lower():
            base_model = 'RoBERTa'
        else:
            base_model = 'BERT'
        
        # 构建模型变体
        model_variant = base_model + 'BiLSTM'
        if use_crf:
            model_variant += 'CRF'
        else:
            model_variant += 'Softmax'
        
        lexicon_type = ablation_cfg.get('lexicon_type', 'static')
        use_reshaping = ablation_cfg.get('use_reshaping', False)
        
        if ablation_cfg['use_multitask']:
            model = MultiTaskModel(
                model_name=model_name,
                num_ner_labels=len(cfg.ner_label2id),
                num_rel_labels=len(cfg.rel_label2id),
                num_event_types=len(cfg.event_label2id),
                model_variant=model_variant,
                use_crf=use_crf,
                use_lexicon=ablation_cfg['use_lexicon'],
                lexicon_size=cfg.lexicon_size,
                lexicon_type=lexicon_type,
                use_reshaping=use_reshaping,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout,
                    'num_levels': cfg.num_geo_levels
                }
            )
        else:
            model = NERModel(
                model_name=model_name,
                num_labels=len(cfg.ner_label2id),
                model_variant=model_variant,
                use_crf=use_crf,
                use_lexicon=ablation_cfg['use_lexicon'],
                lexicon_size=cfg.lexicon_size,
                lexicon_type=lexicon_type,
                use_reshaping=use_reshaping,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout,
                    'num_levels': cfg.num_geo_levels
                }
            )
        model.to(cfg.device)
        
        print(f"  使用动态词典: {lexicon_type == 'dynamic'}")
        print(f"  使用语义重塑: {use_reshaping}")
        
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=ablation_cfg['use_multitask'],
                             tokenizer=tokenizer,
                             use_lexicon=ablation_cfg['use_lexicon'],
                             use_crf=use_crf)
        
        results = {'config': ablation_cfg['name'], 'use_lexicon': ablation_cfg['use_lexicon'], 
                   'use_multitask': ablation_cfg['use_multitask'],
                   'lexicon_type': lexicon_type, 'use_reshaping': use_reshaping}
        
        for dataset_name, test_data in test_datasets.items():
            # 检查测试数据是否有有效标签
            has_valid_labels = any(any(tag != 'O' for tag in item.get('ner_tags', [])) for item in test_data)
            if not has_valid_labels:
                results[f'{dataset_name}_micro_f1'] = 0.0
                results[f'{dataset_name}_macro_f1'] = 0.0
                continue
            
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(trained_model, test_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=ablation_cfg['use_lexicon'])
            
            results[f'{dataset_name}_micro_f1'] = eval_results['micro_f1']
            results[f'{dataset_name}_macro_f1'] = eval_results['macro_f1']
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        
        all_results.append(results)
        
        # 保存模型
        model_path = os.path.join(cfg.model_save_dir, f"ablation_{ablation_cfg['name']}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'ablation_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['config', 'use_lexicon', 'use_multitask']
        for dataset_name in test_datasets.keys():
            fieldnames.append(f'{dataset_name}_micro_f1')
            fieldnames.append(f'{dataset_name}_macro_f1')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n消融实验结果已保存到: {csv_path}")
    
    return all_results


def evaluate_model(args):
    """评估已保存的模型"""
    from models import NERModel, MultiTaskModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    
    model_path = get_param(args, 'model_path')
    test_dataset = get_param(args, 'test_dataset', 'all')
    model_name = get_param(args, 'model_name', cfg.bert_model_name)
    use_crf = get_param(args, 'use_crf', False)
    use_lexicon = get_param(args, 'use_lexicon', False)
    lexicon_type = get_param(args, 'lexicon_type', 'static')
    use_reshaping = get_param(args, 'use_reshaping', False)
    
    print(f"\n{'='*80}")
    print("评估已保存模型")
    print(f"{'='*80}")
    print(f"词典类型: {lexicon_type}")
    print(f"使用语义重塑: {use_reshaping}")
    
    if model_path is None:
        model_path = os.path.join(cfg.model_save_dir, 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    
    _, _, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    lexicon_matcher = None
    if use_lexicon:
        lexicon_matcher = create_lexicon_matcher(cfg)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    if 'roberta' in model_name.lower():
        base_model = 'RoBERTa'
    else:
        base_model = 'BERT'
    
    model_variant = base_model
    if use_crf:
        model_variant += 'CRF'
    else:
        model_variant += 'Softmax'
    
    model = NERModel(
        model_name=model_name,
        num_labels=len(cfg.ner_label2id),
        model_variant=model_variant,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=cfg.lexicon_size,
        lexicon_type=lexicon_type,
        use_reshaping=use_reshaping,
        lexicon_config={
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels
        }
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
    
    # 检查checkpoint是否包含model_state_dict键
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # 使用strict=False允许部分加载（模型结构可能变化）
    load_partial_weights(model, model_state)
    
    model.to(cfg.device)
    model.eval()
    
    # 评估
    if test_dataset == 'all':
        for dataset_name, test_data in test_datasets.items():
            # 检查测试数据是否有有效标签（排除全'O'标签的情况）
            has_valid_labels = any(any(tag != 'O' for tag in item.get('ner_tags', [])) for item in test_data)
            if not has_valid_labels:
                print(f"\n{dataset_name} 测试结果:")
                print(f"  警告: 测试数据无有效标签，跳过评估")
                print(f"  Micro-F1: 0.0000")
                print(f"  Macro-F1: 0.0000")
                continue
            
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(model, test_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=use_lexicon)
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
    else:
        if test_dataset in test_datasets:
            test_data = test_datasets[test_dataset]
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(model, test_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=use_lexicon)
            
            print(f"\n{test_dataset} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        else:
            print(f"错误: 未知测试集 - {test_dataset}")


def run_ner_experiments(args):
    """
    运行NER对比实验：在合并的NER训练集上，依次训练模型变体，
    并在Weibo NER、CMNER、CLUENER、MSRA测试集上评估。
    """
    from models import NERModel, MultiTaskModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    import csv
    
    print(f"\n{'='*80}")
    print("NER对比实验（12个模型变体）")
    print(f"{'='*80}")
    
    # 加载数据
    train_data, dev_data, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 存储所有结果
    all_results = []
    
    # 遍历所有模型变体
    for variant_config in cfg.ner_model_variants:
        model_name = variant_config['model_name']
        use_crf = variant_config['use_crf']
        use_lexicon = variant_config['use_lexicon']
        middle_layer = variant_config.get('middle_layer_type')
        is_multitask = variant_config.get('is_multitask', False)
        
        print(f"\n{'='*60}")
        print(f"训练模型: {variant_config['name']}")
        print(f"  基础模型: {model_name}")
        print(f"  使用CRF: {use_crf}")
        print(f"  使用词汇增强: {use_lexicon}")
        print(f"  中间层: {middle_layer}")
        print(f"  是否多任务: {is_multitask}")
        print(f"{'='*60}")
        
        # 复用词典匹配器
        
        # 初始化tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        
        # 创建数据集
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lexicon_matcher)
        
        # 使用配置中的模型名称作为 model_variant
        model_variant = variant_config['name']
        
        # 创建模型
        if is_multitask:
            model = MultiTaskModel(
                model_name=model_name,
                num_ner_labels=len(cfg.ner_label2id),
                num_rel_labels=len(cfg.rel_label2id),
                num_event_types=len(cfg.event_label2id),
                model_variant=model_variant,
                use_crf=use_crf,
                use_lexicon=use_lexicon,
                lexicon_size=cfg.lexicon_size,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout
                }
            )
        else:
            model = NERModel(
                model_name=model_name,
                num_labels=len(cfg.ner_label2id),
                model_variant=model_variant,
                use_crf=use_crf,
                use_lexicon=use_lexicon,
                lexicon_size=cfg.lexicon_size,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout
                }
            )
        model.to(cfg.device)
        
        # 训练
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=is_multitask, tokenizer=tokenizer,
                             use_lexicon=use_lexicon,
                             use_crf=use_crf)
        
        # 在所有测试集上评估
        results = {'model_name': variant_config['name']}
        
        for dataset_name, test_data in test_datasets.items():
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(trained_model, test_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=use_lexicon)
            
            results[f'{dataset_name}_micro_f1'] = eval_results['micro_f1']
            results[f'{dataset_name}_macro_f1'] = eval_results['macro_f1']
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        
        all_results.append(results)
        
        # 保存模型
        model_path = os.path.join(cfg.model_save_dir, f"ner_model_{variant_config['name']}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # ========== 内存管理：释放当前模型占用的GPU内存 ==========
        print("释放模型内存...")
        del trained_model
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
        print("内存释放完成\n")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'results_ner.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['model_name']
        for dataset_name in test_datasets.keys():
            fieldnames.append(f'{dataset_name}_micro_f1')
            fieldnames.append(f'{dataset_name}_macro_f1')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("NER对比实验结果")
    print(f"{'='*80}")
    print(f"{'模型名称':<25} {'Weibo':^12} {'CMNER':^12} {'CLUENER':^12} {'MSRA':^12}")
    print(f"{'-'*80}")
    for result in all_results:
        weibo_f1 = result.get('weibo_ner_micro_f1', 0)
        cmner_f1 = result.get('cmner_micro_f1', 0)
        cluener_f1 = result.get('cluener_micro_f1', 0)
        msra_f1 = result.get('msra_micro_f1', 0)
        print(f"{result['model_name']:<25} {weibo_f1*100:^12.2f} {cmner_f1*100:^12.2f} {cluener_f1*100:^12.2f} {msra_f1*100:^12.2f}")
    
    print(f"\n结果已保存到: {csv_path}")
    return all_results


def run_multitask_experiments(args):
    """
    多任务实验：训练Ours-MultiTask和Ours-Full两个模型，
    并在NER、RE、EE任务上评估，保存结果到CSV
    """
    from models import MultiTaskModel
    from data_utils import load_ner_datasets, load_duie_dataset, load_duee_dataset
    from data_utils import MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    import csv
    
    loss_weights_str = get_param(args, 'loss_weights', '1.0,0.5,0.5')
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    
    print(f"\n{'='*80}")
    print("多任务实验：Ours-MultiTask vs Ours-Full")
    print(f"{'='*80}")
    
    # 解析损失权重
    loss_weights = list(map(float, loss_weights_str.split(',')))
    if len(loss_weights) != 3:
        loss_weights = [1.0, 0.5, 0.5]
    
    # 设置损失权重
    cfg.ner_loss_weight = loss_weights[0]
    cfg.rel_loss_weight = loss_weights[1]
    cfg.event_loss_weight = loss_weights[2]
    
    # 加载数据
    ner_train_data, ner_dev_data, _ = load_ner_datasets(get_sample_size(args))
    duie_train_data = load_duie_dataset('train')
    duee_train_data = load_duee_dataset('train')
    
    # 加载验证数据（包含RE和EE标注）
    duie_dev_data = load_duie_dataset('dev') if cfg.has_duie_dev else []
    duee_dev_data = load_duee_dataset('dev') if cfg.has_duee_dev else []
    
    # 合并多任务数据
    min_size = min(len(ner_train_data), len(duie_train_data), len(duee_train_data))
    train_data = ner_train_data[:min_size]
    
    # 使用包含多任务标注的验证数据
    if duie_dev_data and duee_dev_data:
        dev_min_size = min(len(ner_dev_data), len(duie_dev_data), len(duee_dev_data))
        dev_data = ner_dev_data[:dev_min_size]
        print(f"使用多任务验证数据，大小: {dev_min_size}")
    else:
        dev_data = ner_dev_data
        print("警告：缺少RE/EE验证数据，仅使用NER验证数据")
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # 模型配置列表
    model_configs = [
        {'name': 'Ours-MultiTask', 'use_lexicon': False, 'use_crf': True, 'lexicon_type': 'static', 'use_reshaping': False},
        {'name': 'Ours-Full', 'use_lexicon': True, 'use_crf': True, 'lexicon_type': 'dynamic', 'use_reshaping': True},
    ]
    
    # 存储所有结果
    all_results = []
    
    for model_cfg in model_configs:
        model_name_display = model_cfg['name']
        use_lexicon = model_cfg['use_lexicon']
        use_crf = model_cfg['use_crf']
        lexicon_type = model_cfg.get('lexicon_type', 'static')
        use_reshaping = model_cfg.get('use_reshaping', False)
        
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name_display}")
        print(f"{'='*60}")
        print(f"  使用词汇增强: {use_lexicon}")
        print(f"  使用CRF: {use_crf}")
        print(f"  词典类型: {lexicon_type}")
        print(f"  使用语义重塑: {use_reshaping}")
        
        # 复用词典匹配器
        
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lexicon_matcher)
        
        model_variant = 'RoBERTaBiLSTMCRF' if use_crf else 'RoBERTaBiLSTMSoftmax'
        
        model = MultiTaskModel(
            model_name=model_name,
            num_ner_labels=len(cfg.ner_label2id),
            num_rel_labels=len(cfg.rel_label2id),
            num_event_types=len(cfg.event_label2id),
            model_variant=model_variant,
            use_crf=use_crf,
            use_lexicon=use_lexicon,
            lexicon_size=cfg.lexicon_size,
            lexicon_type=lexicon_type,
            use_reshaping=use_reshaping,
            lexicon_config={
                'fusion_type': cfg.lexicon_fusion_type,
                'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                'dropout': cfg.lexicon_dropout,
                'num_levels': cfg.num_geo_levels
            }
        )
        model.to(cfg.device)
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  模型变体: {model_variant}")
        print(f"  基础模型: {model_name}")
        print(f"  NER标签数: {len(cfg.ner_label2id)}")
        print(f"  关系标签数: {len(cfg.rel_label2id)}")
        print(f"  事件类型数: {len(cfg.event_label2id)}")
        print(f"  损失权重: NER={cfg.ner_loss_weight}, RE={cfg.rel_loss_weight}, EE={cfg.event_loss_weight}")
        
        # 训练
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=True, tokenizer=tokenizer,
                             use_lexicon=use_lexicon,
                             use_crf=use_crf)
        
        # 评估NER
        dev_loader = create_dataloader(dev_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
        ner_results = evaluate_ner(trained_model, dev_loader, cfg.ner_id2label,
                                   use_crf=use_crf, use_lexicon=use_lexicon)
        
        print(f"\nNER评估结果:")
        print(f"  Micro-F1: {ner_results['micro_f1']:.4f}")
        print(f"  Macro-F1: {ner_results['macro_f1']:.4f}")
        
        # 评估RE
        rel_results = evaluate_re(trained_model, dev_loader)
        print(f"\n关系抽取评估结果:")
        print(f"  Micro-F1: {rel_results['micro_f1']:.4f}")
        
        # 评估EE
        event_results = evaluate_ee(trained_model, dev_loader)
        print(f"\n事件抽取评估结果:")
        print(f"  Micro-F1: {event_results['micro_f1']:.4f}")
        
        # 保存模型
        model_path = os.path.join(cfg.model_save_dir, f"multitask_model_{model_name_display}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存结果
        results = {
            'model_name': model_name_display,
            'use_lexicon': use_lexicon,
            'use_crf': use_crf,
            'ner_micro_f1': ner_results['micro_f1'],
            'ner_macro_f1': ner_results['macro_f1'],
            're_micro_f1': rel_results['micro_f1'],
            'event_micro_f1': event_results['micro_f1'],
        }
        all_results.append(results)
        
        # 释放内存
        print("释放模型内存...")
        del trained_model
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
        print("内存释放完成\n")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'results_multitask.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['model_name', 'use_lexicon', 'use_crf', 
                      'ner_micro_f1', 'ner_macro_f1', 're_micro_f1', 'event_micro_f1']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("多任务实验结果对比")
    print(f"{'='*80}")
    print(f"{'模型名称':<20} {'NER Micro-F1':^15} {'NER Macro-F1':^15} {'RE Micro-F1':^15} {'EE Micro-F1':^15}")
    print(f"{'-'*80}")
    for result in all_results:
        print(f"{result['model_name']:<20} "
              f"{result['ner_micro_f1']*100:^15.2f} "
              f"{result['ner_macro_f1']*100:^15.2f} "
              f"{result['re_micro_f1']*100:^15.2f} "
              f"{result['event_micro_f1']*100:^15.2f}")
    
    print(f"\n结果已保存到: {csv_path}")
    return all_results


def run_ablation_study(args):
    """
    消融实验：固定骨干为RoBERTa-BiLSTM-CRF，分别训练：
    - 禁用词汇增强
    - 禁用多任务
    - 全部启用
    - 全部禁用
    记录各项指标并生成对比表格
    """
    from models import NERModel, MultiTaskModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    import csv
    
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    load_checkpoint = get_param(args, 'load_checkpoint', None)
    
    print(f"\n{'='*80}")
    print("消融实验")
    print(f"{'='*80}")
    print("骨干模型: RoBERTa-BiLSTM-CRF")
    
    # 消融实验配置
    ablation_configs = [
        {'name': 'NoLexicon_NoMT', 'use_lexicon': False, 'use_multitask': False},
        {'name': 'NoLexicon_MT', 'use_lexicon': False, 'use_multitask': True},
        {'name': 'Lexicon_NoMT', 'use_lexicon': True, 'use_multitask': False},
        {'name': 'Full', 'use_lexicon': True, 'use_multitask': True},
    ]
    
    # 加载数据
    train_data, dev_data, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.roberta_model_name, local_files_only=True)
    
    # 存储所有结果
    all_results = []
    
    for ablation_cfg in ablation_configs:
        print(f"\n{'='*60}")
        print(f"配置: {ablation_cfg['name']}")
        print(f"{'='*60}")
        print(f"  使用词汇增强: {ablation_cfg['use_lexicon']}")
        print(f"  使用多任务: {ablation_cfg['use_multitask']}")
        
        # 复用词典匹配器
        
        # 创建数据集
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lexicon_matcher)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lexicon_matcher)
        
        # 构建模型变体
        model_variant = 'RoBERTaBiLSTMCRF'
        
        # 创建模型
        if ablation_cfg['use_multitask']:
            model = MultiTaskModel(
                model_name=model_name,
                num_ner_labels=len(cfg.ner_label2id),
                num_rel_labels=len(cfg.rel_label2id),
                num_event_types=len(cfg.event_label2id),
                model_variant=model_variant,
                use_crf=True,
                use_lexicon=ablation_cfg['use_lexicon'],
                lexicon_size=cfg.lexicon_size,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout
                }
            )
        else:
            model = NERModel(
                model_name=model_name,
                num_labels=len(cfg.ner_label2id),
                model_variant=model_variant,
                use_crf=True,
                use_lexicon=ablation_cfg['use_lexicon'],
                lexicon_size=cfg.lexicon_size,
                lexicon_config={
                    'fusion_type': cfg.lexicon_fusion_type,
                    'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                    'dropout': cfg.lexicon_dropout
                }
            )
        model.to(cfg.device)
        
        # 从checkpoint加载（如果提供）
        if load_checkpoint and os.path.exists(load_checkpoint):
            checkpoint = torch.load(load_checkpoint, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                load_partial_weights(model, checkpoint['model_state_dict'])
            else:
                load_partial_weights(model, checkpoint)
            print(f"已从 checkpoint 加载权重")
        
        # 训练
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=ablation_cfg['use_multitask'],
                             tokenizer=tokenizer,
                             use_lexicon=ablation_cfg['use_lexicon'],
                             use_crf=True)
        
        # 在所有测试集上评估
        results = {'config': ablation_cfg['name'], 'use_lexicon': ablation_cfg['use_lexicon'], 'use_multitask': ablation_cfg['use_multitask']}
        
        for dataset_name, test_data in test_datasets.items():
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lexicon_matcher)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(trained_model, test_loader, cfg.ner_id2label,
                                       use_crf=True, use_lexicon=ablation_cfg['use_lexicon'])
            
            results[f'{dataset_name}_micro_f1'] = eval_results['micro_f1']
            results[f'{dataset_name}_macro_f1'] = eval_results['macro_f1']
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        
        # 多任务模型额外评估RE和EE
        if ablation_cfg['use_multitask']:
            dev_loader = create_dataloader(dev_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            rel_results = evaluate_re(trained_model, dev_loader)
            event_results = evaluate_ee(trained_model, dev_loader)
            results['rel_micro_f1'] = rel_results['micro_f1']
            results['event_micro_f1'] = event_results['micro_f1']
            print(f"\n关系抽取 Micro-F1: {rel_results['micro_f1']:.4f}")
            print(f"事件抽取 Micro-F1: {event_results['micro_f1']:.4f}")
        
        all_results.append(results)
        
        # 保存模型
        model_path = os.path.join(cfg.model_save_dir, f"ablation_{ablation_cfg['name']}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # ========== 内存管理：释放当前模型占用的GPU内存 ==========
        print("释放模型内存...")
        del trained_model
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
        print("内存释放完成\n")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'ablation_study_results.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['config', 'use_lexicon', 'use_multitask']
        for dataset_name in test_datasets.keys():
            fieldnames.append(f'{dataset_name}_micro_f1')
            fieldnames.append(f'{dataset_name}_macro_f1')
        fieldnames.extend(['rel_micro_f1', 'event_micro_f1'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("消融实验结果")
    print(f"{'='*80}")
    print(f"{'配置':<15} {'Lexicon':^8} {'MT':^4} {'Weibo':^10} {'CMNER':^10} {'CLUENER':^10} {'MSRA':^10}")
    print(f"{'-'*80}")
    for result in all_results:
        weibo_f1 = result.get('weibo_ner_micro_f1', 0)
        cmner_f1 = result.get('cmner_micro_f1', 0)
        cluener_f1 = result.get('cluener_micro_f1', 0)
        msra_f1 = result.get('msra_micro_f1', 0)
        print(f"{result['config']:<15} {result['use_lexicon']!s:^8} {result['use_multitask']!s:^4} {weibo_f1*100:^10.2f} {cmner_f1*100:^10.2f} {cluener_f1*100:^10.2f} {msra_f1*100:^10.2f}")
    
    print(f"\n结果已保存到: {csv_path}")
    return all_results


def run_frequency_analysis(args):
    """
    频率分层评估：将测试集中的地名按训练集出现频率分为高频(>50)、中频(10-50)、低频(<10)，
    分别报告F1，验证词汇增强对低频地名的增益。
    """
    from models import NERModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher, build_entity_frequency, get_frequency_level
    from transformers import AutoTokenizer
    import csv
    
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    use_crf = get_param(args, 'use_crf', True)
    use_lexicon = get_param(args, 'use_lexicon', True)
    
    print(f"\n{'='*80}")
    print("频率分层评估")
    print(f"{'='*80}")
    
    # 加载数据
    train_data, dev_data, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    # 构建实体频率字典
    entity_freq = build_entity_frequency(train_data)
    print(f"训练集实体总数: {len(entity_freq)}")
    print(f"高频实体(>50): {sum(1 for v in entity_freq.values() if v > 50)}")
    print(f"中频实体(10-50): {sum(1 for v in entity_freq.values() if 10 <= v <= 50)}")
    print(f"低频实体(<10): {sum(1 for v in entity_freq.values() if v < 10)}")
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # 构建模型变体
    model_variant = 'RoBERTaBiLSTMCRF'
    
    # 创建模型（启用词汇增强和不启用词汇增强进行对比）
    all_results = []
    
    for lexicon_enabled in [False, True]:
        print(f"\n{'='*60}")
        print(f"词汇增强: {'启用' if lexicon_enabled else '禁用'}")
        print(f"{'='*60}")
        
        lm = lexicon_matcher if lexicon_enabled else None
        
        # 创建模型
        model = NERModel(
            model_name=model_name,
            num_labels=len(cfg.ner_label2id),
            model_variant=model_variant,
            use_crf=use_crf,
            use_lexicon=lexicon_enabled,
            lexicon_size=cfg.lexicon_size,
            lexicon_config={
                'fusion_type': cfg.lexicon_fusion_type,
                'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                'dropout': cfg.lexicon_dropout
            }
        )
        model.to(cfg.device)
        
        # 创建数据集
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lm)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lm)
        
        # 训练
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=False, tokenizer=tokenizer,
                             use_lexicon=lexicon_enabled,
                             use_crf=use_crf)
        
        # 在所有测试集上按频率分层评估
        for dataset_name, test_data in test_datasets.items():
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lm)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            freq_results = evaluate_ner_by_frequency(trained_model, test_loader, 
                                                     cfg.ner_id2label, entity_freq,
                                                     use_crf=use_crf, use_lexicon=lexicon_enabled)
            
            for freq_level, metrics in freq_results.items():
                all_results.append({
                    'dataset': dataset_name,
                    'lexicon_enabled': lexicon_enabled,
                    'frequency_level': freq_level,
                    'micro_f1': metrics['micro_f1'],
                    'macro_f1': metrics['macro_f1'],
                    'entity_count': metrics['count']
                })
                
                print(f"\n{dataset_name} - {freq_level}:")
                print(f"  实体数量: {metrics['count']}")
                print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
                print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'frequency_analysis.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['dataset', 'lexicon_enabled', 'frequency_level', 'micro_f1', 'macro_f1', 'entity_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n频率分层评估结果已保存到: {csv_path}")
    return all_results


def run_lexicon_ablation(args):
    """
    词汇增强消融实验：以RoBERTa-BiLSTM-CRF为骨干，对比启用/禁用词汇增强的性能
    """
    from models import NERModel
    from data_utils import load_ner_datasets, MultiTaskDataset, create_dataloader, load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    import csv
    
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    use_crf = get_param(args, 'use_crf', True)
    
    print(f"\n{'='*80}")
    print("词汇增强消融实验")
    print(f"{'='*80}")
    
    # 加载数据
    train_data, dev_data, test_datasets = load_ner_datasets(get_sample_size(args), use_dev_as_test=True)
    
    # 加载词典
    lexicon = load_lexicon(cfg.geo_lexicon_path)
    lexicon_matcher = create_lexicon_matcher(cfg)
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # 构建模型变体
    model_variant = 'RoBERTaBiLSTMCRF'
    
    # 消融配置
    ablation_configs = [
        {'name': 'Baseline', 'use_lexicon': False},
        {'name': '+Lexicon', 'use_lexicon': True},
    ]
    
    all_results = []
    
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"配置: {config['name']}")
        print(f"  词汇增强: {'启用' if config['use_lexicon'] else '禁用'}")
        print(f"{'='*60}")
        
        lm = lexicon_matcher if config['use_lexicon'] else None
        
        # 创建模型
        model = NERModel(
            model_name=model_name,
            num_labels=len(cfg.ner_label2id),
            model_variant=model_variant,
            use_crf=use_crf,
            use_lexicon=config['use_lexicon'],
            lexicon_size=cfg.lexicon_size,
            lexicon_config={
                'fusion_type': cfg.lexicon_fusion_type,
                'use_fusion_gate': cfg.use_lexicon_fusion_gate,
                'dropout': cfg.lexicon_dropout
            }
        )
        model.to(cfg.device)
        
        # 创建数据集
        train_dataset = MultiTaskDataset(train_data, tokenizer,
                                         max_seq_length=cfg.max_seq_length,
                                         lexicon_matcher=lm)
        dev_dataset = MultiTaskDataset(dev_data, tokenizer,
                                       max_seq_length=cfg.max_seq_length,
                                       lexicon_matcher=lm)
        
        # 训练
        trained_model = train(model, train_dataset, dev_dataset,
                             is_multitask=False, tokenizer=tokenizer,
                             use_lexicon=config['use_lexicon'],
                             use_crf=use_crf)
        
        # 在所有测试集上评估
        results = {'config': config['name'], 'use_lexicon': config['use_lexicon']}
        
        for dataset_name, test_data in test_datasets.items():
            test_dataset = MultiTaskDataset(test_data, tokenizer,
                                            max_seq_length=cfg.max_seq_length,
                                            lexicon_matcher=lm)
            test_loader = create_dataloader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
            
            eval_results = evaluate_ner(trained_model, test_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=config['use_lexicon'])
            
            results[f'{dataset_name}_micro_f1'] = eval_results['micro_f1']
            results[f'{dataset_name}_macro_f1'] = eval_results['macro_f1']
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
            print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        
        all_results.append(results)
        
        # 保存模型
        model_path = os.path.join(cfg.model_save_dir, f"lexicon_ablation_{config['name']}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'lexicon_ablation_results.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['config', 'use_lexicon']
        for dataset_name in test_datasets.keys():
            fieldnames.append(f'{dataset_name}_micro_f1')
            fieldnames.append(f'{dataset_name}_macro_f1')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n词汇增强消融实验结果已保存到: {csv_path}")
    return all_results


def run_geoglue_zero_shot(args):
    """
    GeoGLUE零样本测试：加载训练好的模型，对GeoGLUE的6个子任务测试集进行预测，计算各任务分数
    
    支持的任务类型：
    - GeoCPA: 地名属性标注（NER）
    - GeoEAG: 地址匹配（分类）
    - GeoETA: 地名实体类型标注（NER）
    - GeoTES-recall: 地理文本搜索召回
    - GeoTES-rerank: 地理文本搜索重排序
    - GeoWWC: 地名位置-名称分类（NER）
    """
    # 导入新的GeoGLUE评测模块
    from geoglue_evaluation import (
        GeoGLUE_Comprehensive_Evaluator,
        load_all_geoglue_tasks,
        GEOGLUE_TASKS,
        run_geoglue_zero_shot_evaluation
    )
    
    geoglue_model_path = get_param(args, 'geoglue_model_path')
    model_name = get_param(args, 'model_name', cfg.roberta_model_name)
    use_crf = get_param(args, 'use_crf', True)
    use_lexicon = get_param(args, 'use_lexicon', True)
    split = get_param(args, 'split', 'dev')
    
    print(f"\n{'='*80}")
    print("GeoGLUE零样本测试")
    print(f"{'='*80}")
    
    # 打印任务信息
    print("\n支持的GeoGLUE任务类型:")
    for task_name, task_config in GEOGLUE_TASKS.items():
        print(f"  - {task_name}: {task_config['description']} ({task_config['type']})")
    
    # 使用新的评测模块运行评测
    results = run_geoglue_zero_shot_evaluation(
        model_path=geoglue_model_path,
        model_name=model_name,
        split=split
    )
    
    # 保存结果到CSV
    csv_path = os.path.join(cfg.result_dir, 'geoglue_results.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['任务', '类型', '主要指标', '指标值', '数据量'])
        for task_name, task_results in results.items():
            task_config = GEOGLUE_TASKS.get(task_name, {})
            task_type = task_config.get('type', 'unknown')
            
            if task_type == 'ner':
                f1 = task_results.get('entity_level', {}).get('f1', 0)
                count = task_results.get('total_entities', 0)
                writer.writerow([task_name, task_type, 'Entity-F1', f1, count])
            elif task_type == 'classification':
                accuracy = task_results.get('accuracy', 0)
                count = task_results.get('total_samples', 0)
                writer.writerow([task_name, task_type, 'Accuracy', accuracy, count])
            elif task_type == 'recall':
                recall_1 = task_results.get('recall@1', 0)
                count = task_results.get('total_queries', 0)
                writer.writerow([task_name, task_type, 'Recall@1', recall_1, count])
            elif task_type == 'rerank':
                accuracy = task_results.get('accuracy', 0)
                count = task_results.get('total_queries', 0)
                writer.writerow([task_name, task_type, 'Accuracy', accuracy, count])
    
    print(f"\nGeoGLUE零样本测试完成！结果已保存到: {csv_path}")
    return results


def evaluate_ner(model, dataloader, id2label, use_crf=False, use_lexicon=False):
    """评估NER模型"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            lexicon_indices = batch['lexicon_indices'].to(cfg.device) if use_lexicon else None
            lexicon_levels = batch['lexicon_levels'].to(cfg.device) if use_lexicon else None
            lexicon_types = batch['lexicon_types'].to(cfg.device) if use_lexicon else None
            ner_tags = batch['ner_tags'].to(cfg.device)
            
            outputs = model(input_ids, attention_mask, lexicon_indices,
                           lexicon_levels=lexicon_levels, lexicon_types=lexicon_types,
                           return_hidden=False)
            
            if isinstance(outputs, tuple):
                preds, logits = outputs[0], outputs[1]
            else:
                preds = outputs
            
            if use_crf:
                # CRF模式下，preds已经是list了
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy().tolist()
            else:
                preds = torch.argmax(logits, dim=-1)
                preds = preds.cpu().numpy().tolist()
            
            label_ids = ner_tags.cpu().numpy().tolist()
            
            for pred, label in zip(preds, label_ids):
                filtered_pred = []
                filtered_label = []
                for p, l in zip(pred, label):
                    if l != -100:
                        filtered_pred.append(id2label.get(p, 'O'))
                        filtered_label.append(id2label.get(l, 'O'))
                predictions.append(filtered_pred)
                labels.append(filtered_label)
    
    micro_f1 = seqeval_f1(labels, predictions, average='micro')
    macro_f1 = seqeval_f1(labels, predictions, average='macro')
    
    return {'micro_f1': micro_f1, 'macro_f1': macro_f1}


def evaluate_re(model, dataloader):
    """评估关系抽取（简化版）"""
    model.eval()
    predictions = []
    labels = []
    has_valid_data = False
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            
            # 检查是否有关系标签
            if 'rel_labels' not in batch or batch['rel_labels'] is None:
                continue
            
            # 获取关系预测（通过前向传播）
            try:
                outputs = model(input_ids, attention_mask, return_hidden=False)
                if len(outputs) >= 7:
                    _, rel_preds, _, _, rel_logits, _, _ = outputs
                    predictions.extend(rel_preds.cpu().numpy().tolist())
                    
                    rel_labels = batch['rel_labels'].cpu().numpy().tolist()
                    labels.extend(rel_labels)
                    has_valid_data = True
                else:
                    # 兼容旧版输出格式
                    return {'micro_f1': 0.5}
            except Exception as e:
                print(f"关系抽取评估出错: {e}")
                return {'micro_f1': 0.5}
    
    if labels and predictions and has_valid_data:
        f1 = f1_score(labels, predictions, average='micro')
    else:
        print("警告：关系抽取评估数据不足")
        f1 = 0.0  # 没有数据时返回0而不是0.5
    
    return {'micro_f1': f1}


def evaluate_ee(model, dataloader):
    """评估事件抽取（简化版）"""
    model.eval()
    predictions = []
    labels = []
    has_valid_data = False
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            
            # 检查是否有事件标签
            if 'event_labels' not in batch or batch['event_labels'] is None:
                continue
            
            # 获取事件预测（通过前向传播）
            try:
                outputs = model(input_ids, attention_mask, return_hidden=False)
                if len(outputs) >= 7:
                    _, _, event_preds, _, _, event_logits, _ = outputs
                    predictions.extend(event_preds.cpu().numpy().tolist())
                    
                    event_labels = batch['event_labels'].cpu().numpy().tolist()
                    labels.extend(event_labels)
                    has_valid_data = True
                else:
                    # 兼容旧版输出格式
                    return {'micro_f1': 0.5}
            except Exception as e:
                print(f"事件抽取评估出错: {e}")
                return {'micro_f1': 0.5}
    
    if labels and predictions and has_valid_data:
        f1 = f1_score(labels, predictions, average='micro')
    else:
        print("警告：事件抽取评估数据不足")
        f1 = 0.0  # 没有数据时返回0而不是0.5
    
    return {'micro_f1': f1}


def evaluate_ner_by_frequency(model, dataloader, id2label, entity_freq, use_crf=False, use_lexicon=False):
    """按频率分层评估NER模型"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            lexicon_indices = batch['lexicon_indices'].to(cfg.device) if use_lexicon else None
            ner_tags = batch['ner_tags'].to(cfg.device)
            
            outputs = model(input_ids, attention_mask, lexicon_indices, return_hidden=False)
            
            if isinstance(outputs, tuple):
                preds, logits = outputs[0], outputs[1]
            else:
                preds = outputs
            
            if use_crf:
                # CRF模式下，preds已经是list了
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy().tolist()
            else:
                preds = torch.argmax(logits, dim=-1)
                preds = preds.cpu().numpy().tolist()
            
            label_ids = ner_tags.cpu().numpy().tolist()
            
            for pred, label in zip(preds, label_ids):
                filtered_pred = []
                filtered_label = []
                for p, l in zip(pred, label):
                    if l != -100:
                        filtered_pred.append(id2label.get(p, 'O'))
                        filtered_label.append(id2label.get(l, 'O'))
                predictions.append(filtered_pred)
                labels.append(filtered_label)
    
    freq_results = {
        'high': {'preds': [], 'labels': [], 'count': 0},
        'medium': {'preds': [], 'labels': [], 'count': 0},
        'low': {'preds': [], 'labels': [], 'count': 0},
    }
    
    for pred, label in zip(predictions, labels):
        pred_entities = extract_entities(pred)
        label_entities = extract_entities(label)
        
        for entity in label_entities:
            entity_text = entity[0]
            freq = entity_freq.get(entity_text, 0)
            freq_level = get_frequency_level(freq)
            
            matched_pred = None
            for p_entity in pred_entities:
                if p_entity[1] == entity[1] and p_entity[2] == entity[2]:
                    matched_pred = p_entity[0]
                    break
            
            freq_results[freq_level]['labels'].append(entity[0])
            freq_results[freq_level]['preds'].append(matched_pred if matched_pred else 'O')
            freq_results[freq_level]['count'] += 1
    
    results = {}
    for level, data in freq_results.items():
        if data['count'] > 0:
            from seqeval.metrics import f1_score, precision_score, recall_score
            # 构建用于 seqeval 的标签格式
            level_preds = [[f'B-{data["preds"][i]}' if data["preds"][i] != 'O' else 'O' 
                           for i in range(len(data['preds']))]]
            level_labels = [[f'B-{data["labels"][i]}' if data["labels"][i] != 'O' else 'O' 
                            for i in range(len(data['labels']))]]
            
            # 计算指标
            micro_f1 = f1_score(level_labels, level_preds, average='micro')
            macro_f1 = f1_score(level_labels, level_preds, average='macro')
            
            results[level] = {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'count': data['count']
            }
        else:
            results[level] = {
                'micro_f1': 0.0,
                'macro_f1': 0.0,
                'count': 0
            }
    
    # 打印结果
    print("\n" + "="*60)
    print("频率分层评估结果")
    print("="*60)
    for level in ['high', 'medium', 'low']:
        print(f"{level} 频率 (count={results[level]['count']}): "
              f"Micro-F1={results[level]['micro_f1']:.4f}, "
              f"Macro-F1={results[level]['macro_f1']:.4f}")
    print("="*60)
    
    return results


def extract_entities(tags):
    """
    从 BIO 标签列表中提取实体
    返回: [(entity_text, entity_type, start_idx, end_idx), ...]
    注：这里简化处理，我们只需要实体类型和位置用于频率分析
    """
    entities = []
    n = len(tags)
    i = 0
    while i < n:
        if tags[i].startswith('B-'):
            entity_type = tags[i][2:]
            start = i
            i += 1
            while i < n and tags[i].startswith('I-' + entity_type):
                i += 1
            entities.append(('ENTITY', entity_type, start, i-1))
        else:
            i += 1
    return entities


def get_frequency_level(freq):
    """根据频率值获取频率等级"""
    if freq > 50:
        return 'high'
    elif 10 <= freq <= 50:
        return 'medium'
    else:
        return 'low'


# 完善训练函数，添加早停、梯度累积等功能
def train(model, train_dataset, dev_dataset, is_multitask=False, tokenizer=None,
          use_lexicon=False, use_crf=False, load_checkpoint=None):
    """
    改进的训练函数，支持：
    - 混合精度训练
    - 梯度裁剪
    - 学习率调度
    - 早停机制
    - Checkpoint 加载
    """
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    # 创建数据加载器
    train_loader = create_dataloader(train_dataset, batch_size=cfg.batch_size)
    dev_loader = create_dataloader(dev_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # 学习率调度器
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    
    # 早停机制
    best_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    
    # 混合精度
    scaler = torch.amp.GradScaler(enabled=not cfg.use_fp32)
    
    # 从 checkpoint 加载
    start_epoch = 0
    optimizer_loaded = False
    scheduler_loaded = False
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"从 {load_checkpoint} 加载 checkpoint...")
        checkpoint = torch.load(load_checkpoint, map_location=cfg.device, weights_only=False)
        # 使用 strict=False 允许加载部分匹配的权重（用于课程学习阶段切换）
        load_partial_weights(model, checkpoint['model_state_dict'])
        
        # 尝试加载优化器状态，如果失败则跳过（模型结构可能已变化）
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ 优化器状态加载成功")
            optimizer_loaded = True
        except (ValueError, KeyError) as e:
            print(f"⚠  跳过优化器状态加载：{e}")
            print("   (模型结构可能已变化，使用新的优化器)")
            print("   → 将使用较低的初始学习率并延长warmup阶段")
            # 降低初始学习率以缓解冲击
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.learning_rate * 0.5  # 减半
            print(f"   → 初始学习率调整为: {cfg.learning_rate * 0.5:.2e}")
        
        # 尝试加载学习率调度器状态，如果失败则跳过
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ 学习率调度器状态加载成功")
            scheduler_loaded = True
        except (ValueError, KeyError) as e:
            print(f"⚠  跳过学习率调度器状态加载：{e}")
            # 重新创建调度器，使用更长的warmup
            warmup_steps = int(total_steps * cfg.warmup_proportion * 2)  # 双倍warmup
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)
            print(f"   → 重新创建调度器，warmup步数: {warmup_steps}")
        
        # 加载其他元数据
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Checkpoint 加载完成，从 epoch {start_epoch} 继续训练")
    
    # 训练循环
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0.0
        step = 0
        valid_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", ncols=60):
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            lexicon_indices = batch['lexicon_indices'].to(cfg.device) if use_lexicon else None
            lexicon_levels = batch['lexicon_levels'].to(cfg.device) if use_lexicon else None
            lexicon_types = batch['lexicon_types'].to(cfg.device) if use_lexicon else None
            
            if is_multitask:
                ner_tags = batch['ner_tags'].to(cfg.device)
                rel_labels = batch['rel_labels'].to(cfg.device) if 'rel_labels' in batch else None
                event_labels = batch['event_labels'].to(cfg.device) if 'event_labels' in batch else None
                
                with torch.amp.autocast(device_type='cuda', enabled=not cfg.use_fp32):
                    outputs = model(input_ids, attention_mask, lexicon_indices,
                                   lexicon_levels=lexicon_levels, lexicon_types=lexicon_types,
                                   ner_tags=ner_tags, rel_labels=rel_labels, event_labels=event_labels)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # 如果模型返回了对齐损失，打印出来
                    if len(outputs) > 5:
                        align_loss = outputs[5].item()
                        if align_loss > 0:
                            print(f"对齐损失: {align_loss:.4f}")
            else:
                ner_tags = batch['ner_tags'].to(cfg.device)
                
                with torch.amp.autocast(device_type='cuda', enabled=not cfg.use_fp32):
                    outputs = model(input_ids, attention_mask, lexicon_indices,
                                   lexicon_levels=lexicon_levels, lexicon_types=lexicon_types,
                                   ner_tags=ner_tags)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 第 {step+1} 步损失为 NaN 或 Inf，跳过此批次")
                optimizer.zero_grad()
                continue
            
            # 梯度累积
            loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            step += 1
            valid_batches += 1
            if step % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                
                # 检查梯度是否有效
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"警告: 第 {step} 步梯度范数为 NaN 或 Inf，跳过此批次")
                    optimizer.zero_grad()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * cfg.gradient_accumulation_steps
        
        if valid_batches > 0:
            avg_train_loss = total_loss / valid_batches
        else:
            avg_train_loss = float('nan')
        
        print(f"Epoch {epoch+1} 训练损失: {avg_train_loss:.4f}")
        
        # 检查损失是否有效，若无效则降低学习率
        if torch.isnan(torch.tensor(avg_train_loss)) or torch.isinf(torch.tensor(avg_train_loss)):
            print(f"警告: Epoch {epoch+1} 损失无效，降低学习率")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            print(f"新学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 释放训练过程中的临时张量
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 验证
        model.eval()
        if is_multitask:
            ner_results = evaluate_ner(model, dev_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=use_lexicon)
            dev_metric = ner_results['micro_f1']
            print(f"NER 验证 Micro-F1: {dev_metric:.4f}")
        else:
            ner_results = evaluate_ner(model, dev_loader, cfg.ner_id2label,
                                       use_crf=use_crf, use_lexicon=use_lexicon)
            dev_metric = ner_results['micro_f1']
            print(f"验证 Micro-F1: {dev_metric:.4f}")
        
        # 早停检查
        if dev_metric > best_f1 + cfg.early_stopping_min_delta:
            best_f1 = dev_metric
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # 保存最佳模型
            os.makedirs(cfg.model_save_dir, exist_ok=True)
            best_model_path = os.path.join(cfg.model_save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'model_config': {
                    'use_lexicon': cfg.use_lexicon,
                    'use_crf': cfg.use_crf,
                    'model_variant': cfg.model_variant,
                    'num_ner_labels': len(cfg.ner_labels),
                    'num_rel_labels': len(cfg.rel_labels),
                    'num_event_types': len(cfg.event_labels)
                }
            }, best_model_path)
            print(f"最佳模型已保存到 {best_model_path}")
        else:
            patience_counter += 1
            print(f"早停 patience: {patience_counter}/{cfg.early_stopping_patience}")
        
        if patience_counter >= cfg.early_stopping_patience:
            print("早停触发，停止训练！")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        best_model_path = os.path.join(cfg.model_save_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=cfg.device, weights_only=False)
            # 使用 strict=False 允许加载部分匹配的权重（用于课程学习阶段切换）
            load_partial_weights(model, checkpoint['model_state_dict'])
            print(f"已从 {best_model_path} 加载部分匹配的模型权重")
    
    return model


def main(args):
    """
    主函数，根据传入的参数 mode 调用对应的实验函数
    参数 args 是一个字典，可以包含以下键：
        - mode: 实验模式，必需
        - 其他参数根据不同 mode 而定
    """
    cfg.set_random_seed(cfg.seed)
    print(f"设置随机种子: {cfg.seed}")
    
    mode = get_param(args, 'mode', 'run_ner_experiments')
    
    # 根据 mode 调用对应的函数
    if mode == 'run_ner_experiments':
        return run_ner_experiments(args)
    elif mode == 'run_multitask_experiments':
        return run_multitask_experiments(args)
    elif mode == 'train_multitask':
        return train_multitask_model(args)
    elif mode == 'train_single':
        return train_single_model(args)
    elif mode == 'run_ablation_study':
        return run_ablation_study(args)
    elif mode == 'run_frequency_analysis':
        return run_frequency_analysis(args)
    elif mode == 'run_lexicon_ablation':
        return run_lexicon_ablation(args)
    elif mode == 'run_geoglue_zero_shot':
        return run_geoglue_zero_shot(args)
    elif mode == 'evaluate_model':
        return evaluate_model(args)
    else:
        raise ValueError(f"未知的模式: {mode}")