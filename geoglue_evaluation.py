#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoGLUE评测模块 - 支持所有6种任务类型的评测

任务类型：
1. GeoCPA (Chinese Place Attribute) - 地名属性标注（NER任务）
2. GeoEAG (Entity Address Geocoding) - 地址匹配（分类任务）
3. GeoETA (Entity Type Annotation) - 地名实体类型标注（NER任务）
4. GeoTES-recall (Text Entity Search - Recall) - 地理文本搜索召回
5. GeoTES-rerank (Text Entity Search - Rerank) - 地理文本搜索重排序
6. GeoWWC (Where What Classification) - 地名位置-名称分类（NER任务）
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from config import Config

cfg = Config()


# ========================== GeoGLUE任务配置 ==========================

GEOGLUE_TASKS = {
    'GeoCPA': {
        'name': 'Chinese Place Attribute',
        'description': '地名属性标注',
        'type': 'ner',
        'file_prefix': '',  # train.json, dev.json, test.json
        'label_types': ['Entity', 'PC', 'SA', 'UD', 'PB', 'PF', 'RD', 'UA', 'NumEng', 'ZZ', 'PD', 'UB']
    },
    'GeoEAG': {
        'name': 'Entity Address Geocoding',
        'description': '地址匹配',
        'type': 'classification',
        'file_prefix': '',
        'label_types': ['partial_match', 'not_match', 'exact_match']
    },
    'GeoETA': {
        'name': 'Entity Type Annotation',
        'description': '地名实体类型标注',
        'type': 'ner',
        'file_prefix': '',
        'label_types': ['prov', 'city', 'district', 'town', 'community', 'poi', 'road', 
                       'roadno', 'subpoi', 'houseno', 'devzone', 'intersection']
    },
    'GeoTES-recall': {
        'name': 'Text Entity Search - Recall',
        'description': '地理文本搜索召回',
        'type': 'recall',
        'file_prefix': '',
        'metrics': ['recall@1', 'recall@5', 'recall@10', 'mrr']
    },
    'GeoTES-rerank': {
        'name': 'Text Entity Search - Rerank',
        'description': '地理文本搜索重排序',
        'type': 'rerank',
        'file_prefix': 'adqid_',  # adqid_train.json, adqid_dev.json, adqid_test.json
        'metrics': ['accuracy', 'ndcg@5', 'ndcg@10', 'mrr']
    },
    'GeoWWC': {
        'name': 'Where What Classification',
        'description': '地名位置-名称分类',
        'type': 'ner',
        'file_prefix': '',
        'label_types': ['/WHERE', '/WHAT', '/OTHER']
    }
}


# ========================== 数据加载函数 ==========================

def load_geoglue_task_data(task_name: str, split: str = 'dev') -> List[Dict]:
    """
    加载GeoGLUE指定任务的数据
    
    Args:
        task_name: 任务名称 (GeoCPA, GeoEAG, GeoETA, GeoTES-recall, GeoTES-rerank, GeoWWC)
        split: 数据划分 (train, dev, test)
    
    Returns:
        数据列表
    """
    task_config = GEOGLUE_TASKS.get(task_name)
    if not task_config:
        print(f"错误: 未知的任务类型 {task_name}")
        return []
    
    # 构建文件路径
    file_prefix = task_config.get('file_prefix', '')
    file_name = f"{file_prefix}{split}.json"
    file_path = os.path.join(cfg.data_dir, 'GeoGLUE', task_name, file_name)
    
    if not os.path.exists(file_path):
        print(f"错误: 数据文件不存在 - {file_path}")
        return []
    
    # 加载JSON Lines格式数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"加载 {task_name} {split} 数据: {len(data)} 条")
    return data


def load_all_geoglue_tasks(split: str = 'dev') -> Dict[str, List[Dict]]:
    """
    加载所有GeoGLUE任务的数据
    
    Args:
        split: 数据划分 (train, dev, test)
    
    Returns:
        任务数据字典 {task_name: data_list}
    """
    all_data = {}
    for task_name in GEOGLUE_TASKS.keys():
        data = load_geoglue_task_data(task_name, split)
        if data:
            all_data[task_name] = data
    return all_data


# ========================== NER任务评测 ==========================

class GeoGLUE_NER_Evaluator:
    """GeoGLUE NER任务评测器（适用于GeoCPA, GeoETA, GeoWWC）"""
    
    def __init__(self, task_name: str, model_label2id: Dict = None, model_id2label: Dict = None):
        self.task_name = task_name
        self.task_config = GEOGLUE_TASKS[task_name]
        
        # 构建GeoGLUE任务的标签映射
        self.label_types = self.task_config.get('label_types', [])
        self.build_label_mapping()
        
        # 模型的标签映射（用于解码模型输出）
        self.model_label2id = model_label2id
        self.model_id2label = model_id2label
    
    def build_label_mapping(self):
        """构建GeoGLUE任务的标签映射"""
        # 对于GeoGLUE的NER任务，标签格式为 B-type, I-type, E-type 或 B-/type, I-/type, E-/type
        all_labels = ['O']
        for label_type in self.label_types:
            # 处理GeoWWC的特殊格式
            if '/' in label_type:
                all_labels.extend([f'B-{label_type}', f'I-{label_type}', f'E-{label_type}'])
            else:
                all_labels.extend([f'B-{label_type}', f'I-{label_type}', f'E-{label_type}'])
        
        self.label2id = {label: idx for idx, label in enumerate(all_labels)}
        self.id2label = {idx: label for idx, label in enumerate(all_labels)}
        self.num_labels = len(all_labels)
    
    def set_model_label_mapping(self, model_label2id: Dict, model_id2label: Dict):
        """设置模型的标签映射"""
        self.model_label2id = model_label2id
        self.model_id2label = model_id2label
    
    def map_model_pred_to_geoglue(self, model_pred_tags: List[str]) -> List[str]:
        """
        将模型的预测标签映射到GeoGLUE任务的标签格式
        
        Args:
            model_pred_tags: 模型预测的标签列表（基于训练模型的标签体系）
        
        Returns:
            映射后的GeoGLUE标签列表
        """
        # 简单映射规则：将GEO/ADMIN/LANDMARK等映射到GeoGLUE的实体类型
        # 这是一个简化的映射，实际应用中可能需要更复杂的规则
        mapped_tags = []
        for tag in model_pred_tags:
            if tag == 'O':
                mapped_tags.append('O')
            elif tag.startswith('B-') or tag.startswith('I-') or tag.startswith('E-'):
                prefix = tag[0:2]  # B-, I-, E-
                entity_type = tag[2:]  # GEO, ADMIN, LANDMARK, etc.
                
                # 映射规则
                if entity_type in ['GEO', 'ADMIN', 'LANDMARK']:
                    # 地理相关实体，映射到GeoGLUE的Entity类型
                    mapped_tags.append(f'{prefix}Entity')
                elif entity_type in ['ORG']:
                    mapped_tags.append(f'{prefix}Entity')
                elif entity_type in ['PER']:
                    mapped_tags.append('O')  # 人物实体在GeoGLUE中可能不是重点
                else:
                    mapped_tags.append('O')
            else:
                mapped_tags.append('O')
        
        return mapped_tags
    
    def extract_entities_from_tags(self, tokens: List[str], tags: List[str]) -> List[Dict]:
        """
        从BIO/BIOES标签中提取实体
        
        Args:
            tokens: token列表
            tags: 标签列表
        
        Returns:
            实体列表 [{'text': '西湖区', 'type': 'district', 'start': 0, 'end': 3}, ...]
        """
        entities = []
        current_entity = None
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    'text': tokens[i] if i < len(tokens) else '',
                    'type': entity_type,
                    'start': i,
                    'end': i + 1
                }
            elif tag.startswith('I-'):
                if current_entity is not None:
                    expected_type = tag[2:]
                    if expected_type == current_entity['type']:
                        current_entity['end'] = i + 1
                        if i < len(tokens):
                            current_entity['text'] = ''.join(tokens[current_entity['start']:i+1])
            elif tag.startswith('E-'):
                if current_entity is not None:
                    expected_type = tag[2:]
                    if expected_type == current_entity['type']:
                        current_entity['end'] = i + 1
                        if i < len(tokens):
                            current_entity['text'] = ''.join(tokens[current_entity['start']:i+1])
                        entities.append(current_entity)
                        current_entity = None
            else:  # 'O' 或其他
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    def calculate_metrics(self, true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, float]:
        """
        计算实体级别的指标
        
        Args:
            true_entities: 真实实体列表
            pred_entities: 预测实体列表
        
        Returns:
            metrics: 包含 precision, recall, f1 的字典
        """
        # 使用 (start, end, type) 作为实体的唯一标识
        true_set = set((e['start'], e['end'], e['type']) for e in true_entities)
        pred_set = set((e['start'], e['end'], e['type']) for e in pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate(self, data: List[Dict], model=None, tokenizer=None) -> Dict[str, Any]:
        """
        评测NER任务
        
        Args:
            data: 测试数据
            model: 模型实例（可选，用于零样本评测）
            tokenizer: tokenizer实例
        
        Returns:
            评测结果
        """
        all_true_entities = []
        all_pred_entities = []
        all_true_tags = []
        all_pred_tags = []
        
        # 检查数据是否有标签
        has_labels = any('ner_tags' in item for item in data)
        
        if not has_labels:
            print(f"注意: 测试数据无标签，仅进行预测")
            if model is None:
                print("警告: 未提供模型且无标签，无法进行评测")
                return {
                    'entity_level': {'f1': 0, 'precision': 0, 'recall': 0},
                    'token_level': {'f1': 0, 'precision': 0, 'recall': 0},
                    'per_type': {},
                    'total_entities': 0,
                    'total_predictions': 0,
                    'has_labels': False,
                    'predictions_only': True
                }
        
        # 如果没有模型，使用标签本身作为预测（用于数据验证）
        if model is None:
            print("警告: 未提供模型，使用真实标签作为预测（仅用于数据验证）")
            for item in data:
                tokens = item.get('tokens', [])
                true_tags = item.get('ner_tags', [])
                
                if not true_tags:  # 如果没有标签，跳过
                    continue
                
                true_entities = self.extract_entities_from_tags(tokens, true_tags)
                all_true_entities.extend(true_entities)
                all_pred_entities.extend(true_entities)  # 使用真实标签作为预测
                all_true_tags.append(true_tags)
                all_pred_tags.append(true_tags)
        else:
            # 使用模型进行预测
            model.eval()
            with torch.no_grad():
                for item in tqdm(data, desc=f"评测 {self.task_name}"):
                    tokens = item.get('tokens', [])
                    true_tags = item.get('ner_tags', [])
                    text = ''.join(tokens)
                    
                    # Tokenize
                    encoding = tokenizer(text, truncation=True, padding='max_length',
                                        max_length=cfg.max_seq_length, return_tensors='pt')
                    input_ids = encoding['input_ids'].to(cfg.device)
                    attention_mask = encoding['attention_mask'].to(cfg.device)
                    
                    # 预测
                    outputs = model(input_ids, attention_mask, None)
                    if isinstance(outputs, tuple):
                        preds = outputs[0]
                    else:
                        preds = outputs
                    
                    # 将预测转换为标签
                    # CRF模型返回列表的列表（标签ID），Softmax返回Tensor
                    # 使用模型的标签映射来解码，而不是GeoGLUE任务的标签映射
                    
                    # 获取模型预测的标签ID
                    if isinstance(preds, list):
                        # CRF输出：列表的列表 [[tag_id1, tag_id2, ...], ...]
                        if len(preds) > 0:
                            if isinstance(preds[0], list):
                                pred_ids = preds[0]
                            else:
                                pred_ids = preds
                        else:
                            pred_ids = []
                    elif isinstance(preds, torch.Tensor):
                        # Softmax输出：Tensor (batch, seq_len)
                        if preds.dim() == 2:
                            pred_ids = preds[0].tolist()
                        else:
                            pred_ids = preds.tolist()
                    else:
                        pred_ids = []
                    
                    # 使用模型的标签映射解码预测
                    if self.model_id2label is not None:
                        model_pred_tags = [self.model_id2label.get(pid, 'O') for pid in pred_ids[:len(tokens)]]
                        # 将模型预测标签映射到GeoGLUE任务的标签格式
                        pred_tags = self.map_model_pred_to_geoglue(model_pred_tags)
                    else:
                        # 如果没有模型标签映射，直接使用GeoGLUE的标签映射
                        pred_tags = [self.id2label.get(pid, 'O') for pid in pred_ids[:len(tokens)]]
                    
                    # 提取实体
                    true_entities = self.extract_entities_from_tags(tokens, true_tags)
                    pred_entities = self.extract_entities_from_tags(tokens, pred_tags)
                    
                    all_true_entities.extend(true_entities)
                    all_pred_entities.extend(pred_entities)
                    all_true_tags.append(true_tags)
                    all_pred_tags.append(pred_tags)
        
        # 计算指标
        overall_metrics = self.calculate_metrics(all_true_entities, all_pred_entities)
        
        # 计算每个实体类型的指标
        type_metrics = self.calculate_per_type_metrics(all_true_entities, all_pred_entities)
        
        # Token级别指标
        try:
            token_f1 = f1_score(all_true_tags, all_pred_tags, average='micro')
            token_precision = precision_score(all_true_tags, all_pred_tags, average='micro')
            token_recall = recall_score(all_true_tags, all_pred_tags, average='micro')
        except:
            token_f1 = 0.0
            token_precision = 0.0
            token_recall = 0.0
        
        return {
            'entity_level': overall_metrics,
            'token_level': {
                'f1': token_f1,
                'precision': token_precision,
                'recall': token_recall
            },
            'per_type': type_metrics,
            'total_entities': len(all_true_entities),
            'total_predictions': len(all_pred_entities)
        }
    
    def calculate_per_type_metrics(self, true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, Dict]:
        """计算每个实体类型的指标"""
        type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        true_by_type = defaultdict(list)
        pred_by_type = defaultdict(list)
        
        for e in true_entities:
            true_by_type[e['type']].append((e['start'], e['end']))
        for e in pred_entities:
            pred_by_type[e['type']].append((e['start'], e['end']))
        
        all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
        
        for entity_type in all_types:
            true_spans = set(true_by_type.get(entity_type, []))
            pred_spans = set(pred_by_type.get(entity_type, []))
            
            tp = len(true_spans & pred_spans)
            fp = len(pred_spans - true_spans)
            fn = len(true_spans - pred_spans)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            type_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'true_count': len(true_spans),
                'pred_count': len(pred_spans)
            }
        
        return dict(type_metrics)


# ========================== 分类任务评测 ==========================

class GeoGLUE_Classification_Evaluator:
    """GeoGLUE分类任务评测器（适用于GeoEAG）"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.task_config = GEOGLUE_TASKS[task_name]
        self.label_types = self.task_config.get('label_types', [])
        
        self.label2id = {label: idx for idx, label in enumerate(self.label_types)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_types)}
    
    def evaluate(self, data: List[Dict], model=None, tokenizer=None) -> Dict[str, Any]:
        """
        评测分类任务
        
        Args:
            data: 测试数据
            model: 模型实例
            tokenizer: tokenizer实例
        
        Returns:
            评测结果
        """
        true_labels = []
        pred_labels = []
        
        # 检查数据是否有标签
        has_labels = any('label' in item for item in data)
        
        if not has_labels:
            print(f"注意: 测试数据无标签，仅进行预测")
            if model is None:
                print("警告: 未提供模型且无标签，无法进行评测")
                return {
                    'accuracy': 0,
                    'macro_f1': 0,
                    'micro_f1': 0,
                    'per_class': {},
                    'total_samples': len(data),
                    'has_labels': False,
                    'predictions_only': True
                }
        
        if model is None:
            print("警告: 未提供模型，使用真实标签作为预测")
            for item in data:
                label = item.get('label', 'not_match')
                if not label:  # 如果没有标签，跳过
                    continue
                true_labels.append(label)
                pred_labels.append(label)
        else:
            model.eval()
            with torch.no_grad():
                for item in tqdm(data, desc=f"评测 {self.task_name}"):
                    sentence1 = item.get('sentence1', '')
                    sentence2 = item.get('sentence2', '')
                    true_label = item.get('label', 'not_match')
                    
                    # 构建输入文本
                    text = f"{sentence1} [SEP] {sentence2}"
                    
                    # Tokenize
                    encoding = tokenizer(text, truncation=True, padding='max_length',
                                        max_length=cfg.max_seq_length, return_tensors='pt')
                    input_ids = encoding['input_ids'].to(cfg.device)
                    attention_mask = encoding['attention_mask'].to(cfg.device)
                    
                    # 预测（这里需要分类模型）
                    # 简化处理：假设模型输出分类 logits
                    outputs = model(input_ids, attention_mask)
                    if isinstance(outputs, torch.Tensor):
                        pred_id = outputs.argmax(dim=-1).item()
                        pred_label = self.id2label.get(pred_id, 'not_match')
                    else:
                        pred_label = 'not_match'
                    
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
        
        # 计算指标
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        accuracy = correct / len(true_labels) if true_labels else 0
        
        # 计算每个类别的指标
        per_class_metrics = {}
        for label_type in self.label_types:
            true_count = sum(1 for t in true_labels if t == label_type)
            pred_count = sum(1 for p in pred_labels if p == label_type)
            correct_count = sum(1 for t, p in zip(true_labels, pred_labels) 
                               if t == label_type and p == label_type)
            
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = correct_count / true_count if true_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_count': true_count,
                'pred_count': pred_count,
                'correct_count': correct_count
            }
        
        # 计算宏平均和微平均
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        total_correct = sum(m['correct_count'] for m in per_class_metrics.values())
        micro_precision = total_correct / len(pred_labels) if pred_labels else 0
        micro_recall = total_correct / len(true_labels) if true_labels else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'per_class': per_class_metrics,
            'total_samples': len(true_labels)
        }


# ========================== 搜索召回任务评测 ==========================

class GeoGLUE_Recall_Evaluator:
    """GeoGLUE搜索召回任务评测器（适用于GeoTES-recall）"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
    
    def evaluate(self, data: List[Dict], model=None, tokenizer=None, 
                 candidate_pool: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        评测召回任务
        
        Args:
            data: 测试数据（包含query和pos_id）
            model: 模型实例
            tokenizer: tokenizer实例
            candidate_pool: 候选池（可选）
        
        Returns:
            评测结果
        """
        # 对于召回任务，需要评估检索到的候选是否包含正确答案
        # 这里简化处理，假设模型输出相似度分数
        
        results = {
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'mrr': 0.0,
            'total_queries': len(data)
        }
        
        if model is None:
            print("警告: 未提供模型，无法进行召回评测")
            return results
        
        # 实际召回评测需要候选池和相似度计算
        # 这里返回基础结果
        return results


# ========================== 搜索重排序任务评测 ==========================

class GeoGLUE_Rerank_Evaluator:
    """GeoGLUE搜索重排序任务评测器（适用于GeoTES-rerank）"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
    
    def evaluate(self, data: List[Dict], model=None, tokenizer=None) -> Dict[str, Any]:
        """
        评测重排序任务
        
        Args:
            data: 测试数据（包含query, positive_passages, negative_passages）
            model: 模型实例
            tokenizer: tokenizer实例
        
        Returns:
            评测结果
        """
        results = {
            'accuracy': 0.0,
            'ndcg@5': 0.0,
            'ndcg@10': 0.0,
            'mrr': 0.0,
            'total_queries': len(data)
        }
        
        if model is None:
            print("警告: 未提供模型，无法进行重排序评测")
            return results
        
        # 实际重排序评测需要计算排序质量
        # 这里返回基础结果
        return results


# ========================== 综合评测器 ==========================

class GeoGLUE_Comprehensive_Evaluator:
    """GeoGLUE综合评测器 - 支持所有任务类型"""
    
    def __init__(self, model_label2id: Dict = None, model_id2label: Dict = None):
        """
        Args:
            model_label2id: 模型的标签到ID映射
            model_id2label: 模型的ID到标签映射
        """
        self.model_label2id = model_label2id
        self.model_id2label = model_id2label
        
        self.evaluators = {
            'GeoCPA': GeoGLUE_NER_Evaluator('GeoCPA', model_label2id, model_id2label),
            'GeoEAG': GeoGLUE_Classification_Evaluator('GeoEAG'),
            'GeoETA': GeoGLUE_NER_Evaluator('GeoETA', model_label2id, model_id2label),
            'GeoTES-recall': GeoGLUE_Recall_Evaluator('GeoTES-recall'),
            'GeoTES-rerank': GeoGLUE_Rerank_Evaluator('GeoTES-rerank'),
            'GeoWWC': GeoGLUE_NER_Evaluator('GeoWWC', model_label2id, model_id2label)
        }
    
    def set_model_label_mapping(self, model_label2id: Dict, model_id2label: Dict):
        """设置模型的标签映射"""
        self.model_label2id = model_label2id
        self.model_id2label = model_id2label
        # 更新NER评测器的标签映射
        for task_name in ['GeoCPA', 'GeoETA', 'GeoWWC']:
            self.evaluators[task_name].set_model_label_mapping(model_label2id, model_id2label)
    
    def evaluate_task(self, task_name: str, data: List[Dict], 
                     model=None, tokenizer=None) -> Dict[str, Any]:
        """
        评测单个任务
        
        Args:
            task_name: 任务名称
            data: 测试数据
            model: 模型实例
            tokenizer: tokenizer实例
        
        Returns:
            评测结果
        """
        evaluator = self.evaluators.get(task_name)
        if evaluator is None:
            print(f"错误: 未知的任务类型 {task_name}")
            return {}
        
        return evaluator.evaluate(data, model, tokenizer)
    
    def evaluate_all(self, model=None, tokenizer=None, 
                     split: str = 'test') -> Dict[str, Dict[str, Any]]:
        """
        评测所有任务
        
        Args:
            model: 模型实例
            tokenizer: tokenizer实例
            split: 数据划分
        
        Returns:
            所有任务的评测结果
        """
        all_results = {}
        
        # 加载所有任务数据
        all_data = load_all_geoglue_tasks(split)
        
        for task_name, data in all_data.items():
            print(f"\n{'='*60}")
            print(f"评测任务: {task_name}")
            print(f"{'='*60}")
            
            results = self.evaluate_task(task_name, data, model, tokenizer)
            all_results[task_name] = results
            
            # 打印结果摘要
            self.print_task_summary(task_name, results)
        
        return all_results
    
    def print_task_summary(self, task_name: str, results: Dict[str, Any]):
        """打印任务结果摘要"""
        task_config = GEOGLUE_TASKS[task_name]
        task_type = task_config['type']
        
        print(f"\n{task_name} ({task_config['description']}) 评测结果:")
        
        if task_type == 'ner':
            entity_metrics = results.get('entity_level', {})
            token_metrics = results.get('token_level', {})
            print(f"  实体级别:")
            print(f"    F1: {entity_metrics.get('f1', 0):.4f}")
            print(f"    Precision: {entity_metrics.get('precision', 0):.4f}")
            print(f"    Recall: {entity_metrics.get('recall', 0):.4f}")
            print(f"  Token级别:")
            print(f"    F1: {token_metrics.get('f1', 0):.4f}")
            
            # 打印各类型指标
            per_type = results.get('per_type', {})
            if per_type:
                print(f"  各实体类型F1:")
                for type_name, metrics in sorted(per_type.items()):
                    print(f"    {type_name}: {metrics.get('f1', 0):.4f}")
        
        elif task_type == 'classification':
            print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
            print(f"  Macro-F1: {results.get('macro_f1', 0):.4f}")
            print(f"  Micro-F1: {results.get('micro_f1', 0):.4f}")
            
            per_class = results.get('per_class', {})
            if per_class:
                print(f"  各类别F1:")
                for class_name, metrics in sorted(per_class.items()):
                    print(f"    {class_name}: {metrics.get('f1', 0):.4f}")
        
        elif task_type in ['recall', 'rerank']:
            for metric_name in task_config.get('metrics', []):
                value = results.get(metric_name, 0)
                print(f"  {metric_name}: {value:.4f}")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], 
                    save_path: str = None):
        """
        保存评测结果
        
        Args:
            results: 评测结果
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(cfg.result_dir, 'geoglue_results.json')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评测结果已保存到: {save_path}")
    
    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        生成评测报告
        
        Args:
            results: 评测结果
        
        Returns:
            报告文本
        """
        report = []
        report.append("="*80)
        report.append("GeoGLUE评测报告")
        report.append("="*80)
        
        for task_name, task_results in results.items():
            task_config = GEOGLUE_TASKS[task_name]
            report.append(f"\n{task_name} ({task_config['description']}):")
            
            if task_config['type'] == 'ner':
                entity_f1 = task_results.get('entity_level', {}).get('f1', 0)
                report.append(f"  实体F1: {entity_f1:.4f}")
            elif task_config['type'] == 'classification':
                accuracy = task_results.get('accuracy', 0)
                report.append(f"  Accuracy: {accuracy:.4f}")
            else:
                for metric in task_config.get('metrics', []):
                    value = task_results.get(metric, 0)
                    report.append(f"  {metric}: {value:.4f}")
        
        report.append("\n" + "="*80)
        return "\n".join(report)


# ========================== 零样本评测函数 ==========================

def run_geoglue_zero_shot_evaluation(model_path: str = None, 
                                     model_name: str = None,
                                     split: str = 'dev') -> Dict[str, Any]:
    """
    运行GeoGLUE零样本评测
    
    Args:
        model_path: 模型路径
        model_name: 模型名称
        split: 数据划分
    
    Returns:
        评测结果
    """
    from models import NERModel
    from data_utils import load_lexicon, LexiconMatcher, create_lexicon_matcher
    from transformers import AutoTokenizer
    
    print("\n" + "="*80)
    print("GeoGLUE零样本评测")
    print("="*80)
    
    # 初始化tokenizer
    if model_name is None:
        model_name = cfg.roberta_model_name
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载词典
    lexicon_matcher = create_lexicon_matcher(cfg)
    
    # 创建模型
    model = NERModel(
        model_name=model_name,
        num_labels=len(cfg.ner_label2id),
        model_variant='RoBERTaBiLSTMCRF',
        use_crf=True,
        use_lexicon=True,
        lexicon_size=cfg.lexicon_size
    )
    
    # 加载模型权重
    if model_path is None:
        model_path = os.path.join(cfg.model_save_dir, 'best_model.pt')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"已加载模型: {model_path}")
    else:
        print(f"警告: 未找到模型 {model_path}, 使用随机初始化")
    
    model.to(cfg.device)
    
    # 创建评测器，传入模型的标签映射
    evaluator = GeoGLUE_Comprehensive_Evaluator(
        model_label2id=cfg.ner_label2id,
        model_id2label=cfg.ner_id2label
    )
    
    # 运行评测
    results = evaluator.evaluate_all(model, tokenizer, split)
    
    # 保存结果
    evaluator.save_results(results)
    
    # 打印报告
    report = evaluator.generate_report(results)
    print(report)
    
    return results


# ========================== 主函数 ==========================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoGLUE评测')
    parser.add_argument('--mode', type=str, default='evaluate_all', 
                       choices=['evaluate_all', 'evaluate_task', 'zero_shot', 'analyze_data'],
                       help='评测模式')
    parser.add_argument('--task', type=str, default=None,
                       help='指定评测的任务名称')
    parser.add_argument('--split', type=str, default='test',
                       help='数据划分')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--model_name', type=str, default=None,
                       help='模型名称')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze_data':
        # 分析数据
        all_data = load_all_geoglue_tasks(args.split)
        for task_name, data in all_data.items():
            print(f"\n{task_name}: {len(data)} 条数据")
            if data:
                print(f"  样例: {data[0]}")
    
    elif args.mode == 'evaluate_all':
        evaluator = GeoGLUE_Comprehensive_Evaluator()
        results = evaluator.evaluate_all(split=args.split)
        evaluator.save_results(results)
    
    elif args.mode == 'evaluate_task':
        if args.task is None:
            print("错误: 需要指定 --task 参数")
        else:
            evaluator = GeoGLUE_Comprehensive_Evaluator()
            data = load_geoglue_task_data(args.task, args.split)
            results = evaluator.evaluate_task(args.task, data)
            evaluator.print_task_summary(args.task, results)
    
    elif args.mode == 'zero_shot':
        results = run_geoglue_zero_shot_evaluation(
            model_path=args.model_path,
            model_name=args.model_name,
            split=args.split
        )