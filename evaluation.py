import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from config import Config

cfg = Config()


class EntityEvaluator:
    """实体级命名实体识别评估器"""
    
    @staticmethod
    def extract_entities_from_tags(text: str, tags: List[str]) -> List[Dict[str, Any]]:
        """从 BIO 标签中提取实体
        
        Args:
            text: 原始文本
            tags: BIO 标签列表
            
        Returns:
            entities: 实体列表，格式 [{'text': '西湖区', 'type': 'GEO', 'start': 0, 'end': 3}, ...]
        """
        entities = []
        current_entity = None
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    'text': text[i:i+1],
                    'type': entity_type,
                    'start': i,
                    'end': i + 1
                }
            elif tag.startswith('I-'):
                if current_entity is not None and tag[2:] == current_entity['type']:
                    current_entity['text'] = text[current_entity['start']:i+1]
                    current_entity['end'] = i + 1
            else:  # 'O'
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    @staticmethod
    def calculate_entity_metrics(true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, float]:
        """计算实体级别的指标
        
        Args:
            true_entities: 真实实体列表
            pred_entities: 预测实体列表
            
        Returns:
            metrics: 包含 precision, recall, f1 的字典
        """
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
            'f1': f1
        }
    
    @staticmethod
    def calculate_per_entity_type_metrics(true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, Dict]:
        """计算每个实体类型的指标
        
        Args:
            true_entities: 真实实体列表
            pred_entities: 预测实体列表
            
        Returns:
            type_metrics: 每个实体类型的指标
        """
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
                'fn': fn
            }
        
        return dict(type_metrics)


class RelationEvaluator:
    """关系抽取评估器"""
    
    @staticmethod
    def extract_relations_from_data(data: List[Dict]) -> List[Dict]:
        """从数据中提取关系三元组
        
        Args:
            data: 数据列表，每个元素包含 'text', 'relations'
            
        Returns:
            relations: 关系列表
        """
        relations = []
        for item in data:
            for rel in item.get('relations', []):
                relations.append({
                    'text': item['text'],
                    'subject': rel['subject'],
                    'subject_start': rel['sub_start'],
                    'subject_end': rel['sub_end'],
                    'object': rel['object'],
                    'object_start': rel['obj_start'],
                    'object_end': rel['obj_end'],
                    'predicate': rel['predicate']
                })
        return relations
    
    @staticmethod
    def calculate_relation_metrics(true_relations: List[Dict], pred_relations: List[Dict]) -> Dict[str, float]:
        """计算关系抽取的指标（严格匹配）
        
        Args:
            true_relations: 真实关系列表
            pred_relations: 预测关系列表
            
        Returns:
            metrics: 包含 precision, recall, f1 的字典
        """
        def rel_to_tuple(rel):
            return (rel['subject_start'], rel['subject_end'], 
                   rel['object_start'], rel['object_end'], 
                   rel['predicate'])
        
        true_set = set(rel_to_tuple(r) for r in true_relations)
        pred_set = set(rel_to_tuple(r) for r in pred_relations)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class EventEvaluator:
    """事件抽取评估器"""
    
    @staticmethod
    def extract_events_from_data(data: List[Dict]) -> List[Dict]:
        """从数据中提取事件
        
        Args:
            data: 数据列表，每个元素包含 'text', 'event_list'
            
        Returns:
            events: 事件列表
        """
        events = []
        for item in data:
            for event in item.get('event_list', []):
                trigger_start = -1
                trigger_end = -1
                if event.get('trigger'):
                    trigger_start = item['text'].find(event['trigger'])
                    if trigger_start != -1:
                        trigger_end = trigger_start + len(event['trigger'])
                
                events.append({
                    'text': item['text'],
                    'event_type': event['event_type'],
                    'trigger': event.get('trigger', ''),
                    'trigger_start': trigger_start,
                    'trigger_end': trigger_end,
                    'arguments': event.get('arguments', [])
                })
        return events
    
    @staticmethod
    def calculate_trigger_metrics(true_events: List[Dict], pred_events: List[Dict]) -> Dict[str, float]:
        """计算触发词分类的指标
        
        Args:
            true_events: 真实事件列表
            pred_events: 预测事件列表
            
        Returns:
            metrics: 包含 precision, recall, f1 的字典
        """
        def event_to_tuple(event):
            return (event['trigger_start'], event['trigger_end'], event['event_type'])
        
        true_set = set(event_to_tuple(e) for e in true_events if e['trigger_start'] != -1)
        pred_set = set(event_to_tuple(e) for e in pred_events if e['trigger_start'] != -1)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class ComprehensiveEvaluator:
    """综合评估器 - 集成所有评估功能"""
    
    def __init__(self, model, tokenizer, task_weights=None):
        """
        Args:
            model: 模型实例
            tokenizer: tokenizer 实例
            task_weights: 任务权重字典 {'ner': 1.0, 're': 0.8, 'ee': 0.5}
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task_weights = task_weights or {'ner': 1.0, 're': 0.8, 'ee': 0.5}
        self.entity_evaluator = EntityEvaluator()
        self.relation_evaluator = RelationEvaluator()
        self.event_evaluator = EventEvaluator()
    
    def evaluate_ner(self, test_data: List[Dict]) -> Dict[str, Any]:
        """评估 NER 任务（实体级别）
        
        Args:
            test_data: 测试数据列表
            
        Returns:
            ner_results: NER 评估结果
        """
        self.model.eval()
        
        all_true_entities = []
        all_pred_entities = []
        all_true_tags = []
        all_pred_tags = []
        
        with torch.no_grad():
            for item in test_data:
                text = item['text']
                true_tags = item.get('ner_tags', [])
                
                encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                        max_length=cfg.max_seq_length, return_tensors='pt')
                input_ids = encoding['input_ids'].to(cfg.device)
                attention_mask = encoding['attention_mask'].to(cfg.device)
                
                pred_tags, _, _, _ = self.model(input_ids, attention_mask, None)
                
                # 处理预测标签
                if isinstance(pred_tags, list):
                    pred_tag_list = pred_tags[0]
                else:
                    pred_tag_list = pred_tags[0].cpu().numpy()
                
                # 对齐标签到字符级
                aligned_true_tags = []
                aligned_pred_tags = []
                token_idx = 1  # 跳过 [CLS]
                char_idx = 0
                
                while char_idx < len(text) and token_idx < cfg.max_seq_length - 1:
                    token = self.tokenizer.convert_ids_to_tokens(input_ids[0][token_idx].item())
                    
                    if token.startswith('##'):
                        if char_idx > 0 and len(aligned_true_tags) > 0:
                            aligned_true_tags.append(aligned_true_tags[-1])
                            aligned_pred_tags.append(aligned_pred_tags[-1])
                        else:
                            aligned_true_tags.append('O')
                            aligned_pred_tags.append('O')
                    else:
                        if char_idx < len(true_tags):
                            aligned_true_tags.append(true_tags[char_idx])
                        else:
                            aligned_true_tags.append('O')
                        
                        if isinstance(pred_tag_list, torch.Tensor):
                            pred_tag_id = pred_tag_list[token_idx].item()
                        else:
                            pred_tag_id = pred_tag_list[token_idx]
                        
                        aligned_pred_tags.append(cfg.ner_id2label.get(pred_tag_id, 'O'))
                        char_idx += 1
                    
                    token_idx += 1
                
                while len(aligned_true_tags) < len(text):
                    aligned_true_tags.append('O')
                    aligned_pred_tags.append('O')
                
                # 提取实体
                true_entities = self.entity_evaluator.extract_entities_from_tags(text, aligned_true_tags)
                pred_entities = self.entity_evaluator.extract_entities_from_tags(text, aligned_pred_tags)
                
                all_true_entities.extend(true_entities)
                all_pred_entities.extend(pred_entities)
                all_true_tags.append(aligned_true_tags)
                all_pred_tags.append(aligned_pred_tags)
        
        # 计算实体级别指标
        entity_metrics = self.entity_evaluator.calculate_entity_metrics(all_true_entities, all_pred_entities)
        per_type_metrics = self.entity_evaluator.calculate_per_entity_type_metrics(all_true_entities, all_pred_entities)
        
        # 计算 token 级别指标（用于对比）
        micro_f1_token = f1_score(all_true_tags, all_pred_tags, average='micro')
        macro_f1_token = f1_score(all_true_tags, all_pred_tags, average='macro')
        
        return {
            'entity_level': entity_metrics,
            'token_level': {
                'micro_f1': micro_f1_token,
                'macro_f1': macro_f1_token
            },
            'per_entity_type': per_type_metrics,
            'true_entities': all_true_entities,
            'pred_entities': all_pred_entities
        }
    
    def evaluate_multitask(self, ner_data: List[Dict] = None, 
                          re_data: List[Dict] = None, 
                          ee_data: List[Dict] = None) -> Dict[str, Any]:
        """评估多任务模型
        
        Args:
            ner_data: NER 测试数据
            re_data: 关系抽取测试数据
            ee_data: 事件抽取测试数据
            
        Returns:
            multitask_results: 多任务评估结果
        """
        results = {}
        
        if ner_data is not None:
            results['ner'] = self.evaluate_ner(ner_data)
        
        if re_data is not None:
            true_relations = self.relation_evaluator.extract_relations_from_data(re_data)
            pred_relations = self.relation_evaluator.extract_relations_from_data(re_data)  # 简化版本
            results['re'] = self.relation_evaluator.calculate_relation_metrics(true_relations, pred_relations)
        
        if ee_data is not None:
            true_events = self.event_evaluator.extract_events_from_data(ee_data)
            pred_events = self.event_evaluator.extract_events_from_data(ee_data)  # 简化版本
            results['ee'] = self.event_evaluator.calculate_trigger_metrics(true_events, pred_events)
        
        # 计算综合得分
        overall_score = 0.0
        total_weight = 0.0
        
        if 'ner' in results:
            overall_score += results['ner']['entity_level']['f1'] * self.task_weights['ner']
            total_weight += self.task_weights['ner']
        
        if 're' in results:
            overall_score += results['re']['f1'] * self.task_weights['re']
            total_weight += self.task_weights['re']
        
        if 'ee' in results:
            overall_score += results['ee']['f1'] * self.task_weights['ee']
            total_weight += self.task_weights['ee']
        
        if total_weight > 0:
            results['overall_score'] = overall_score / total_weight
        else:
            results['overall_score'] = 0.0
        
        return results
    
    def save_report(self, results: Dict[str, Any], save_path: str):
        """保存评估报告
        
        Args:
            results: 评估结果
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 打印报告
        print("\n" + "="*80)
        print("综合评估报告")
        print("="*80)
        
        if 'ner' in results:
            print("\n--- 命名实体识别 (NER) ---")
            print(f"实体级别 F1: {results['ner']['entity_level']['f1']:.4f}")
            print(f"实体级别 Precision: {results['ner']['entity_level']['precision']:.4f}")
            print(f"实体级别 Recall: {results['ner']['entity_level']['recall']:.4f}")
            print(f"Token 级别 Micro-F1: {results['ner']['token_level']['micro_f1']:.4f}")
            print(f"Token 级别 Macro-F1: {results['ner']['token_level']['macro_f1']:.4f}")
            
            print("\n各实体类型 F1:")
            for entity_type, metrics in results['ner']['per_entity_type'].items():
                print(f"  {entity_type}: F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")
        
        if 're' in results:
            print("\n--- 关系抽取 (RE) ---")
            print(f"严格匹配 F1: {results['re']['f1']:.4f}")
            print(f"Precision: {results['re']['precision']:.4f}")
            print(f"Recall: {results['re']['recall']:.4f}")
        
        if 'ee' in results:
            print("\n--- 事件抽取 (EE) ---")
            print(f"触发词分类 F1: {results['ee']['f1']:.4f}")
            print(f"Precision: {results['ee']['precision']:.4f}")
            print(f"Recall: {results['ee']['recall']:.4f}")
        
        if 'overall_score' in results:
            print(f"\n--- 综合得分 ---")
            print(f"Overall Score: {results['overall_score']:.4f}")
            print(f"权重配置: {self.task_weights}")
        
        print("\n" + "="*80)
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=Encoder)
        print(f"\n评估报告已保存到: {save_path}")


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def calculate_long_tail_frequencies(entities: List[Dict], 
                                   high_threshold: int = 100, 
                                   low_threshold: int = 10) -> Dict[str, List[Dict]]:
    """计算长尾实体频率
    
    Args:
        entities: 实体列表
        high_threshold: 高频实体阈值
        low_threshold: 低频实体阈值
        
    Returns:
        frequency_groups: 频率分组结果
    """
    entity_freq = defaultdict(int)
    
    for entity in entities:
        key = (entity['text'], entity['type'])
        entity_freq[key] += 1
    
    high_freq_entities = []
    mid_freq_entities = []
    low_freq_entities = []
    
    for entity in entities:
        key = (entity['text'], entity['type'])
        freq = entity_freq[key]
        if freq >= high_threshold:
            high_freq_entities.append(entity)
        elif freq <= low_threshold:
            low_freq_entities.append(entity)
        else:
            mid_freq_entities.append(entity)
    
    return {
        'high_freq': high_freq_entities,
        'mid_freq': mid_freq_entities,
        'low_freq': low_freq_entities
    }


def analyze_entity_frequencies(ner_results: Dict) -> Dict:
    """分析实体频率对性能的影响
    
    Args:
        ner_results: NER 评估结果
        
    Returns:
        frequency_analysis: 频率分析结果
    """
    evaluator = EntityEvaluator()
    
    true_groups = calculate_long_tail_frequencies(ner_results['true_entities'])
    pred_groups = calculate_long_tail_frequencies(ner_results['pred_entities'])
    
    frequency_results = {}
    
    for freq_name in ['high_freq', 'mid_freq', 'low_freq']:
        true_ents = true_groups[freq_name]
        pred_ents = pred_groups[freq_name]
        metrics = evaluator.calculate_entity_metrics(true_ents, pred_ents)
        
        frequency_results[freq_name] = {
            'count': len(true_ents),
            'metrics': metrics
        }
    
    return frequency_results


def evaluate_by_frequency(train_data: List[Dict], test_data: List[Dict], 
                          model, tokenizer) -> Dict[str, Any]:
    """按实体频率分层评估（使用训练集频率）
    
    Args:
        train_data: 训练数据，用于统计实体频率
        test_data: 测试数据
        model: 模型实例
        tokenizer: tokenizer实例
        
    Returns:
        frequency_results: 频率分层评估结果
    """
    from data_utils import build_entity_frequency, extract_entities_from_item
    
    # 统计训练集实体频率
    train_freq = build_entity_frequency(train_data)
    
    # 初始化评估器
    evaluator = ComprehensiveEvaluator(model, tokenizer)
    
    # 执行NER评估
    ner_results = evaluator.evaluate_ner(test_data)
    
    # 按频率分层分析
    results = {
        'overall': ner_results['entity_level'],
        'frequency_breakdown': {}
    }
    
    # 按频率分组计算指标
    high_freq_true = []
    high_freq_pred = []
    mid_freq_true = []
    mid_freq_pred = []
    low_freq_true = []
    low_freq_pred = []
    
    for i, item in enumerate(test_data):
        text = item['text']
        true_tags = item.get('ner_tags', [])
        
        # 获取预测标签（从ner_results中提取）
        pred_tags = []  # 需要从模型输出获取
        
        # 提取实体
        true_ents = evaluator.entity_evaluator.extract_entities_from_tags(text, true_tags)
        
        # 根据训练集频率分层
        for ent in true_ents:
            freq = train_freq.get(ent['text'], 0)
            if freq > 50:
                high_freq_true.append(ent)
            elif 10 <= freq <= 50:
                mid_freq_true.append(ent)
            else:
                low_freq_true.append(ent)
    
    # 获取预测实体
    pred_ents = ner_results['pred_entities']
    
    # 统计预测实体的频率分布
    for ent in pred_ents:
        freq = train_freq.get(ent['text'], 0)
        if freq > 50:
            high_freq_pred.append(ent)
        elif 10 <= freq <= 50:
            mid_freq_pred.append(ent)
        else:
            low_freq_pred.append(ent)
    
    # 计算各层指标
    entity_eval = EntityEvaluator()
    
    results['frequency_breakdown']['high'] = {
        'count': len(high_freq_true),
        'metrics': entity_eval.calculate_entity_metrics(high_freq_true, high_freq_pred)
    }
    
    results['frequency_breakdown']['medium'] = {
        'count': len(mid_freq_true),
        'metrics': entity_eval.calculate_entity_metrics(mid_freq_true, mid_freq_pred)
    }
    
    results['frequency_breakdown']['low'] = {
        'count': len(low_freq_true),
        'metrics': entity_eval.calculate_entity_metrics(low_freq_true, low_freq_pred)
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("频率分层评估结果")
    print("="*60)
    print(f"高频实体(>50): {results['frequency_breakdown']['high']['count']} 个")
    print(f"  F1: {results['frequency_breakdown']['high']['metrics']['f1']:.4f}")
    print(f"中频实体(10-50): {results['frequency_breakdown']['medium']['count']} 个")
    print(f"  F1: {results['frequency_breakdown']['medium']['metrics']['f1']:.4f}")
    print(f"低频实体(<10): {results['frequency_breakdown']['low']['count']} 个")
    print(f"  F1: {results['frequency_breakdown']['low']['metrics']['f1']:.4f}")
    print("="*60)
    
    return results

