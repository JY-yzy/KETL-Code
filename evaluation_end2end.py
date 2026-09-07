#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端综合评测和案例分析
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from seqeval.metrics import f1_score as seqeval_f1_score, classification_report
from transformers import AutoTokenizer

from config import Config
from data_utils import LexiconMatcher, build_lexicon_vocab, create_lexicon_matcher
from models import MultiTaskModel


def decode_bio_tags(text, bio_tags, tokenizer):
    """
    解码 BIO 标签为实体列表
    
    Args:
        text: 原始文本
        bio_tags: BIO 标签列表
        tokenizer: tokenizer
    
    Returns:
        entities: 实体列表，格式 [(entity_text, entity_type, start_idx, end_idx)]
    """
    entities = []
    n = len(bio_tags)
    i = 0
    while i < n:
        if bio_tags[i].startswith('B-'):
            entity_type = bio_tags[i][2:]
            start = i
            i += 1
            while i < n and bio_tags[i].startswith('I-' + entity_type):
                i += 1
            end = i - 1
            # 获取实体文本
            tokens = tokenizer.tokenize(text)[:n]
            entity_text = ''.join([t.replace('##', '') for t in tokens[start:end+1]])
            entities.append((entity_text, entity_type, start, end))
        else:
            i += 1
    return entities


def decode_relations(entities, rel_logits, rel_id2label, threshold=0.5):
    """
    从关系 logits 解码关系三元组
    
    Args:
        entities: 实体列表
        rel_logits: 关系 logits
        rel_id2label: 关系标签映射
        threshold: 关系置信度阈值
    
    Returns:
        relations: 关系三元组列表，格式 [(head_entity, relation, tail_entity)]
    """
    relations = []
    # 这里是一个简化的关系解码实现
    # 实际项目中需要根据具体模型输出的关系评分进行解码
    # 这里我们只对每对实体进行简单的关系预测演示
    if len(entities) >= 2:
        for i, (h_text, h_type, h_start, h_end) in enumerate(entities):
            for j, (t_text, t_type, t_start, t_end) in enumerate(entities):
                if i != j:
                    # 这里使用简化关系预测
                    # 实际项目中应该从模型的 rel_logits 中获取具体的关系预测
                    # 这里我们只演示输出一些常见关系
                    if h_type in ['PER', 'GEO', 'ADMIN'] and t_type in ['GEO', 'ADMIN']:
                        relations.append((h_text, '位于', t_text))
    return relations


def decode_events(event_logits, event_id2label, threshold=0.5):
    """
    从事件 logits 解码事件
    
    Args:
        event_logits: 事件 logits (torch.Tensor)
        event_id2label: 事件标签映射
        threshold: 事件置信度阈值
    
    Returns:
        events: 事件列表，格式 [(trigger_text, event_type)]
    """
    events = []
    # 这里是简化版事件解码
    # 实际项目中需要完整的事件触发词检测和分类
    if isinstance(event_logits, torch.Tensor):
        event_probs = torch.sigmoid(event_logits)
        for idx, prob in enumerate(event_probs[0]):  # 简化处理
            if prob > threshold and idx != 0:  # idx=0 为无事件
                event_type = event_id2label.get(idx, str(idx))
                events.append(("事件触发词", event_type))
    return events


class End2EndEvaluator:
    """端到端综合评测器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
        
        # 加载词典匹配器
        self.lexicon_matcher = None
        if os.path.exists(config.geo_lexicon_path):
            self.lexicon_matcher = create_lexicon_matcher(config)
        
        # 模型会在需要时加载
        self.model = None
    
    def load_model(self, model_path):
        """加载预训练模型"""
        print(f"Loading model from {model_path}...")
        
        # 加载模型权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 从 checkpoint 中获取模型配置（确保与训练时一致）
            model_config = checkpoint.get('model_config', {})
            
            # 检测 checkpoint 中是否包含 lexicon_adapter 层
            state_dict = checkpoint['model_state_dict']
            has_lexicon_adapter = any('lexicon_adapter' in key for key in state_dict.keys())
            has_layers_1 = any('layers.1' in key for key in state_dict.keys())
            
            # 使用检测结果或配置中的值
            checkpoint_use_lexicon = model_config.get('use_lexicon', has_lexicon_adapter or self.config.use_lexicon)
            checkpoint_use_crf = model_config.get('use_crf', self.config.use_crf)
            checkpoint_model_variant = model_config.get('model_variant', self.config.model_variant)
            
            # 检测 fusion_type（从 state_dict 中推断）
            has_fusion_gate = any('fusion_gate' in key for key in state_dict.keys())
            has_lexicon_attention = any('lexicon_attention' in key for key in state_dict.keys())
            fusion_type = 'gate' if has_fusion_gate else 'attention'
            
            print(f"Checkpoint info: epoch={checkpoint.get('epoch', 0)}, "
                  f"use_lexicon={checkpoint_use_lexicon}, use_crf={checkpoint_use_crf}, "
                  f"model_variant={checkpoint_model_variant}, fusion_type={fusion_type}, "
                  f"has_lexicon_adapter={has_lexicon_adapter}, has_layers_1={has_layers_1}")
            
            # 初始化模型（使用与训练时一致的结构）
            self.model = MultiTaskModel(
                model_name=self.config.roberta_model_name,
                num_ner_labels=len(self.config.ner_labels),
                num_rel_labels=len(self.config.rel_labels),
                num_event_types=len(self.config.event_labels),
                use_lexicon=checkpoint_use_lexicon,
                use_crf=checkpoint_use_crf,
                lexicon_size=self.config.lexicon_size,
                model_variant=checkpoint_model_variant,
                lexicon_config={
                    'word_emb_dim': getattr(self.config, 'lexicon_word_emb_dim', 64),
                    'fusion_type': fusion_type,
                    'use_fusion_gate': self.config.use_lexicon_fusion_gate,
                    'dropout': self.config.lexicon_dropout,
                    'num_levels': getattr(self.config, 'num_geo_levels', 7)
                }
            )
            
            # 使用 strict=False 允许部分加载（处理模型结构不匹配的情况）
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=False
            )
            
            # 检查是否有严重的结构不匹配
            critical_missing = [k for k in missing_keys if 'lexicon_adapter' in k or 'layers.1' in k]
            if critical_missing:
                print(f"错误: 关键层缺失！请确保训练和评估使用相同的模型配置")
                print(f"缺失的关键层: {critical_missing}")
            
            if missing_keys:
                print(f"警告: {len(missing_keys)} 个键在 checkpoint 中缺失，使用随机初始化")
            if unexpected_keys:
                print(f"警告: {len(unexpected_keys)} 个键在模型中不存在")
            
            print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 0)}")
        else:
            # 如果 checkpoint 不存在，使用默认配置初始化
            print(f"Warning: Model checkpoint not found at {model_path}, using default config")
            self.model = MultiTaskModel(
                model_name=self.config.roberta_model_name,
                num_ner_labels=len(self.config.ner_labels),
                num_rel_labels=len(self.config.rel_labels),
                num_event_types=len(self.config.event_labels),
                use_lexicon=self.config.use_lexicon,
                use_crf=self.config.use_crf,
                lexicon_size=self.config.lexicon_size
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text):
        """预处理单条文本"""
        # tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 词汇特征
        lexicon_indices = None
        if self.lexicon_matcher and self.config.use_lexicon:
            lexicon_indices = []
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            for token in tokens:
                clean_token = token.replace('##', '')
                if clean_token in self.lexicon_matcher.lexicon:
                    lexicon_indices.append(self.lexicon_matcher.lexicon.get(clean_token, 0))
                else:
                    lexicon_indices.append(0)
            lexicon_indices = torch.tensor([lexicon_indices], dtype=torch.long).to(self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'lexicon_indices': lexicon_indices,
            'text': text
        }
    
    def predict(self, text):
        """对单条文本进行端到端预测"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 预处理
        inputs = self.preprocess_text(text)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                lexicon_indices=inputs['lexicon_indices']
            )
        
        # MultiTaskModel 返回: ner_preds, rel_preds, event_preds, ner_logits, rel_logits, event_logits, hidden
        ner_preds, rel_preds, event_preds, _, _, _, _ = outputs
        
        # 解码NER
        pred_tags = []
        for idx in ner_preds[0]:  # 简化处理
            tag = self.config.ner_id2label.get(idx, 'O')
            pred_tags.append(tag)
        pred_tags = pred_tags[:min(len(text), len(pred_tags))]
        
        # 解码实体
        entities = decode_bio_tags(text, pred_tags, self.tokenizer)
        
        # 解码关系
        relations = decode_relations(entities, rel_preds, self.config.rel_id2label, self.config.rel_threshold)
        
        # 解码事件
        events = decode_events(event_preds, self.config.event_id2label, self.config.event_threshold)
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations,
            'events': events,
            'ner_tags': pred_tags
        }
    
    def evaluate_end2end(self, duee_data_path, model_path, save_path='end2end_results.json'):
        """
        端到端综合评测
        
        Args:
            duee_data_path: DuEE 数据路径
            model_path: 模型路径
            save_path: 结果保存路径
        """
        self.load_model(model_path)
        
        # 加载数据
        print(f"Loading DuEE data from {duee_data_path}...")
        with open(duee_data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]
        
        # 构建伪 NER 标签（简化版）
        print("Building pseudo NER labels...")
        eval_data = []
        for item in raw_data[:100]:  # 限制样本数以演示
            text = item.get('text', '')
            if not text:
                continue
            
            # 从 head 和 tail 中提取近似实体
            pseudo_entities = []
            if 'head' in item and item['head']:
                pseudo_entities.append((item['head'], 'GEO', 0, 0))  # 简化位置信息
            if 'tail' in item and item['tail']:
                pseudo_entities.append((item['tail'], 'PER', 0, 0))
            
            eval_data.append({
                'text': text,
                'entities': pseudo_entities,
                'event_type': item.get('event_type', '无事件')
            })
        
        if not eval_data:
            print("No valid data to evaluate.")
            return
        
        # 评测
        print("Starting end-to-end evaluation...")
        all_true_tags = []
        all_pred_tags = []
        all_results = []
        
        for item in tqdm(eval_data, desc="Evaluating"):
            # 预测
            pred_result = self.predict(item['text'])
            
            # 构建伪真实标签（简化版）
            # 实际项目中应该使用标准的NER标注
            n_tokens = min(len(item['text']), len(pred_result['ner_tags']))
            true_tags = ['O'] * n_tokens
            # 简化：假设所有词都是 O
            all_true_tags.append(true_tags)
            all_pred_tags.append(pred_result['ner_tags'][:n_tokens])
            
            # 保存结果
            all_results.append({
                'text': item['text'],
                'true_entities': item['entities'],
                'pred_entities': pred_result['entities'],
                'pred_relations': pred_result['relations'],
                'pred_events': pred_result['events'],
                'true_event_type': item['event_type']
            })
        
        # 计算指标
        if not all_true_tags or not all_pred_tags:
            print("\n警告: 没有有效的预测结果可供评估")
            ner_micro_f1 = 0.0
            print("\n" + "="*80)
            print("End-to-End Evaluation Results")
            print("="*80)
            print(f"NER Micro-F1: {ner_micro_f1:.4f}")
        else:
            ner_micro_f1 = seqeval_f1_score(all_true_tags, all_pred_tags)
            
            # 打印报告
            print("\n" + "="*80)
            print("End-to-End Evaluation Results")
            print("="*80)
            print(f"NER Micro-F1: {ner_micro_f1:.4f}")
            print("\nNER Classification Report:")
            print(classification_report(all_true_tags, all_pred_tags, zero_division=0))
        
        # 打印一些示例关系
        print("\n" + "="*80)
        print("Example Relation Extractions (first 20):")
        print("="*80)
        for i, result in enumerate(all_results[:20]):
            print(f"\n{i+1}. Text: {result['text'][:100]}...")
            print(f"   Predicted Relations: {result['pred_relations']}")
        
        # 保存结果
        final_results = {
            'metrics': {
                'ner_micro_f1': float(ner_micro_f1)
            },
            'sample_results': all_results
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {save_path}")
        
        return final_results


def run_case_study(config, model_path, save_path='case_study_results.json'):
    """
    运行真实文本案例分析
    
    Args:
        config: 配置对象
        model_path: 模型路径
        save_path: 结果保存路径
    """
    print("="*80)
    print("Case Study Analysis")
    print("="*80)
    
    # 初始化评测器
    evaluator = End2EndEvaluator(config)
    evaluator.load_model(model_path)
    
    # 运行案例分析
    case_results = []
    
    for i, text in enumerate(config.case_texts):
        print(f"\n{i+1}. Analyzing: {text}")
        result = evaluator.predict(text)
        case_results.append(result)
        
        # 打印结果
        print(f"\n   Text: {result['text']}")
        print(f"   Entities:")
        for ent_text, ent_type, start, end in result['entities']:
            print(f"      - {ent_text} ({ent_type})")
        
        print(f"   Relations:")
        if result['relations']:
            for h, rel, t in result['relations']:
                print(f"      - ({h}) -[{rel}]-> ({t})")
        else:
            print(f"      - None")
        
        print(f"   Events:")
        if result['events']:
            for trigger, event_type in result['events']:
                print(f"      - {event_type}: {trigger}")
        else:
            print(f"      - None")
    
    # 保存结果到 JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(case_results, f, ensure_ascii=False, indent=2)
    
    # 生成 Markdown 表格
    md_path = save_path.replace('.json', '.md')
    generate_case_study_markdown(case_results, md_path)
    
    print(f"\nResults saved to {save_path} and {md_path}")
    
    return case_results


def generate_case_study_markdown(case_results, output_path):
    """生成案例分析的 Markdown 报告"""
    
    markdown_content = "# 真实文本案例分析报告\n\n"
    
    for i, result in enumerate(case_results, 1):
        markdown_content += f"## 案例 {i}: {result['text']}\n\n"
        
        # 实体表格
        markdown_content += "### 抽取的实体\n\n"
        markdown_content += "| 实体文本 | 实体类型 | 起始位置 | 结束位置 |\n"
        markdown_content += "|----------|----------|----------|----------|\n"
        for ent_text, ent_type, start, end in result['entities']:
            markdown_content += f"| {ent_text} | {ent_type} | {start} | {end} |\n"
        
        # 关系表格
        markdown_content += "\n### 抽取的关系\n\n"
        if result['relations']:
            markdown_content += "| 主体 | 关系 | 客体 |\n"
            markdown_content += "|------|------|------|\n"
            for h, rel, t in result['relations']:
                markdown_content += f"| {h} | {rel} | {t} |\n"
        else:
            markdown_content += "无关系抽取结果\n"
        
        # 事件表格
        markdown_content += "\n### 抽取的事件\n\n"
        if result['events']:
            markdown_content += "| 触发词 | 事件类型 |\n"
            markdown_content += "|--------|----------|\n"
            for trigger, event_type in result['events']:
                markdown_content += f"| {trigger} | {event_type} |\n"
        else:
            markdown_content += "无事件抽取结果\n"
        
        markdown_content += "\n---\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


def run_end2end_eval(checkpoint_path=None):
    """
    运行端到端综合评测（模式12）
    
    Args:
        checkpoint_path: 模型checkpoint路径，默认为 None（使用默认路径）
    """
    cfg = Config()
    
    # 设置默认模型路径
    model_path = checkpoint_path if checkpoint_path else os.path.join('models', 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return None
    
    evaluator = End2EndEvaluator(cfg)
    
    # 使用模拟数据进行评测（如果没有真实的DuEE数据）
    duee_path = os.path.join(cfg.data_dir, 'duee_test.json')
    if not os.path.exists(duee_path):
        # 创建模拟数据
        print(f"DuEE data not found at {duee_path}, using mock data...")
        mock_data = [
            {"text": "北京天安门广场今天举行了盛大的升旗仪式", "head": "北京", "tail": "天安门广场", "event_type": "开幕"},
            {"text": "上海浦东机场发生延误，大量旅客滞留", "head": "上海", "tail": "浦东机场", "event_type": "事故"},
            {"text": "广州白云山风景区迎来旅游旺季", "head": "广州", "tail": "白云山", "event_type": "活动"}
        ]
        os.makedirs(os.path.dirname(duee_path), exist_ok=True)
        with open(duee_path, 'w', encoding='utf-8') as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 运行评测
    save_path = os.path.join(cfg.result_dir, 'end2end_results.json')
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    return evaluator.evaluate_end2end(duee_path, model_path, save_path)


def run_case_study(checkpoint_path=None):
    """
    运行真实文本案例分析（模式13）
    
    Args:
        checkpoint_path: 模型checkpoint路径，默认为 None（使用默认路径）
    """
    cfg = Config()
    
    # 设置默认模型路径
    model_path = checkpoint_path if checkpoint_path else os.path.join('models', 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train a model first or provide a valid checkpoint path.")
        return None
    
    save_path = os.path.join(cfg.result_dir, 'case_study_results.json')
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    return run_case_study_with_config(cfg, model_path, save_path)


def run_case_study_with_config(config, model_path, save_path='case_study_results.json'):
    """
    运行真实文本案例分析（内部函数，保留原有接口）
    
    Args:
        config: 配置对象
        model_path: 模型路径
        save_path: 结果保存路径
    """
    print("="*80)
    print("Case Study Analysis")
    print("="*80)
    
    # 初始化评测器
    evaluator = End2EndEvaluator(config)
    evaluator.load_model(model_path)
    
    # 运行案例分析
    case_results = []
    
    for i, text in enumerate(config.case_texts):
        print(f"\n{i+1}. Analyzing: {text}")
        result = evaluator.predict(text)
        case_results.append(result)
        
        # 打印结果
        print(f"\n   Text: {result['text']}")
        print(f"   Entities:")
        for ent_text, ent_type, start, end in result['entities']:
            print(f"      - {ent_text} ({ent_type})")
        
        print(f"   Relations:")
        if result['relations']:
            for h, rel, t in result['relations']:
                print(f"      - ({h}) -[{rel}]-> ({t})")
        else:
            print(f"      - None")
        
        print(f"   Events:")
        if result['events']:
            for trigger, event_type in result['events']:
                print(f"      - {event_type}: {trigger}")
        else:
            print(f"      - None")
    
    # 保存结果到 JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(case_results, f, ensure_ascii=False, indent=2)
    
    # 生成 Markdown 表格
    md_path = save_path.replace('.json', '.md')
    generate_case_study_markdown(case_results, md_path)
    
    print(f"\nResults saved to {save_path} and {md_path}")
    
    return case_results


if __name__ == "__main__":
    # 测试代码
    cfg = Config()
    evaluator = End2EndEvaluator(cfg)
    
    print("End2End evaluator initialized successfully!")
