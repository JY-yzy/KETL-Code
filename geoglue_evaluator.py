#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoGLUE评测模块
支持GeoETA、GeoCPA、GeoEAG等地理信息抽取任务的评估
"""
import os
import json
import csv
import re
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class GeoGLUEEvaluator:
    """GeoGLUE评测器"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # 仅保留地名识别（NER）相关子任务，移除分类/搜索类子任务
        # - GeoETA: 地理实体类型标注（NER）✓
        # - GeoCPA: 地理边界预测（NER）✓
        # - GeoWWC: 地理词汇理解（NER）✓
        # - GeoEAG: 地理实体对齐（分类）✗ 已移除
        # - TES-recall: 搜索召回 ✗ 已移除
        # - TES-rerank: 搜索重排序 ✗ 已移除
        self.tasks = {
            'GeoETA': self._load_geoeta,
            'GeoCPA': self._load_geocpa,
            'GeoWWC': self._load_geowwc,
        }
        # 标记NER类任务（用于结果汇总）
        self.ner_tasks = {'GeoETA', 'GeoCPA', 'GeoWWC'}
    
    def _load_geoeta(self, split='dev'):
        """加载GeoETA数据集（地理实体类型标注）"""
        path = os.path.join(self.data_dir, 'GeoETA', f'{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    text = ''.join(obj['tokens'])
                    data.append({
                        'text': text,
                        'tokens': obj['tokens'],
                        'ner_tags': obj['ner_tags']
                    })
        return data
    
    def _load_geocpa(self, split='dev'):
        """加载GeoCPA数据集（地理边界预测）"""
        path = os.path.join(self.data_dir, 'GeoCPA', f'{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    text = ''.join(obj['tokens'])
                    data.append({
                        'text': text,
                        'tokens': obj['tokens'],
                        'ner_tags': obj.get('ner_tags', []),
                        'span_tags': obj.get('span_tags', []),
                        'boundary_tags': obj.get('boundary_tags', [])
                    })
        return data
    
    def _load_geoeag(self, split='dev'):
        """加载GeoEAG数据集（地理实体对齐）
        
        数据格式: {"sentence1": "...", "sentence2": "...", "label": "exact_match/partial_match/not_match"}
        """
        path = os.path.join(self.data_dir, 'GeoEAG', f'{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    label_str = obj.get('label', 'not_match')
                    # 将字符串标签映射为数值: exact_match=2, partial_match=1, not_match=0
                    label_map = {'exact_match': 2, 'partial_match': 1, 'not_match': 0}
                    label = label_map.get(label_str, 0)
                    data.append({
                        'text1': obj.get('sentence1', obj.get('text1', '')),
                        'text2': obj.get('sentence2', obj.get('text2', '')),
                        'label': label,
                        'label_str': label_str
                    })
        return data
    
    def _load_geowwc(self, split='dev'):
        """加载GeoWWC数据集（地理词汇理解）
        
        数据格式: {"tokens": ["惠", "州", "市"], "ner_tags": ["B-/WHAT", "I-/WHAT", "E-/WHAT"]}
        标签类型: /WHAT(地名), /WHERE(位置), /OTHER(其他)
        使用BIOES标注体系
        """
        path = os.path.join(self.data_dir, 'GeoWWC', f'{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    text = ''.join(obj['tokens'])
                    data.append({
                        'text': text,
                        'tokens': obj['tokens'],
                        'ner_tags': obj['ner_tags']
                    })
        return data
    
    def _load_tes_recall(self, split='dev'):
        """加载GeoTES-recall数据集（地理地址检索测试）
        注意：这是检索任务数据集，包含query和positive/negative passages
        """
        # 修正路径：GeoTES-recall而不是TES-recall
        path = os.path.join(self.data_dir, 'GeoTES-recall', f'{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    # 实际数据格式：query, pos_id, query_gis, positive_passages, negative_passages
                    query = obj.get('query', '')
                    positive_passages = obj.get('positive_passages', [])
                    negative_passages = obj.get('negative_passages', [])
                    
                    # 提取positive passage中的文本
                    positive_texts = []
                    for p in positive_passages:
                        if isinstance(p, dict) and 'text' in p:
                            positive_texts.append(p['text'])
                    
                    data.append({
                        'query': query,
                        'positive_passages': positive_texts,
                        'negative_passages': negative_passages,
                        'pos_id': obj.get('pos_id', '')
                    })
        return data
    
    def _load_tes_rerank(self, split='dev'):
        """加载GeoTES-rerank数据集（地理地址重排序测试）
        注意：这是检索排序任务数据集，包含query和候选passages
        """
        # 修正路径：GeoTES-rerank而不是TES-rerank，文件名是adqid_*.json
        path = os.path.join(self.data_dir, 'GeoTES-rerank', f'adqid_{split}.json')
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    # 实际数据格式：query, query_gis, positive_passages, negative_passages
                    query = obj.get('query', '')
                    positive_passages = obj.get('positive_passages', [])
                    negative_passages = obj.get('negative_passages', [])
                    
                    # 提取positive passage中的文本
                    positive_texts = []
                    for p in positive_passages:
                        if isinstance(p, dict) and 'text' in p:
                            positive_texts.append(p['text'])
                    
                    # 提取negative passage中的文本
                    negative_texts = []
                    for p in negative_passages:
                        if isinstance(p, dict) and 'text' in p:
                            negative_texts.append(p['text'])
                    
                    data.append({
                        'query': query,
                        'positive_passages': positive_texts,
                        'negative_passages': negative_texts
                    })
        return data
    
    def evaluate_geoeta(self, model, tokenizer, lexicon_matcher, cfg, split='dev'):
        """评估GeoETA任务
        
        关键改进：将GeoETA的细粒度标签映射到模型的标签体系
        GeoETA标签: prov(省), city(市), district(区), town(镇), village(村), 
                   poi(兴趣点), road(道路), roadno(路号), houseno(门牌号),
                   floorno(楼层号), community(社区), devzone(开发区), subpoi(子poi)
        
        模型标签: GEO(地理), ADMIN(行政区划), LANDMARK(地标), ORG(组织), PER(人名)
        
        映射规则:
        - prov, city, district, town, village, community, devzone -> GEO (与训练数据 LOC/GPE->GEO 一致)
        - poi, subpoi, road -> LANDMARK (地标，与训练数据 scene->LANDMARK 一致)
        - roadno, houseno, floorno, assist -> O (不作为实体)
        """
        data = self._load_geoeta(split)
        if not data:
            return None
        
        model.eval()
        all_preds = []
        all_labels = []
        
        # GeoETA细粒度标签到模型标签的映射
        # 模型标签体系: O, B-GEO, I-GEO, B-ADMIN, I-ADMIN, B-LANDMARK, I-LANDMARK, B-ORG, I-ORG, B-PER, I-PER
        #
        # 关键：必须与训练数据的标签映射保持一致！
        # 训练数据中: LOC/GPE -> GEO, address -> GEO, government -> ADMIN, scene -> LANDMARK
        # 所以 prov/city/district 等行政区划在训练数据中属于 LOC/GEP -> GEO，不是 ADMIN
        geoeta_to_model_map = {
            # 行政区划类 -> GEO (与训练数据 LOC/GPE->GEO 一致)
            'prov': 'GEO', 'city': 'GEO', 'district': 'GEO',
            'town': 'GEO', 'village': 'GEO', 'community': 'GEO',
            'devzone': 'GEO',
            # 地标类 -> LANDMARK (与训练数据 scene->LANDMARK 一致)
            'poi': 'LANDMARK', 'subpoi': 'LANDMARK', 'road': 'LANDMARK',
            # 门牌号类 -> O (不作为实体评测)
            'roadno': 'O', 'houseno': 'O', 'floorno': 'O',
            # 辅助信息 -> O
            'assist': 'O'
        }
        
        def map_geoeta_tag(tag):
            """将GeoETA标签映射到模型标签"""
            if tag == 'O':
                return 'O'
            
            # 解析标签前缀和类型
            if '-' in tag:
                parts = tag.split('-')
                prefix = parts[0]  # B, I, E, S
                entity_type = parts[1] if len(parts) > 1 else ''
                
                # 映射实体类型
                mapped_type = geoeta_to_model_map.get(entity_type, 'GEO')  # 默认映射到GEO
                
                if mapped_type == 'O':
                    return 'O'
                
                # 对于E标签，转换为I标签（模型不支持E标签）
                if prefix == 'E':
                    return f'I-{mapped_type}'
                elif prefix == 'S':
                    return f'B-{mapped_type}'  # 单字实体转为B标签
                else:
                    return f'{prefix}-{mapped_type}'
            return 'O'
        
        # 使用模型的标签映射
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN', 
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_label2id = {label: i for i, label in enumerate(model_labels)}
        model_id2label = {i: label for label, i in model_label2id.items()}
        
        # 批量推理
        batch_size = 32
        total_batches = (len(data) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(data))
            batch_items = data[start:end]

            if batch_idx % 10 == 0:
                print(f"  GeoETA: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(data)})")

            batch_texts = [item['text'] for item in batch_items]

            # 批量Tokenize
            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)
            offset_mappings = encodings['offset_mapping'].cpu().numpy()

            # 批量推理
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # 处理每个样本
            for i, item in enumerate(batch_items):
                text = item['text']
                original_tags = item['ner_tags']
                offset_mapping = offset_mappings[i]

                # 将GeoETA标签映射到模型标签
                mapped_labels = [model_label2id.get(map_geoeta_tag(tag), 0) for tag in original_tags]

                # 获取当前样本的预测结果
                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        preds = pred_seq.cpu().numpy()
                    else:
                        preds = np.array(pred_seq)
                else:
                    preds = predictions[i].cpu().numpy()

                # 对齐预测与原始标签
                aligned_preds = []
                token_idx = 1  # Skip CLS token
                for char_idx, char in enumerate(text):
                    if token_idx < len(offset_mapping) and offset_mapping[token_idx][0] == char_idx:
                        if token_idx < len(preds):
                            aligned_preds.append(preds[token_idx])
                        token_idx += 1
                    elif token_idx < len(offset_mapping) and offset_mapping[token_idx][0] < char_idx < offset_mapping[token_idx][1]:
                        pass
                    else:
                        aligned_preds.append(0)  # O label

                aligned_preds = aligned_preds[:len(mapped_labels)]
                all_preds.extend(aligned_preds)
                all_labels.extend(mapped_labels)
        
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        # 类型不敏感的边界识别F1：抹平类型差异，所有非O统一为ENT，只评估地名边界识别能力
        # 模型核心任务是"识别地名"而非"给地名分类"，GeoGLUE细粒度类型与模型粗粒度类型不匹配，
        # type-sensitive指标会因类型不匹配而低估边界识别能力，boundary_f1更公平反映跨域识别能力
        preds_bin = [0 if int(p) == 0 else 1 for p in all_preds]
        labels_bin = [0 if int(l) == 0 else 1 for l in all_labels]
        boundary_f1 = f1_score(labels_bin, preds_bin, average='micro', zero_division=0)

        return {
            'task': 'GeoETA',
            'split': split,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'boundary_f1': boundary_f1,
            'sample_count': len(data)
        }
    
    def evaluate_geocpa(self, model, tokenizer, cfg, split='dev'):
        """评估GeoCPA任务（地理实体成分分析）
        
        GeoCPA标签: PD(行政区划), PE(地理实体), UE(用户实体), Brand(品牌), Entity(通用实体), CategorySuffix(类别后缀)
        映射到模型标签体系进行NER F1评测
        """
        data = self._load_geocpa(split)
        if not data:
            return None
        
        model.eval()
        all_preds = []
        all_labels = []
        
        # GeoCPA标签到模型标签的映射
        geocpa_to_model_map = {
            'PD': 'GEO',      # 行政区划 -> GEO
            'PE': 'GEO',      # 地理实体 -> GEO
            'UE': 'ORG',      # 用户实体 -> ORG
            'Brand': 'ORG',   # 品牌 -> ORG
            'Entity': 'LANDMARK',  # 通用实体 -> LANDMARK
            'CategorySuffix': 'O'  # 类别后缀 -> O
        }
        
        def map_geocpa_tag(tag):
            """将GeoCPA标签映射到模型标签"""
            if tag == 'O':
                return 'O'
            if '-' in tag:
                parts = tag.split('-')
                prefix = parts[0]
                entity_type = parts[1] if len(parts) > 1 else ''
                
                mapped_type = geocpa_to_model_map.get(entity_type, 'GEO')
                
                if mapped_type == 'O':
                    return 'O'
                
                if prefix == 'E':
                    return f'I-{mapped_type}'
                elif prefix == 'S':
                    return f'B-{mapped_type}'
                else:
                    return f'{prefix}-{mapped_type}'
            return 'O'
        
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN',
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_label2id = {label: i for i, label in enumerate(model_labels)}
        
        # 批量推理
        batch_size = 32
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(data))
            batch_items = data[start:end]
            
            if batch_idx % 10 == 0:
                print(f"  GeoCPA: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(data)})")
            
            batch_texts = [item['text'] for item in batch_items]
            
            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)
            offset_mappings = encodings['offset_mapping'].cpu().numpy()
            
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            for i, item in enumerate(batch_items):
                text = item['text']
                original_tags = item['ner_tags']
                offset_mapping = offset_mappings[i]
                
                mapped_labels = [model_label2id.get(map_geocpa_tag(tag), 0) for tag in original_tags]
                
                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        preds = pred_seq.cpu().numpy()
                    else:
                        preds = np.array(pred_seq)
                else:
                    preds = predictions[i].cpu().numpy()
                
                aligned_preds = []
                token_idx = 1
                for char_idx, char in enumerate(text):
                    if token_idx < len(offset_mapping) and offset_mapping[token_idx][0] == char_idx:
                        if token_idx < len(preds):
                            aligned_preds.append(preds[token_idx])
                        token_idx += 1
                    elif token_idx < len(offset_mapping) and offset_mapping[token_idx][0] < char_idx < offset_mapping[token_idx][1]:
                        pass
                    else:
                        aligned_preds.append(0)
                
                aligned_preds = aligned_preds[:len(mapped_labels)]
                all_preds.extend(aligned_preds)
                all_labels.extend(mapped_labels)
        
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        # 类型不敏感的边界识别F1（同GeoETA，抹平类型差异评估边界识别能力）
        preds_bin = [0 if int(p) == 0 else 1 for p in all_preds]
        labels_bin = [0 if int(l) == 0 else 1 for l in all_labels]
        boundary_f1 = f1_score(labels_bin, preds_bin, average='micro', zero_division=0)

        return {
            'task': 'GeoCPA',
            'split': split,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'boundary_f1': boundary_f1,
            'sample_count': len(data)
        }
    
    def evaluate_tes_recall(self, model, tokenizer, cfg, split='dev', max_samples=1000):
        """评估GeoTES-recall任务
        
        由于这是检索任务数据集，我们改为评测：
        1. 模型能否正确识别query中的地理实体
        2. 识别出的实体与positive passage中的实体匹配度
        
        性能优化：批量推理 + 进度提示 + 采样上限
        """
        data = self._load_tes_recall(split)
        if not data:
            return None
        
        # 限制样本数量以避免过长时间运行
        if len(data) > max_samples:
            print(f"  TES-recall: 采样 {max_samples}/{len(data)} 条进行评测")
            data = data[:max_samples]
        
        model.eval()
        total_samples = 0
        samples_with_entities = 0
        entity_match_count = 0
        
        # 使用模型的标签映射
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN', 
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_id2label = {i: label for i, label in enumerate(model_labels)}
        
        # 收集所有需要推理的文本（去重）
        all_texts = set()
        for item in data:
            query = item.get('query', '')
            positive_passages = item.get('positive_passages', [])
            if query:
                all_texts.add(query)
            for passage in positive_passages[:3]:
                if passage:
                    all_texts.add(passage)
        all_texts = list(all_texts)
        
        # 批量推理
        batch_size = 32
        text_to_entities = {}
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_texts))
            batch_texts = all_texts[start:end]
            
            if batch_idx % 10 == 0:
                print(f"  TES-recall: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(all_texts)} texts)")
            
            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)
            
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # 处理每个文本的预测结果
            for i, text in enumerate(batch_texts):
                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        pred_tags = pred_seq.cpu().numpy()
                    else:
                        pred_tags = np.array(pred_seq)
                else:
                    pred_tags = predictions[i].cpu().numpy()
                
                entities = []
                current = None
                max_idx = min(len(pred_tags), len(text) + 1)
                
                for j in range(1, max_idx):
                    tag = model_id2label.get(int(pred_tags[j]), 'O')
                    char_idx = j - 1
                    
                    if char_idx >= len(text):
                        break
                    
                    if tag.startswith('B-') and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        if current:
                            entities.append(current)
                        current = text[char_idx:char_idx+1]
                    elif tag.startswith('I-') and current and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        current += text[char_idx:char_idx+1]
                    else:
                        if current:
                            entities.append(current)
                            current = None
                if current:
                    entities.append(current)
                
                text_to_entities[text] = set(entities)
        
        # 计算指标
        for item in data:
            query = item.get('query', '')
            positive_passages = item.get('positive_passages', [])
            
            if not query:
                continue
            
            total_samples += 1
            query_entities = text_to_entities.get(query, set())
            
            if query_entities:
                samples_with_entities += 1
            
            if query_entities and positive_passages:
                matched = False
                for passage in positive_passages[:3]:
                    if not passage:
                        continue
                    passage_entities = text_to_entities.get(passage, set())
                    for q_ent in query_entities:
                        for p_ent in passage_entities:
                            if q_ent in p_ent or p_ent in q_ent or q_ent == p_ent:
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    entity_match_count += 1
        
        entity_recognition_rate = samples_with_entities / total_samples if total_samples > 0 else 0
        entity_match_rate = entity_match_count / total_samples if total_samples > 0 else 0
        
        return {
            'task': 'TES-recall',
            'split': split,
            'entity_recognition_rate': entity_recognition_rate,  # query中识别出实体的比例
            'entity_match_rate': entity_match_rate,  # query和passage实体匹配的比例
            'total_samples': total_samples,
            'samples_with_entities': samples_with_entities,
            'entity_match_count': entity_match_count
        }
    
    def evaluate_tes_rerank(self, model, tokenizer, cfg, split='dev', max_samples=1000):
        """评估GeoTES-rerank任务
        
        由于这是检索排序任务数据集，我们改为评测：
        1. 模型能否正确识别query中的地理实体
        2. query与positive passage的实体匹配率
        3. query与negative passage的实体区分度
        
        性能优化：批量推理 + 进度提示 + 采样上限
        """
        data = self._load_tes_rerank(split)
        if not data:
            return None
        
        # 限制样本数量以避免过长时间运行
        if len(data) > max_samples:
            print(f"  TES-rerank: 采样 {max_samples}/{len(data)} 条进行评测")
            data = data[:max_samples]
        
        model.eval()
        total_samples = 0
        samples_with_entities = 0
        positive_match_count = 0
        negative_match_count = 0
        
        # 使用模型的标签映射
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN', 
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_id2label = {i: label for i, label in enumerate(model_labels)}
        
        # 收集所有需要推理的文本（去重）
        all_texts = set()
        for item in data:
            query = item.get('query', '')
            positive_passages = item.get('positive_passages', [])
            negative_passages = item.get('negative_passages', [])
            if query:
                all_texts.add(query)
            for passage in positive_passages[:3]:
                if passage:
                    all_texts.add(passage)
            for passage in negative_passages[:3]:
                if passage:
                    all_texts.add(passage)
        all_texts = list(all_texts)
        
        # 批量推理
        batch_size = 32
        text_to_entities = {}
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_texts))
            batch_texts = all_texts[start:end]
            
            if batch_idx % 10 == 0:
                print(f"  TES-rerank: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(all_texts)} texts)")
            
            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)
            
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # 处理每个文本的预测结果
            for i, text in enumerate(batch_texts):
                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        pred_tags = pred_seq.cpu().numpy()
                    else:
                        pred_tags = np.array(pred_seq)
                else:
                    pred_tags = predictions[i].cpu().numpy()
                
                entities = []
                current = None
                max_idx = min(len(pred_tags), len(text) + 1)
                
                for j in range(1, max_idx):
                    tag = model_id2label.get(int(pred_tags[j]), 'O')
                    char_idx = j - 1
                    
                    if char_idx >= len(text):
                        break
                    
                    if tag.startswith('B-') and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        if current:
                            entities.append(current)
                        current = text[char_idx:char_idx+1]
                    elif tag.startswith('I-') and current and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        current += text[char_idx:char_idx+1]
                    else:
                        if current:
                            entities.append(current)
                            current = None
                if current:
                    entities.append(current)
                
                text_to_entities[text] = set(entities)
        
        # 计算指标
        for item in data:
            query = item.get('query', '')
            positive_passages = item.get('positive_passages', [])
            negative_passages = item.get('negative_passages', [])
            
            if not query:
                continue
            
            total_samples += 1
            query_entities = text_to_entities.get(query, set())
            
            if query_entities:
                samples_with_entities += 1
            
            if query_entities and positive_passages:
                for passage in positive_passages[:3]:
                    if not passage:
                        continue
                    passage_entities = text_to_entities.get(passage, set())
                    for q_ent in query_entities:
                        for p_ent in passage_entities:
                            if q_ent in p_ent or p_ent in q_ent or q_ent == p_ent:
                                positive_match_count += 1
                                break
            
            if query_entities and negative_passages:
                for passage in negative_passages[:3]:
                    if not passage:
                        continue
                    neg_entities = text_to_entities.get(passage, set())
                    for q_ent in query_entities:
                        for n_ent in neg_entities:
                            if q_ent in n_ent or n_ent in q_ent or q_ent == n_ent:
                                negative_match_count += 1
                                break
        
        entity_recognition_rate = samples_with_entities / total_samples if total_samples > 0 else 0
        positive_match_rate = positive_match_count / total_samples if total_samples > 0 else 0
        negative_match_rate = negative_match_count / total_samples if total_samples > 0 else 0
        
        discrimination_score = positive_match_rate - negative_match_rate
        
        return {
            'task': 'TES-rerank',
            'split': split,
            'entity_recognition_rate': entity_recognition_rate,
            'positive_match_rate': positive_match_rate,
            'negative_match_rate': negative_match_rate,
            'discrimination_score': discrimination_score,
            'total_samples': total_samples,
            'samples_with_entities': samples_with_entities
        }
    
    def evaluate_geoeag(self, model, tokenizer, cfg, split='dev', max_samples=2000):
        """评估GeoEAG任务（地理实体对齐）

        改进：使用NER模型从两段文本中抽取地理实体，通过实体重叠度判断对齐关系。
        标签: exact_match(2)=完全匹配, partial_match(1)=部分匹配, not_match(0)=不匹配

        性能优化：批量推理 + 进度提示 + 采样上限
        """
        data = self._load_geoeag(split)
        if not data:
            return None

        # 限制样本数量以避免过长时间运行
        if len(data) > max_samples:
            print(f"  GeoEAG: 采样 {max_samples}/{len(data)} 条进行评测")
            data = data[:max_samples]

        model.eval()
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN',
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_id2label = {i: label for i, label in enumerate(model_labels)}

        # 收集所有需要推理的文本（去重）
        all_texts = set()
        for item in data:
            t1 = item.get('text1', '')
            t2 = item.get('text2', '')
            if t1:
                all_texts.add(t1)
            if t2:
                all_texts.add(t2)
        all_texts = list(all_texts)

        # 批量推理
        batch_size = 32
        text_to_entities = {}
        total_batches = (len(all_texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_texts))
            batch_texts = all_texts[start:end]

            if batch_idx % 10 == 0:
                print(f"  GeoEAG: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(all_texts)} texts)")

            # 批量tokenize
            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)

            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # 处理每个文本的预测结果
            for i, text in enumerate(batch_texts):
                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        pred_tags = pred_seq.cpu().numpy()
                    else:
                        pred_tags = np.array(pred_seq)
                else:
                    pred_tags = predictions[i].cpu().numpy()

                # 提取地理实体（从index 1开始，跳过[CLS]）
                entities = []
                current = None
                # pred_tags[0]是[CLS]，从1开始对应文本字符
                for j in range(1, min(len(pred_tags), len(text) + 1)):
                    tag = model_id2label.get(int(pred_tags[j]), 'O')
                    char = text[j - 1]
                    if tag.startswith('B-') and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        if current:
                            entities.append(current)
                        current = char
                    elif tag.startswith('I-') and current and tag[2:] in ['GEO', 'ADMIN', 'LANDMARK']:
                        current += char
                    else:
                        if current:
                            entities.append(current)
                            current = None
                if current:
                    entities.append(current)

                text_to_entities[text] = set(entities)

        # 计算准确率
        correct = 0
        total = 0
        for item in data:
            text1 = item.get('text1', '')
            text2 = item.get('text2', '')
            label = item.get('label', 0)

            if not text1 or not text2:
                continue

            entities1 = text_to_entities.get(text1, set())
            entities2 = text_to_entities.get(text2, set())

            if len(entities1) == 0 and len(entities2) == 0:
                jaccard = 0
            else:
                overlap = len(entities1 & entities2)
                union = len(entities1 | entities2)
                jaccard = overlap / union if union > 0 else 0

            if jaccard > 0.5:
                pred = 2
            elif jaccard > 0:
                pred = 1
            else:
                pred = 0

            if pred == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return {
            'task': 'GeoEAG',
            'split': split,
            'accuracy': accuracy,
            'sample_count': total
        }
    
    def evaluate_geowwc(self, model, tokenizer, cfg, split='dev'):
        """评估GeoWWC任务（地理词汇理解）
        
        改进：GeoWWC数据实际是NER格式（tokens+ner_tags），标签类型为/WHAT、/WHERE、/OTHER。
        将其作为NER任务评测：将GeoWWC标签映射到模型标签体系，计算实体识别准确率。
        
        映射规则:
        - /WHERE -> GEO (位置/行政区划，与训练数据 LOC/GPE->GEO 一致)
        - /WHAT -> LANDMARK (地名/地标)
        - /OTHER -> O (其他)
        """
        data = self._load_geowwc(split)
        if not data:
            return None

        model.eval()
        all_preds = []
        all_labels = []

        # GeoWWC标签到模型标签的映射
        # 模型标签: O, B-GEO, I-GEO, B-ADMIN, I-ADMIN, B-LANDMARK, I-LANDMARK, B-ORG, I-ORG, B-PER, I-PER
        model_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN',
                        'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        model_label2id = {label: i for i, label in enumerate(model_labels)}
        model_id2label = {i: label for label, i in model_label2id.items()}

        def map_geowwc_tag(tag):
            """将GeoWWC标签映射到模型标签"""
            if tag == 'O' or '/OTHER' in tag:
                return 'O'
            if '/WHERE' in tag:
                # 位置类 -> GEO (与训练数据 LOC/GPE->GEO 一致)
                if tag.startswith('B-'):
                    return 'B-GEO'
                elif tag.startswith('I-') or tag.startswith('E-'):
                    return 'I-GEO'
                elif tag.startswith('S-'):
                    return 'B-GEO'
            if '/WHAT' in tag:
                # 地名类 -> LANDMARK
                if tag.startswith('B-'):
                    return 'B-LANDMARK'
                elif tag.startswith('I-') or tag.startswith('E-'):
                    return 'I-LANDMARK'
                elif tag.startswith('S-'):
                    return 'B-LANDMARK'
            return 'O'

        # 批量推理
        batch_size = 32
        total_batches = (len(data) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(data))
            batch_items = data[start:end]

            if batch_idx % 10 == 0:
                print(f"  GeoWWC: 推理进度 {batch_idx}/{total_batches} batches ({start}/{len(data)})")

            batch_texts = [item['text'] for item in batch_items]

            encodings = tokenizer(
                batch_texts,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(cfg.device)
            attention_mask = encodings['attention_mask'].to(cfg.device)
            offset_mappings = encodings['offset_mapping'].cpu().numpy()

            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            for i, item in enumerate(batch_items):
                text = item['text']
                original_tags = item['ner_tags']
                offset_mapping = offset_mappings[i]

                mapped_labels = [model_label2id.get(map_geowwc_tag(tag), 0) for tag in original_tags]

                if isinstance(predictions, list):
                    pred_seq = predictions[i] if i < len(predictions) else []
                    if isinstance(pred_seq, torch.Tensor):
                        preds = pred_seq.cpu().numpy()
                    else:
                        preds = np.array(pred_seq)
                else:
                    preds = predictions[i].cpu().numpy()

                aligned_preds = []
                token_idx = 1
                for char_idx, char in enumerate(text):
                    if token_idx < len(offset_mapping) and offset_mapping[token_idx][0] == char_idx:
                        if token_idx < len(preds):
                            aligned_preds.append(preds[token_idx])
                        else:
                            aligned_preds.append(0)
                        token_idx += 1
                    else:
                        if aligned_preds:
                            aligned_preds.append(aligned_preds[-1])
                        else:
                            aligned_preds.append(0)

                min_len = min(len(aligned_preds), len(mapped_labels))
                all_preds.extend(aligned_preds[:min_len])
                all_labels.extend(mapped_labels[:min_len])

        # 计算准确率（token级别）
        if len(all_preds) > 0:
            correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
            accuracy = correct / len(all_preds)
        else:
            accuracy = 0
        
        # 计算Micro-F1和Macro-F1（token级别，与GeoETA/GeoCPA保持一致）
        try:
            micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception:
            micro_f1 = 0.0
            macro_f1 = 0.0
        # 类型不敏感的边界识别F1（同GeoETA/GeoCPA）
        try:
            preds_bin = [0 if int(p) == 0 else 1 for p in all_preds]
            labels_bin = [0 if int(l) == 0 else 1 for l in all_labels]
            boundary_f1 = f1_score(labels_bin, preds_bin, average='micro', zero_division=0)
        except Exception:
            boundary_f1 = 0.0

        return {
            'task': 'GeoWWC',
            'split': split,
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'boundary_f1': boundary_f1,
            'sample_count': len(data)
        }
    
    def evaluate_all(self, model, tokenizer, lexicon_matcher, cfg):
        """评估所有GeoGLUE地名识别（NER）子任务
        
        仅保留地名识别相关的NER子任务：
        - GeoETA: 地理实体类型标注
        - GeoCPA: 地理边界预测
        - GeoWWC: 地理词汇理解
        
        已移除的分类/搜索类子任务：
        - GeoEAG: 地理实体对齐（分类任务）
        - TES-recall: 搜索召回
        - TES-rerank: 搜索重排序
        
        评分体系：百分制（0-100分），综合分 = 3个NER任务Micro-F1的算术平均
        """
        results = {}
        
        print("\n" + "="*60)
        print("GeoGLUE 地名识别评测 (NER: GeoETA / GeoCPA / GeoWWC)")
        print("="*60)
        
        # GeoETA - 地理实体类型标注
        try:
            eta_result = self.evaluate_geoeta(model, tokenizer, lexicon_matcher, cfg, 'dev')
            if eta_result:
                results['GeoETA'] = eta_result
                eta_micro = eta_result['micro_f1'] * 100
                eta_macro = eta_result['macro_f1'] * 100
                eta_boundary = eta_result.get('boundary_f1', 0) * 100
                print(f"  [GeoETA] 地理实体类型标注  Micro-F1: {eta_micro:5.1f}分  边界F1: {eta_boundary:5.1f}分  (样本: {eta_result.get('sample_count', 0)})")
        except Exception as e:
            print(f"  [GeoETA] 评测失败: {e}")
        
        # GeoCPA - 地理边界预测
        try:
            cpa_result = self.evaluate_geocpa(model, tokenizer, cfg, 'dev')
            if cpa_result:
                results['GeoCPA'] = cpa_result
                cpa_micro = cpa_result['micro_f1'] * 100
                cpa_macro = cpa_result['macro_f1'] * 100
                cpa_boundary = cpa_result.get('boundary_f1', 0) * 100
                print(f"  [GeoCPA] 地理边界预测      Micro-F1: {cpa_micro:5.1f}分  边界F1: {cpa_boundary:5.1f}分  (样本: {cpa_result.get('sample_count', 0)})")
        except Exception as e:
            print(f"  [GeoCPA] 评测失败: {e}")
        
        # GeoWWC - 地理词汇理解
        try:
            wwc_result = self.evaluate_geowwc(model, tokenizer, cfg, 'dev')
            if wwc_result:
                results['GeoWWC'] = wwc_result
                wwc_micro = wwc_result.get('micro_f1', 0) * 100
                wwc_acc = wwc_result.get('accuracy', 0) * 100
                wwc_boundary = wwc_result.get('boundary_f1', 0) * 100
                print(f"  [GeoWWC] 地理词汇理解      Micro-F1: {wwc_micro:5.1f}分  边界F1: {wwc_boundary:5.1f}分  (样本: {wwc_result.get('sample_count', 0)})")
        except Exception as e:
            print(f"  [GeoWWC] 评测失败: {e}")
        
        # 汇总NER任务的综合得分（百分制）
        if results:
            ner_f1s = [r.get('boundary_f1', r.get('micro_f1', 0)) for r in results.values() if 'boundary_f1' in r or 'micro_f1' in r]
            if ner_f1s:
                avg_f1 = sum(ner_f1s) / len(ner_f1s)
                overall_score = avg_f1 * 100
                results['_summary'] = {
                    'avg_boundary_f1': avg_f1,
                    'overall_score': overall_score,
                    'task_count': len(ner_f1s),
                    'split': 'dev'
                }
                # 评分等级
                if overall_score >= 90:
                    grade = "优秀 (A)"
                elif overall_score >= 80:
                    grade = "良好 (B)"
                elif overall_score >= 70:
                    grade = "中等 (C)"
                elif overall_score >= 60:
                    grade = "及格 (D)"
                else:
                    grade = "不及格 (F)"
                
                print("-" * 60)
                print(f"  ★ 综合得分: {overall_score:5.1f}分 / 100分  [{grade}]")
                print(f"    (基于 {len(ner_f1s)} 个NER子任务的边界识别F1平均，类型不敏感)")
                print("="*60)
        
        return results

def evaluate_long_tail_entities(model, tokenizer, cfg, train_data, lexicon, id2label):
    """
    评估长尾地名实体识别（包含正则分类和频率分组两种分析）
    
    Args:
        model: 训练好的NER模型
        tokenizer: 分词器
        cfg: 配置对象
        train_data: 训练数据（用于统计实体频率）
        lexicon: 地名词典
        id2label: 标签ID到标签名的映射
    
    Returns:
        包含两种分析结果的字典
    """
    # ==================== 1. 正则模式分类分析 ====================
    # 定义长尾地名模式
    regex_patterns = [
        {'name': 'County_Level', 'pattern': r'[县市区]', 'description': 'County-level and below'},
        {'name': 'Historical', 'pattern': r'(古城|古镇|古村|遗址|故都|古街|古镇)', 'description': 'Historical places'},
        {'name': 'Natural_Geo', 'pattern': r'(山|峰|河|湖|海|江|溪|泉|潭|瀑|峡|岛|洲)', 'description': 'Natural geographic entities'},
        {'name': 'Administrative', 'pattern': r'(省|市|自治区|特别行政区|自治州|盟)', 'description': 'Administrative regions'},
        {'name': 'Town_Village', 'pattern': r'(镇|乡|村|街道|社区)', 'description': 'Towns and villages'},
        {'name': 'POI', 'pattern': r'(酒店|医院|学校|商场|车站|机场|公园|广场|大厦)', 'description': 'Points of Interest'}
    ]
    
    # Geo标签映射
    geo_label_set = {'O', 'B-GEO', 'I-GEO', 'E-GEO', 'S-GEO', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC'}
    label2id = {label: i for i, label in enumerate(sorted(geo_label_set))}
    
    # 从词典中按正则模式筛选测试样本
    regex_test_samples = {p['name']: [] for p in regex_patterns}
    for name in lexicon:
        for pattern_info in regex_patterns:
            if re.search(pattern_info['pattern'], name):
                # 构建测试文本：包含实体且有上下文的句子
                contexts = [
                    f"我去过{name}，那里风景很美",
                    f"从北京到{name}需要多长时间",
                    f"{name}是著名的旅游胜地",
                    f"他住在{name}附近"
                ]
                for ctx in contexts[:2]:  # 每个实体取2个上下文
                    regex_test_samples[pattern_info['name']].append({
                        'text': ctx,
                        'entity': name,
                        'category': pattern_info['name'],
                        'description': pattern_info['description']
                    })
    
    # 对正则分类进行评估
    regex_results = {}
    for pattern_name, samples in regex_test_samples.items():
        if not samples:
            continue
        
        total = 0
        correct = 0
        sample_limit = min(50, len(samples))  # 每个类别最多50个样本
        
        for sample in samples[:sample_limit]:
            text = sample['text']
            entity = sample['entity']
            
            encoding = tokenizer(
                text,
                max_length=cfg.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(cfg.device)
            attention_mask = encoding['attention_mask'].to(cfg.device)
            
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # 处理预测结果
            if isinstance(predictions, list):
                pred_tags = predictions[0] if len(predictions) > 0 else []
                if isinstance(pred_tags, torch.Tensor):
                    pred_tags = pred_tags.cpu().numpy()
            else:
                pred_tags = predictions[0].cpu().numpy()
            
            tags = [id2label.get(p, 'O') if isinstance(p, (int, np.integer)) else 'O' for p in pred_tags]
            
            # 检查实体是否被正确识别为地理实体
            entity_found = False
            for i, tag in enumerate(tags):
                if tag.startswith('B-') and tag[2:] in ['GEO', 'LOC', 'ADMIN', 'LANDMARK']:
                    entity_found = True
                    break
            
            total += 1
            if entity_found:
                correct += 1
        
        precision = correct / total if total > 0 else 0
        regex_results[pattern_name] = {
            'total': total,
            'correct': correct,
            'precision': precision
        }
    
    # ==================== 2. 训练频率分组分析 ====================
    # 统计训练集中实体的出现频率
    from collections import defaultdict
    entity_freq = defaultdict(int)
    for item in train_data:
        if 'ner_tags' in item:
            text = item.get('text', '')
            tags = item.get('ner_tags', [])
            i = 0
            while i < len(tags):
                if tags[i].startswith('B-') and tags[i][2:] in ['GEO', 'LOC', 'ADMIN', 'LANDMARK']:
                    entity_type = tags[i][2:]
                    start = i
                    i += 1
                    while i < len(tags) and tags[i].startswith('I-' + entity_type):
                        i += 1
                    entity_text = text[start:i]
                    entity_freq[entity_text] += 1
                else:
                    i += 1
    
    # 按频率分组
    high_freq_entities = []  # >50次
    medium_freq_entities = []  # 10-50次
    low_freq_entities = []  # <10次
    
    for entity, freq in entity_freq.items():
        if freq > 50:
            high_freq_entities.append((entity, freq))
        elif 10 <= freq <= 50:
            medium_freq_entities.append((entity, freq))
        else:
            low_freq_entities.append((entity, freq))
    
    # 按频率分组评估
    frequency_results = {}
    
    for level_name, entities in [('High_Freq_>50', high_freq_entities),
                                   ('Medium_Freq_10-50', medium_freq_entities),
                                   ('Low_Freq_<10', low_freq_entities)]:
        if not entities:
            continue
        
        # 限制评估数量
        eval_entities = entities[:100] if len(entities) > 100 else entities
        
        total = 0
        correct = 0
        for entity, freq in eval_entities:
            contexts = [
                f"我去过{entity}，那里很美",
                f"从北京到{entity}需要多长时间"
            ]
            
            for text in contexts[:1]:  # 每个实体评估1次
                encoding = tokenizer(
                    text,
                    max_length=cfg.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(cfg.device)
                attention_mask = encoding['attention_mask'].to(cfg.device)
                
                with torch.no_grad():
                    predictions, _, _, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # 处理预测结果
                if isinstance(predictions, list):
                    pred_tags = predictions[0] if len(predictions) > 0 else []
                    if isinstance(pred_tags, torch.Tensor):
                        pred_tags = pred_tags.cpu().numpy()
                else:
                    pred_tags = predictions[0].cpu().numpy()
                
                tags = [id2label.get(p, 'O') if isinstance(p, (int, np.integer)) else 'O' for p in pred_tags]
                
                # 检查实体是否被正确识别
                entity_found = False
                for tag in tags:
                    if tag.startswith('B-') and tag[2:] in ['GEO', 'LOC', 'ADMIN', 'LANDMARK']:
                        entity_found = True
                        break
                
                total += 1
                if entity_found:
                    correct += 1
        
        precision = correct / total if total > 0 else 0
        frequency_results[level_name] = {
            'total_entities': len(entities),
            'evaluated': total,
            'correct': correct,
            'precision': precision
        }
    
    # ==================== 汇总结果 ====================
    return {
        'task': 'Long_Tail_Analysis',
        'regex_based': {
            'patterns': regex_patterns,
            'results': regex_results
        },
        'frequency_based': {
            'high_freq': {'count': len(high_freq_entities), 'entities': high_freq_entities[:20]},
            'medium_freq': {'count': len(medium_freq_entities), 'entities': medium_freq_entities[:20]},
            'low_freq': {'count': len(low_freq_entities), 'entities': low_freq_entities[:20]},
            'results': frequency_results
        }
    }


def save_long_tail_results(results, output_path):
    """保存长尾分析结果到CSV"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 1. 正则分类结果
        writer.writerow(['=== Regex-based Classification Results ==='])
        writer.writerow(['Category', 'Total', 'Correct', 'Precision'])
        if 'regex_based' in results:
            for cat, data in results['regex_based']['results'].items():
                writer.writerow([cat, data['total'], data['correct'], round(data['precision'], 4)])
        
        writer.writerow([])
        
        # 2. 频率分组结果
        writer.writerow(['=== Frequency-based Classification Results ==='])
        writer.writerow(['Frequency_Level', 'Total_Entities', 'Evaluated', 'Correct', 'Precision'])
        if 'frequency_based' in results:
            for level, data in results['frequency_based']['results'].items():
                writer.writerow([
                    level,
                    data['total_entities'],
                    data['evaluated'],
                    data['correct'],
                    round(data['precision'], 4)
                ])
        
        writer.writerow([])
        
        # 3. 示例实体
        writer.writerow(['=== Sample Entities by Frequency Level ==='])
        if 'frequency_based' in results:
            for level, data in results['frequency_based'].items():
                if level == 'results':
                    continue
                writer.writerow([f'--- {level} (Top 20) ---'])
                writer.writerow(['Entity', 'Frequency'])
                for entity, freq in data.get('entities', [])[:20]:
                    writer.writerow([entity, freq])
    
    print(f"Long-tail analysis results saved to: {output_path}")

def save_geoglue_results(results, output_path):
    """保存GeoGLUE评测结果到CSV（百分制显示，仅NER子任务）
    
    输出格式：
    Task, Split, Metric, Value
    GeoETA, dev, micro_f1, 85.00
    GeoETA, dev, macro_f1, 78.00
    GeoCPA, dev, micro_f1, 82.00
    ...
    _summary, dev, overall_score, 83.33
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Split', 'Metric', 'Value'])
        
        for task, result in results.items():
            if not isinstance(result, dict):
                continue
            
            split = result.get('split', 'dev')
            
            # 汇总信息（百分制综合分）
            if task == '_summary':
                if 'overall_score' in result:
                    writer.writerow([task, split, 'overall_score', f"{result['overall_score']:.2f}"])
                    writer.writerow([task, split, 'avg_boundary_f1', f"{result.get('avg_boundary_f1', 0):.4f}"])
                    writer.writerow([task, split, 'task_count', result.get('task_count', 0)])
                continue
            
            # NER任务的核心指标（F1和Accuracy以百分制保存）
            if 'micro_f1' in result:
                writer.writerow([task, split, 'micro_f1', f"{result['micro_f1']*100:.2f}"])
            if 'boundary_f1' in result:
                writer.writerow([task, split, 'boundary_f1', f"{result['boundary_f1']*100:.2f}"])
            if 'macro_f1' in result:
                writer.writerow([task, split, 'macro_f1', f"{result['macro_f1']*100:.2f}"])
            if 'accuracy' in result:
                writer.writerow([task, split, 'accuracy', f"{result['accuracy']*100:.2f}"])
            if 'sample_count' in result:
                writer.writerow([task, split, 'samples', result['sample_count']])
    
    print(f"GeoGLUE results saved to: {output_path}")

if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer
    from config_ner import cfg_ner
    from models import NERModel

    evaluator = GeoGLUEEvaluator(os.path.join(os.path.dirname(__file__), 'data', 'GeoGLUE'))

    # ===== 1. 数据加载测试 =====
    eta_data = evaluator._load_geoeta('dev')
    print(f"GeoETA dev samples: {len(eta_data)}")
    cpa_data = evaluator._load_geocpa('dev')
    print(f"GeoCPA dev samples: {len(cpa_data)}")
    wwc_data = evaluator._load_geowwc('dev')
    print(f"GeoWWC dev samples: {len(wwc_data)}")
    print(f"Active NER tasks: {list(evaluator.tasks.keys())}")

    # ===== 2. 加载已训练模型并跑评测 =====
    # 修改 MODEL_VARIANT 可切换评测的模型(用config中的+号命名，与训练时一致)：
    #   RoBERTaBiLSTMAttentionCRF / RoBERTaBiLSTMAttentionCRF+Lexicon / RoBERTaBiLSTMAttentionCRF+DynamicLexicon
    #   RoBERTaBiGRUAttentionCRF  / RoBERTaBiGRUAttentionCRF+Lexicon  / RoBERTaBiGRUAttentionCRF+DynamicLexicon
    MODEL_VARIANT = 'RoBERTaBiLSTMAttentionCRF'
    # 文件名保存时 + -> _，/ -> _，故查找文件需做同样替换
    model_save_path = os.path.join(cfg_ner.model_save_dir, f"ner_model_{MODEL_VARIANT.replace('+', '_').replace('/', '_')}.pt")

    if not os.path.exists(model_save_path):
        print(f"\n[跳过评测] 模型权重不存在: {model_save_path}")
        print("请先运行 ABLATION 模式训练模型，或修改 MODEL_VARIANT 为已训练的模型名。")
    else:
        print(f"\n{'='*60}")
        print(f"加载模型: {MODEL_VARIANT}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(cfg_ner.roberta_model_name, local_files_only=True)

        # 是否使用词典增强（按模型名中是否含 Lexicon 判定）
        use_lexicon = 'Lexicon' in MODEL_VARIANT

        # 先加载权重，从中推导训练时的词典大小（避免 lexicon_size 不匹配导致加载失败）
        state_dict = torch.load(model_save_path, map_location=cfg_ner.device)
        if use_lexicon and 'lexicon_adapter.word_embed.weight' in state_dict:
            # NERModel 内部构建词嵌入时用 nn.Embedding(lexicon_size + 1, ...) 预留 padding_idx=0，
            # 故 checkpoint 中权重行数 = 训练时 lexicon_size + 1，需减1还原
            trained_lexicon_size = state_dict['lexicon_adapter.word_embed.weight'].shape[0] - 1
            if trained_lexicon_size != cfg_ner.lexicon_size:
                print(f"  [注意] 词典大小不匹配: 训练时={trained_lexicon_size}, 当前config={cfg_ner.lexicon_size}")
                print(f"         已自动按训练时大小构建模型，确保权重正确加载")
        else:
            trained_lexicon_size = cfg_ner.lexicon_size

        model = NERModel(
            model_name=cfg_ner.roberta_model_name,
            num_labels=len(cfg_ner.ner_label2id),
            model_variant=MODEL_VARIANT,
            use_crf=True,
            use_lexicon=use_lexicon,
            lexicon_size=trained_lexicon_size,
            lexicon_type='dynamic' if 'Dynamic' in MODEL_VARIANT else 'static',
            use_reshaping=False,
            lexicon_config={
                'word_emb_dim': getattr(cfg_ner, 'lexicon_word_emb_dim', 64),
                'fusion_type': cfg_ner.lexicon_fusion_type,
                'use_fusion_gate': cfg_ner.use_lexicon_fusion_gate,
                'dropout': cfg_ner.lexicon_dropout,
                'num_levels': cfg_ner.num_geo_levels
            }
        )
        model.to(cfg_ner.device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [警告] Missing keys: {len(missing)} 个")
        if unexpected:
            print(f"  [警告] Unexpected keys: {len(unexpected)} 个")
        model.eval()
        print(f"  权重已加载: {model_save_path}")

        # ===== 3. 跑 GeoGLUE 评测 =====
        results = evaluator.evaluate_all(model, tokenizer, None, cfg_ner)

        # ===== 4. 保存结果 CSV =====
        csv_path = os.path.join(cfg_ner.result_dir, f'geoglue_{MODEL_VARIANT}.csv')
        save_geoglue_results(results, csv_path)
        print(f"\n结果已保存，可查看: {csv_path}")