#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下游任务价值验证：NER -> Geocoding 空间匹配率评估

流程：
  1) 加载 KETL (Ours-StaticLex) 与 baseline (RoBERTa-BiLSTM-Attention-CRF) 检查点
  2) 对 25 条网络文本做地名识别（BIO 解码 -> 实体+类型）
  3) Geocoding 流水线：精确匹配 -> 别名匹配 -> 去行政后缀 -> 编辑距离 -> 输出经纬度
  4) 指标：总体匹配率 / 按行政层级 / 按实体类型，KETL vs baseline 对比
  5) 未匹配案例分析

输出：
  output_ner/results/geocoding_details.csv   明细
  output_ner/results/geocoding_report.md     汇总表 + 未匹配分析 + 结论
"""
import os
import sys
import csv
import json
import difflib
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_ner import cfg_ner as cfg
from models import NERModel
from data_utils import create_lexicon_matcher
from geo_knowledge import GeoKnowledgeGraph
from transformers import AutoTokenizer

LEVEL_NAMES = ['Unknown', 'Country', 'Province', 'City', 'District', 'Town', 'Village']

# 常见行政区划/通名后缀（用于模糊匹配的去后缀归一化）
ADMIN_SUFFIXES = [
    '特别行政区', '维吾尔自治区', '壮族自治区', '回族自治区', '自治区',
    '自治州', '自治县', '国际机场', '经济开发区', '高新技术开发区',
    '省', '市', '区', '县', '州', '盟', '旗', '镇', '乡', '村', '街道',
    '地区', '新区', '古城', '古镇', '机场', '风景区', '景区',
]


# ========================== 1. 模型推理 ==========================

def build_model(model_cfg, ckpt_path):
    """按 benchmark 配置构建模型并加载检查点"""
    model = NERModel(
        model_name=model_cfg['model_name'],
        num_labels=len(cfg.ner_label2id),
        model_variant=model_cfg['model_variant'],
        use_crf=model_cfg['use_crf'],
        use_lexicon=model_cfg['use_lexicon'],
        lexicon_size=cfg.lexicon_size,
        lexicon_type=model_cfg.get('lexicon_type', 'static'),
        use_reshaping=False,
        ablation_config=model_cfg.get('ablation_config', None),
        lexicon_config={
            'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout,
            'num_levels': cfg.num_geo_levels,
        }
    )
    model.to(cfg.device)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=cfg.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [Load] {os.path.basename(ckpt_path)}  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"  [WARN] 检查点不存在: {ckpt_path}")
    model.eval()
    return model


def compute_lexicon_features(text, offset_mapping, matcher, max_len):
    """与 MultiTaskDataset._match_lexicon 完全一致的词典特征编码"""
    lexicon_indices = torch.zeros((max_len,), dtype=torch.long)
    lexicon_levels = torch.zeros((max_len,), dtype=torch.long)
    lexicon_types = torch.zeros((max_len,), dtype=torch.long)
    lexicon_coords = torch.zeros((max_len, 2), dtype=torch.float)

    matches = matcher.match_with_attributes(text)
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        for match_start, match_end, word, word_id, level_id, type_id, coords in matches:
            if start >= match_start and end <= match_end + 1:
                lexicon_indices[i] = word_id
                lexicon_levels[i] = level_id
                lexicon_types[i] = type_id
                if coords is not None:
                    lexicon_coords[i][0] = coords[0] / 90.0
                    lexicon_coords[i][1] = coords[1] / 180.0
                break
    return lexicon_indices, lexicon_levels, lexicon_types, lexicon_coords


def extract_entities_from_bio(tags, text, offset_mapping, id2label):
    """从 BIO 标签序列提取实体（带 offset 映射）"""
    entities = []
    current = None
    for i, tag_id in enumerate(tags):
        tag = id2label.get(int(tag_id), 'O')
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
    for e in entities:
        e['text'] = text[e['start']:e['end']]
    return [e for e in entities if e['text']]


def infer_entities(model, tokenizer, matcher, text, use_lexicon):
    """对单条文本推理，返回实体列表 [{text, type, start, end}]"""
    encoding = tokenizer(text, max_length=cfg.max_seq_length, padding='max_length',
                         truncation=True, return_offsets_mapping=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(cfg.device)
    attention_mask = encoding['attention_mask'].to(cfg.device)
    offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()

    kwargs = {}
    if use_lexicon and matcher is not None:
        li, ll, lt, lc = compute_lexicon_features(text, offset_mapping, matcher, cfg.max_seq_length)
        kwargs = {
            'lexicon_indices': li.unsqueeze(0).to(cfg.device),
            'lexicon_levels': ll.unsqueeze(0).to(cfg.device),
            'lexicon_types': lt.unsqueeze(0).to(cfg.device),
            'lexicon_coords': lc.unsqueeze(0).to(cfg.device),
        }

    with torch.no_grad():
        predictions, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    if isinstance(predictions, list):
        pred_tags = predictions[0] if predictions else []
        if isinstance(pred_tags, torch.Tensor):
            pred_tags = pred_tags.cpu().numpy()
    else:
        pred_tags = predictions[0].cpu().numpy()

    return extract_entities_from_bio(pred_tags, text, offset_mapping, cfg.ner_id2label)


# ========================== 2. Geocoding 流水线 ==========================

def build_zh_coord_map(cn_path, cache_path):
    """
    从 CN.txt (GeoNames 中国数据) 构建 中文名 -> (lat, lng) 坐标字典。
    GeoNames 每行第 4 列为别名列表（含中文名），第 5/6 列为纬度/经度。
    结果缓存为 JSON 以加速后续运行。
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [Geocoder] 中文坐标字典从缓存加载: {len(data)} 条")
        return data

    def _valid(name):
        if not name or len(name) > 20:
            return False
        if not any('一' <= c <= '鿿' for c in name):
            return False
        if any(c in name for c in '<>/\\|@#$%^&*()=+[]{};'):
            return False
        return True

    zh_map = {}
    with open(cn_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            names_str, lat, lon = parts[3], parts[4], parts[5]
            try:
                flat, flon = float(lat), float(lon)
            except ValueError:
                continue
            if not (-90 <= flat <= 90 and -180 <= flon <= 180):
                continue
            for name in names_str.split(','):
                name = name.strip()
                if _valid(name) and name not in zh_map:
                    zh_map[name] = [flat, flon]
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(zh_map, f, ensure_ascii=False)
    print(f"  [Geocoder] 中文坐标字典构建完成: {len(zh_map)} 条 (已缓存)")
    return zh_map


class Geocoder:
    """地名 -> 经纬度 地理编码器（精确 -> 别名 -> 去后缀 -> 编辑距离 -> 省级近似）"""

    def __init__(self, matcher, zh_coord_map):
        self.lexicon = matcher.lexicon            # set of names
        self.alias_map = matcher.alias_map        # alias -> standard_name
        self.hierarchy_map = matcher.hierarchy_map  # name -> level id (0-6)
        self.type_map = matcher.type_map          # name -> type id
        self.zh_coord_map = zh_coord_map          # 中文名 -> [lat, lng] (主坐标源)
        self.en_coord_map = matcher.coord_map     # GeoNames 主名(多为罗马化) -> (lat, lng)
        self.approx_coord_map = matcher.approx_coord_map  # 省级近似坐标

        # 编辑距离候选索引：首字 -> 中文坐标词典中的名称列表
        self.first_char_index = defaultdict(list)
        for name in self.zh_coord_map:
            if len(name) >= 2:
                self.first_char_index[name[0]].append(name)

    @staticmethod
    def normalize_coords(raw):
        """坐标顺序修正：数据中存在 [lng,lat] 与 [lat,lng] 混排，按取值范围判定"""
        if raw is None:
            return None
        try:
            a, b = float(raw[0]), float(raw[1])
        except (TypeError, ValueError, IndexError):
            return None
        # 中国范围：纬度 3~54，经度 73~136；先做合法性判断
        if abs(a) > 90 and abs(b) <= 90:      # a 必为经度
            lat, lng = b, a
        elif abs(a) <= 90 and abs(b) > 90:    # b 必为经度（>90 不可能是纬度）
            lat, lng = a, b
        elif abs(a) <= 90 and abs(b) <= 90:
            # 二者都在纬度合法域：中国经度必 >73，若 a>54 则更可能是经度
            if a > 54 and b <= 54:
                lat, lng = b, a
            else:
                lat, lng = a, b
        else:
            return None
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        return (round(lat, 6), round(lng, 6))

    def get_coords(self, name):
        """优先中文坐标字典（[lat, lng]），其次 GeoNames 主名坐标（(lat, lng) 元组）"""
        if name in self.zh_coord_map:
            return self.normalize_coords(self.zh_coord_map[name])
        if name in self.en_coord_map:
            return self.normalize_coords(self.en_coord_map[name])
        return None

    def get_approx_coords(self, name):
        """省级近似坐标（实体名包含省份名时回退到省会/省中心）"""
        for prov, coords in self.approx_coord_map.items():
            if prov in name:
                return (round(float(coords[0]), 6), round(float(coords[1]), 6))
        return None

    def get_level(self, name):
        """行政层级：优先词典层级表，未覆盖时按后缀启发式推断"""
        lv = self.hierarchy_map.get(name, 0)
        if lv:
            return lv
        return self.infer_level_by_suffix(name)

    @staticmethod
    def infer_level_by_suffix(name):
        """按地名后缀启发式推断行政层级（1国 2省 3市 4区县 5乡镇 6村）"""
        if not name:
            return 0
        if name.endswith(('特别行政区', '自治区', '省')):
            return 2
        if name.endswith(('自治州', '地区', '盟', '市')):
            return 3
        if name.endswith(('自治县', '市辖区', '区', '县', '旗')):
            return 4
        if name.endswith(('镇', '乡', '街道')):
            return 5
        if name.endswith('村'):
            return 6
        if name.endswith('国'):
            return 1
        return 0

    def strip_suffix(self, name):
        """去除行政区划/通名后缀，返回候选归一化名称列表（长后缀优先）"""
        candidates = []
        for suf in ADMIN_SUFFIXES:
            if name.endswith(suf) and len(name) > len(suf):
                candidates.append(name[:-len(suf)])
        return candidates

    def geocode(self, name):
        """
        返回 dict: {matched, method, standard_name, lat, lng, level}
        method: exact / alias / suffix / edit_distance / approx / none
        """
        # ---- a. 精确匹配（全名在坐标词典中命中）----
        coords = self.get_coords(name)
        if coords is not None:
            return {'matched': True, 'method': 'exact', 'standard_name': name,
                    'lat': coords[0], 'lng': coords[1], 'level': self.get_level(name)}

        # ---- b1. 别名匹配（别名 -> 标准名，要求标准名有坐标；过滤明显错误的别名污染）----
        std = self.alias_map.get(name)
        if std and std != name and len(std) <= 20:
            coords = self.get_coords(std)
            if coords is not None:
                return {'matched': True, 'method': 'alias', 'standard_name': std,
                        'lat': coords[0], 'lng': coords[1], 'level': self.get_level(std)}

        # ---- b2. 去行政后缀后精确/别名匹配 ----
        for cand in self.strip_suffix(name):
            coords = self.get_coords(cand)
            if coords is not None:
                return {'matched': True, 'method': 'suffix', 'standard_name': cand,
                        'lat': coords[0], 'lng': coords[1], 'level': self.get_level(cand)}
            std2 = self.alias_map.get(cand)
            if std2 and std2 != cand:
                coords = self.get_coords(std2)
                if coords is not None:
                    return {'matched': True, 'method': 'suffix', 'standard_name': std2,
                            'lat': coords[0], 'lng': coords[1], 'level': self.get_level(std2)}

        # ---- b3. 编辑距离匹配（同首字、长度差<=2，相似度阈值 0.8）----
        if len(name) >= 2:
            pool = self.first_char_index.get(name[0], [])
            pool = [w for w in pool if abs(len(w) - len(name)) <= 2]
            if pool:
                close = difflib.get_close_matches(name, pool, n=1, cutoff=0.8)
                if close:
                    best = close[0]
                    coords = self.get_coords(best)
                    if coords is not None:
                        return {'matched': True, 'method': 'edit_distance', 'standard_name': best,
                                'lat': coords[0], 'lng': coords[1], 'level': self.get_level(best)}

        # ---- c. 省级近似坐标兜底（粗粒度定位，单独标注）----
        approx = self.get_approx_coords(name)
        if approx is not None:
            return {'matched': True, 'method': 'approx', 'standard_name': name,
                    'lat': approx[0], 'lng': approx[1], 'level': self.get_level(name)}

        return {'matched': False, 'method': 'none', 'standard_name': name,
                'lat': None, 'lng': None, 'level': self.lookup_level(name),
                'fail_reason': 'oov' if name not in self.lexicon else 'hit_lexicon_no_coord'}

    def lookup_level(self, name):
        """仅为分组统计查询层级（精确->别名->去后缀），与是否 geocode 成功无关"""
        lv = self.get_level(name)
        if lv:
            return lv
        if name in self.alias_map:
            lv = self.get_level(self.alias_map[name])
            if lv:
                return lv
        for cand in self.strip_suffix(name):
            lv = self.get_level(cand)
            if lv:
                return lv
        return 0


# ========================== 3. 主流程 ==========================

def run_for_model(model_cfg, ckpt_name, cases, tokenizer, matcher, geocoder):
    print(f"\n{'='*60}\n[Infer] {model_cfg['name']}\n{'='*60}")
    model = build_model(model_cfg, os.path.join(cfg.model_save_dir, ckpt_name))
    use_lexicon = model_cfg['use_lexicon']

    records = []
    for item in cases:
        case_id, text = item['id'], item['text']
        entities = infer_entities(model, tokenizer, matcher, text, use_lexicon)
        for ent in entities:
            geo = geocoder.geocode(ent['text'])
            level = geo['level'] if geo['matched'] else geocoder.lookup_level(ent['text'])
            records.append({
                'model': model_cfg['name'],
                'case_id': case_id,
                'entity': ent['text'],
                'pred_type': ent['type'],
                'level': level,
                'level_name': LEVEL_NAMES[level] if 0 <= level < len(LEVEL_NAMES) else 'Unknown',
                'matched': geo['matched'],
                'method': geo['method'] if geo['matched'] else '',
                'lat': geo['lat'] if geo['matched'] else '',
                'lng': geo['lng'] if geo['matched'] else '',
                'standard_name': geo.get('standard_name', ent['text']),
                'fail_reason': '' if geo['matched'] else geo.get('fail_reason', ''),
            })
    print(f"  共识别 {len(records)} 个实体，匹配 {sum(1 for r in records if r['matched'])} 个")

    del model
    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()
    return records


def match_rate(records):
    return sum(1 for r in records if r['matched']) / len(records) if records else 0.0


def group_rates(records, key):
    groups = defaultdict(lambda: [0, 0])  # group -> [matched, total]
    for r in records:
        g = r[key]
        groups[g][1] += 1
        if r['matched']:
            groups[g][0] += 1
    return {g: (m, t, m / t) for g, (m, t) in sorted(groups.items(), key=lambda x: str(x[0]))}


def main():
    print("[Geocoding Eval] 加载地理知识图谱与词典匹配器...")
    load_ownthink = getattr(cfg, 'load_ownthink', False)
    kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink, skip_cache=False)
    matcher = create_lexicon_matcher(cfg, load_ownthink=load_ownthink, kg_instance=kg)
    zh_coord_map = build_zh_coord_map(
        os.path.join(cfg.data_dir, 'CN.txt'),
        os.path.join(cfg.data_dir, 'processed', 'zh_coord_map.json'),
    )
    geocoder = Geocoder(matcher, zh_coord_map)
    print(f"  词典: {len(matcher.lexicon)}, 中文坐标: {len(zh_coord_map)}, 别名: {len(matcher.alias_map)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.roberta_model_name, local_files_only=True)

    with open(cfg.network_text_cases_path, 'r', encoding='utf-8-sig') as f:
        cases = json.load(f)
    print(f"  测试语料: {len(cases)} 条网络文本")

    # KETL 与 baseline 配置（与 staticlex_benchmark 一致）
    # 注意：model_variant 必须是可解析的架构名（Ours-StaticLex / RoBERTaBiLSTMAttentionCRF），
    # 'name' 仅作展示用途
    ketl_cfg = {'name': 'KETL', 'model_variant': 'Ours-StaticLex',
                'model_name': cfg.roberta_model_name,
                'use_crf': True, 'use_lexicon': True, 'lexicon_type': 'static',
                'ablation_config': None}
    base_cfg = {'name': 'Baseline', 'model_variant': 'RoBERTaBiLSTMAttentionCRF',
                'model_name': cfg.roberta_model_name,
                'use_crf': True, 'use_lexicon': False, 'lexicon_type': 'static',
                'ablation_config': None}

    all_records = []
    all_records += run_for_model(ketl_cfg, 'ner_model_Ours-StaticLex.pt', cases, tokenizer, matcher, geocoder)
    all_records += run_for_model(base_cfg, 'ner_model_RoBERTaBiLSTMAttentionCRF.pt', cases, tokenizer, matcher, geocoder)

    # ========== 明细 CSV ==========
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'geocoding_details.csv')
    fieldnames = ['model', 'case_id', 'entity', 'pred_type', 'level', 'level_name',
                  'matched', 'method', 'lat', 'lng', 'standard_name', 'fail_reason']
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[Output] 明细 CSV: {csv_path}")

    # ========== 指标汇总 ==========
    ketl_recs = [r for r in all_records if r['model'].startswith('KETL')]
    base_recs = [r for r in all_records if r['model'].startswith('Baseline')]

    summary = {
        'ketl': {
            'overall': match_rate(ketl_recs),
            'strict': (sum(1 for r in ketl_recs if r['matched'] and r['method'] != 'approx') / len(ketl_recs)) if ketl_recs else 0.0,
            'total': len(ketl_recs), 'matched': sum(1 for r in ketl_recs if r['matched']),
            'by_level': group_rates(ketl_recs, 'level_name'),
            'by_type': group_rates(ketl_recs, 'pred_type'),
            'by_method': defaultdict(int),
        },
        'base': {
            'overall': match_rate(base_recs),
            'strict': (sum(1 for r in base_recs if r['matched'] and r['method'] != 'approx') / len(base_recs)) if base_recs else 0.0,
            'total': len(base_recs), 'matched': sum(1 for r in base_recs if r['matched']),
            'by_level': group_rates(base_recs, 'level_name'),
            'by_type': group_rates(base_recs, 'pred_type'),
            'by_method': defaultdict(int),
        },
    }
    for r in all_records:
        if r['matched']:
            key = 'ketl' if r['model'].startswith('KETL') else 'base'
            summary[key]['by_method'][r['method']] += 1
    summary['ketl']['by_method'] = dict(summary['ketl']['by_method'])
    summary['base']['by_method'] = dict(summary['base']['by_method'])

    # 未匹配案例
    unmatched_ketl = [r for r in ketl_recs if not r['matched']]
    unmatched_type_dist = defaultdict(int)
    for r in unmatched_ketl:
        unmatched_type_dist[r['pred_type']] += 1
    summary['ketl']['unmatched_type_dist'] = dict(unmatched_type_dist)
    summary['ketl']['unmatched_examples'] = unmatched_ketl[:15]

    json_path = os.path.join(cfg.result_dir, 'geocoding_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Output] 汇总 JSON: {json_path}")

    # ========== 控制台打印汇总 ==========
    print(f"\n{'='*70}")
    print(f"总体匹配率: KETL {summary['ketl']['matched']}/{summary['ketl']['total']} = {summary['ketl']['overall']*100:.1f}%"
          f" (严格 {summary['ketl']['strict']*100:.1f}%)  |  "
          f"Baseline {summary['base']['matched']}/{summary['base']['total']} = {summary['base']['overall']*100:.1f}%"
          f" (严格 {summary['base']['strict']*100:.1f}%)")
    print(f"\n按行政层级 (KETL):")
    for g, (m, t, rate) in summary['ketl']['by_level'].items():
        print(f"  {g:<10} {m}/{t} = {rate*100:.1f}%")
    print(f"\n按实体类型 (KETL):")
    for g, (m, t, rate) in summary['ketl']['by_type'].items():
        print(f"  {g:<10} {m}/{t} = {rate*100:.1f}%")
    print(f"\n匹配方式分布 (KETL): {summary['ketl']['by_method']}")
    print(f"未匹配类型分布 (KETL): {summary['ketl']['unmatched_type_dist']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
