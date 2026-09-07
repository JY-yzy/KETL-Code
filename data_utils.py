#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据工具模块
功能：
1. 标签映射和统一
2. 词典加载和匹配
3. Trie树实现
4. 数据集加载（NER、DuIE、DuEE）
5. DataLoader创建和MultiTaskDataset类
"""
import os
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from config import Config

cfg = Config()

# ========================== 标签映射 ==========================
DATASET_LABEL_MAP = {
    "cluener": {
        "address": "GEO",
        "location": "GEO",
        "company": "ORG",
        "person": "PER",
        "game": "ORG",
        "movie": "ORG",
        "book": "ORG",
        "name": "PER",
        "government": "ADMIN",
        "position": "PER",
        "scene": "LANDMARK",
        "organization": "ORG",
    },
    "msra": {
        "PER": "PER",
        "LOC": "GEO",
        "ORG": "ORG",
        "GPE": "GEO",
    },
    "weibo": {
        "PER": "PER",
        "LOC": "GEO",
        "ORG": "ORG",
        "GPE": "GEO",
    },
    "cmner": {
        "PER": "PER",
        "LOC": "GEO",
        "ORG": "ORG",
        "GPE": "GEO",
    }
}

SUPPORTED_TYPES = {'GEO', 'ADMIN', 'LANDMARK', 'ORG', 'PER'}

def unify_tag(tag, dataset_name=None):
    """统一标签格式，将未知标签映射到O"""
    if tag == 'O':
        return 'O'
    if '-' in tag:
        prefix, entity_type = tag.split('-', 1)
        if prefix in ['B', 'I']:
            base_type = entity_type.split('.')[0]
            
            if dataset_name and dataset_name in DATASET_LABEL_MAP:
                if base_type in DATASET_LABEL_MAP[dataset_name]:
                    mapped_type = DATASET_LABEL_MAP[dataset_name][base_type]
                    return f'{prefix}-{mapped_type}'
                if entity_type in DATASET_LABEL_MAP[dataset_name]:
                    mapped_type = DATASET_LABEL_MAP[dataset_name][entity_type]
                    return f'{prefix}-{mapped_type}'
            
            if base_type in SUPPORTED_TYPES:
                return f'{prefix}-{base_type}'
    return 'O'

def apply_label_mapping(data, dataset_name):
    """对数据应用标签映射"""
    for item in data:
        if 'ner_tags' in item:
            item['ner_tags'] = [unify_tag(tag, dataset_name) for tag in item['ner_tags']]
    return data

# ========================== 词典加载 ==========================

def load_lexicon(path):
    """加载词典"""
    lexicon = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    lexicon.add(word)
    print(f"已加载 {len(lexicon)} 个地名")
    return lexicon

def create_lexicon_matcher(cfg, load_ownthink=None, kg_instance=None):
    """
    统一创建 LexiconMatcher，确保所有入口使用相同的词汇来源
    避免不同创建路径导致 word_id 映射不一致
    
    Args:
        cfg: 配置对象
        load_ownthink: 是否加载ownthink数据
        kg_instance: 已有的 GeoKnowledgeGraph 实例，用于复用
    
    Returns:
        LexiconMatcher 实例
    """
    from geo_knowledge import GeoKnowledgeGraph
    
    if load_ownthink is None:
        load_ownthink = getattr(cfg, 'load_ownthink', False)
    
    if kg_instance is not None:
        print("[LexiconMatcher] Using provided GeoKnowledgeGraph instance")
        kg = kg_instance
    else:
        print(f"[LexiconMatcher] Creating new GeoKnowledgeGraph (load_ownthink={load_ownthink})")
        kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink, skip_cache=False)
    
    lexicon_matcher = LexiconMatcher(use_geo_knowledge=True, kg_instance=kg)
    
    # 同步 cfg.lexicon_size（词典已在 GeoKnowledgeGraph 内部按实验逻辑筛选）
    actual_size = len(lexicon_matcher.lexicon)
    if actual_size != cfg.lexicon_size:
        print(f"[Warning] Lexicon size mismatch: config={cfg.lexicon_size}, actual={actual_size}")
        cfg.lexicon_size = actual_size
    
    return lexicon_matcher

def build_lexicon_vocab(lexicon_path):
    """
    从文件路径构建词典词汇表
    
    Args:
        lexicon_path: 词典文件路径
    
    Returns:
        lexicon_set: 词典集合
    """
    if isinstance(lexicon_path, set):
        return lexicon_path
    return load_lexicon(lexicon_path)

# ========================== Trie树实现 ==========================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.word_id = None  # 词汇ID（整数）或其他数据（如元组）

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, word_id):
        """
        插入词汇到Trie
        
        Args:
            word: 词汇字符串
            word_id: 词汇ID（整数）或其他数据（如元组(alias, standard_name)）
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word
        node.word_id = word_id  # 保存词汇ID或其他数据
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

# ========================== 词典匹配器 ==========================

class LexiconMatcher:
    def __init__(self, lexicon_set=None, use_geo_knowledge=True, load_ownthink=False, kg_instance=None):
        """
        初始化词典匹配器
        
        Args:
            lexicon_set: 词典集合、词典文件路径，或 None
            use_geo_knowledge: 是否使用扩展的地理知识图谱
            load_ownthink: 是否加载ownthink_v2.csv（8GB，包含组织、人名等）
            kg_instance: 已有的 GeoKnowledgeGraph 实例，用于复用避免重复加载
        """
        self.trie = Trie()
        self.lexicon = set()
        self.word2id = {}
        self.hierarchy_map = {}
        self.type_map = {}
        self.alias_map = {}
        self.coord_map = {}
        self.approx_coord_map = {}
        
        if use_geo_knowledge:
            self._init_with_geo_knowledge(load_ownthink, kg_instance)
        elif lexicon_set is not None:
            if isinstance(lexicon_set, str) and os.path.exists(lexicon_set):
                lexicon_set = load_lexicon(lexicon_set)
            elif isinstance(lexicon_set, str):
                lexicon_set = None
            self.build(lexicon_set)
            self._load_geo_knowledge()
    
    def _init_with_geo_knowledge(self, load_ownthink=None, kg_instance=None):
        """使用扩展的地理知识图谱初始化"""
        from geo_knowledge import GeoKnowledgeGraph
        
        if load_ownthink is None:
            load_ownthink = getattr(cfg, 'load_ownthink', False)
        
        if kg_instance is not None:
            print("[LexiconMatcher] Using provided GeoKnowledgeGraph instance")
            kg = kg_instance
        else:
            print(f"[LexiconMatcher] Creating new GeoKnowledgeGraph (load_ownthink={load_ownthink})")
            kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink)
        
        self.lexicon = set(kg.lexicon)
        self.alias_map = kg.alias_map
        self.hierarchy_map = kg.hierarchy_map
        self.type_map = kg.type_map
        self.coord_map = kg.coord_map
        
        self.build(self.lexicon)
        
        # 构建别名Trie
        self._build_alias_trie()
        
        self.approx_coord_map = {
            '北京': (39.9042, 116.4074),
            '上海': (31.2304, 121.4737),
            '天津': (39.0842, 117.2008),
            '重庆': (29.4316, 106.9123),
            '河北': (38.0423, 114.5075),
            '山西': (37.8706, 112.5489),
            '辽宁': (41.8056, 123.4315),
            '吉林': (43.8256, 125.3245),
            '黑龙江': (45.8038, 126.5349),
            '江苏': (32.0603, 118.7969),
            '浙江': (30.2741, 120.1551),
            '安徽': (31.8206, 117.2272),
            '福建': (26.0745, 119.2965),
            '江西': (28.6826, 115.8579),
            '山东': (36.6512, 117.1202),
            '河南': (34.7466, 113.6253),
            '湖北': (30.5928, 114.3055),
            '湖南': (28.2280, 112.9388),
            '广东': (23.1291, 113.2644),
            '海南': (20.0440, 110.2247),
            '四川': (30.5728, 104.0668),
            '贵州': (26.6476, 106.6301),
            '云南': (24.8820, 102.8329),
            '陕西': (34.2619, 108.9463),
            '甘肃': (36.0611, 103.8343),
            '青海': (36.6171, 101.7782),
            '台湾': (25.0330, 121.5654),
            '内蒙古': (40.8426, 111.7510),
            '广西': (22.8170, 108.3665),
            '西藏': (29.6540, 91.1765),
            '宁夏': (38.4864, 106.2332),
            '新疆': (43.8256, 87.6168),
            '香港': (22.3193, 114.1694),
            '澳门': (22.1987, 113.5439),
            '北京省': (39.9042, 116.4074),
            '北京市': (39.9042, 116.4074),
            '上海市': (31.2304, 121.4737),
            '天津市': (39.0842, 117.2008),
            '重庆市': (29.4316, 106.9123),
            '河北省': (38.0423, 114.5075),
            '山西省': (37.8706, 112.5489),
            '辽宁省': (41.8056, 123.4315),
            '吉林省': (43.8256, 125.3245),
            '黑龙江省': (45.8038, 126.5349),
            '江苏省': (32.0603, 118.7969),
            '浙江省': (30.2741, 120.1551),
            '安徽省': (31.8206, 117.2272),
            '福建省': (26.0745, 119.2965),
            '江西省': (28.6826, 115.8579),
            '山东省': (36.6512, 117.1202),
            '河南省': (34.7466, 113.6253),
            '湖北省': (30.5928, 114.3055),
            '湖南省': (28.2280, 112.9388),
            '广东省': (23.1291, 113.2644),
            '海南省': (20.0440, 110.2247),
            '四川省': (30.5728, 104.0668),
            '贵州省': (26.6476, 106.6301),
            '云南省': (24.8820, 102.8329),
            '陕西省': (34.2619, 108.9463),
            '甘肃省': (36.0611, 103.8343),
            '青海省': (36.6171, 101.7782),
            '台湾省': (25.0330, 121.5654),
            '内蒙古自治区': (40.8426, 111.7510),
            '广西壮族自治区': (22.8170, 108.3665),
            '西藏自治区': (29.6540, 91.1765),
            '宁夏回族自治区': (38.4864, 106.2332),
            '新疆维吾尔自治区': (43.8256, 87.6168),
            '香港特别行政区': (22.3193, 114.1694),
            '澳门特别行政区': (22.1987, 113.5439),
        }
        
        print(f"[LexiconMatcher] 使用地理知识图谱初始化")
        print(f"  - 词典大小: {len(self.lexicon)}")
        print(f"  - 别名映射: {len(self.alias_map)}")
        print(f"  - 层级信息: {len(self.hierarchy_map)}")
        print(f"  - 类型信息: {len(self.type_map)}")
        print(f"  - 坐标信息: {len(self.coord_map)}")
    
    def _load_geo_knowledge(self):
        """加载地理层级、别名、类型、坐标文件"""
        # 加载层级文件
        hierarchy_path = os.path.join(cfg.data_dir, 'processed', 'geo_lexicon_hierarchy.txt')
        if os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            level = int(parts[1].strip())
                            self.hierarchy_map[name] = level
            print(f"已加载 {len(self.hierarchy_map)} 个地名的层级信息")
        else:
            print(f"警告：层级文件不存在: {hierarchy_path}")
        
        # 加载别名映射
        alias_path = os.path.join(cfg.data_dir, 'processed', 'geo_alias_map.json')
        if os.path.exists(alias_path):
            with open(alias_path, 'r', encoding='utf-8') as f:
                self.alias_map = json.load(f)
            print(f"已加载 {len(self.alias_map)} 个别名映射")
        else:
            print(f"警告：别名文件不存在: {alias_path}")
        
        # 加载类型映射
        type_path = os.path.join(cfg.data_dir, 'processed', 'geo_type_map.json')
        if os.path.exists(type_path):
            with open(type_path, 'r', encoding='utf-8') as f:
                self.type_map = json.load(f)
            print(f"已加载 {len(self.type_map)} 个类型映射")
        else:
            print(f"警告：类型文件不存在: {type_path}")
        
        # 加载坐标映射
        coord_path = os.path.join(cfg.data_dir, 'geo_coordinates.json')
        if os.path.exists(coord_path):
            with open(coord_path, 'r', encoding='utf-8') as f:
                self.coord_map = json.load(f)
            print(f"已加载 {len(self.coord_map)} 个坐标映射")
        else:
            self.coord_map = {}
            print(f"警告：坐标文件不存在: {coord_path}")
        
        # 初始化近似坐标字典（基于中国各省会城市中心坐标）
        # 格式: 省份简称 -> (纬度, 经度)
        self.approx_coord_map = {
            '北京': (39.9042, 116.4074),
            '上海': (31.2304, 121.4737),
            '天津': (39.0842, 117.2008),
            '重庆': (29.4316, 106.9123),
            '河北': (38.0423, 114.5075),
            '山西': (37.8706, 112.5489),
            '辽宁': (41.8056, 123.4315),
            '吉林': (43.8256, 125.3245),
            '黑龙江': (45.8038, 126.5349),
            '江苏': (32.0603, 118.7969),
            '浙江': (30.2741, 120.1551),
            '安徽': (31.8206, 117.2272),
            '福建': (26.0745, 119.2965),
            '江西': (28.6826, 115.8579),
            '山东': (36.6512, 117.1202),
            '河南': (34.7466, 113.6253),
            '湖北': (30.5928, 114.3055),
            '湖南': (28.2280, 112.9388),
            '广东': (23.1291, 113.2644),
            '海南': (20.0440, 110.2247),
            '四川': (30.5728, 104.0668),
            '贵州': (26.6476, 106.6301),
            '云南': (24.8820, 102.8329),
            '陕西': (34.2619, 108.9463),
            '甘肃': (36.0611, 103.8343),
            '青海': (36.6171, 101.7782),
            '台湾': (25.0330, 121.5654),
            '内蒙古': (40.8426, 111.7510),
            '广西': (22.8170, 108.3665),
            '西藏': (29.6540, 91.1765),
            '宁夏': (38.4864, 106.2332),
            '新疆': (43.8256, 87.6168),
            '香港': (22.3193, 114.1694),
            '澳门': (22.1987, 113.5439),
            '北京省': (39.9042, 116.4074),
            '北京市': (39.9042, 116.4074),
            '上海市': (31.2304, 121.4737),
            '天津市': (39.0842, 117.2008),
            '重庆市': (29.4316, 106.9123),
            '河北省': (38.0423, 114.5075),
            '山西省': (37.8706, 112.5489),
            '辽宁省': (41.8056, 123.4315),
            '吉林省': (43.8256, 125.3245),
            '黑龙江省': (45.8038, 126.5349),
            '江苏省': (32.0603, 118.7969),
            '浙江省': (30.2741, 120.1551),
            '安徽省': (31.8206, 117.2272),
            '福建省': (26.0745, 119.2965),
            '江西省': (28.6826, 115.8579),
            '山东省': (36.6512, 117.1202),
            '河南省': (34.7466, 113.6253),
            '湖北省': (30.5928, 114.3055),
            '湖南省': (28.2280, 112.9388),
            '广东省': (23.1291, 113.2644),
            '海南省': (20.0440, 110.2247),
            '四川省': (30.5728, 104.0668),
            '贵州省': (26.6476, 106.6301),
            '云南省': (24.8820, 102.8329),
            '陕西省': (34.2619, 108.9463),
            '甘肃省': (36.0611, 103.8343),
            '青海省': (36.6171, 101.7782),
            '台湾省': (25.0330, 121.5654),
            '内蒙古自治区': (40.8426, 111.7510),
            '广西壮族自治区': (22.8170, 108.3665),
            '西藏自治区': (29.6540, 91.1765),
            '宁夏回族自治区': (38.4864, 106.2332),
            '新疆维吾尔自治区': (43.8256, 87.6168),
            '香港特别行政区': (22.3193, 114.1694),
            '澳门特别行政区': (22.1987, 113.5439),
        }
        print(f"已加载 {len(self.approx_coord_map)} 个近似坐标")
    
    def build(self, lexicon_set):
        self.lexicon = set(lexicon_set)
        self.trie = Trie()
        self.word2id = {}
        
        word_list = sorted(self.lexicon)
        for idx, word in enumerate(word_list, start=1):
            self.word2id[word] = idx
            if word:
                self.trie.insert(word, idx)
        
        print(f"Trie树构建完成，共 {len(self.lexicon)} 个词汇（包含单字词）")
    
    def match(self, text, max_overlap=True):
        """词典匹配，返回(start, end, word, word_id)"""
        if not text:
            return []
        if max_overlap:
            return self._match_longest(text)
        else:
            return self._match_all(text)
    
    def _match_all(self, text):
        """匹配所有可能的词汇"""
        matched = []
        n = len(text)
        for i in range(n):
            node = self.trie.root
            j = i
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_end:
                    matched.append((i, j - 1, node.word, node.word_id))
        return matched
    
    def _match_longest(self, text):
        """匹配最长词汇（贪心算法）"""
        matched = []
        n = len(text)
        i = 0
        while i < n:
            node = self.trie.root
            j = i
            last_match = None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_end:
                    last_match = (i, j - 1, node.word, node.word_id)
            if last_match:
                matched.append(last_match)
                i = last_match[1] + 1
            else:
                i += 1
        return matched
    
    def match_with_type(self, text):
        """匹配并返回NER标签"""
        matches = self.match(text)
        ner_tags = ['O'] * len(text)
        for start, end, word in matches:
            ner_tags[start] = 'B-GEO'
            for i in range(start + 1, end + 1):
                ner_tags[i] = 'I-GEO'
        return ner_tags
    
    def _get_approx_coords(self, word):
        """
        根据词汇获取近似坐标（基于行政层级推断）
        
        Args:
            word: 地名词汇
        
        Returns:
            coordinates: (latitude, longitude) or None
        """
        # 首先检查精确坐标
        if word in self.coord_map:
            return self.coord_map[word]
        
        # 检查别名的精确坐标
        if word in self.alias_map and self.alias_map[word] in self.coord_map:
            return self.coord_map[self.alias_map[word]]
        
        # 使用近似坐标：查找词汇中包含的省份名称
        for province, coords in self.approx_coord_map.items():
            if province in word:
                return coords
        
        return None
    
    def match_with_attributes(self, text):
        """
        匹配并返回词汇的ID、层级ID、类型ID和坐标（支持近似坐标）
        
        优化策略：
        1. 先检查别名匹配（高优先级）- 使用别名Trie加速
        2. 再检查精确匹配（标准优先级）- 使用主词典Trie
        
        Args:
            text: 输入文本
        
        Returns:
            list of tuples: (start, end, word, word_id, level_id, type_id, coordinates)
            coordinates: (latitude, longitude) or None
        """
        results = []
        matched_positions = set()  # 记录已匹配的位置，避免重复
        
        # ========== 阶段1：别名匹配（高优先级）==========
        # 使用别名Trie进行高效匹配
        if hasattr(self, 'alias_trie') and self.alias_trie:
            alias_matches = self._match_alias_trie(text)
            for start, end, alias, standard_name in alias_matches:
                end_idx = end + 1
                
                # 检查是否已被其他匹配覆盖
                is_overlapping = False
                for pos in matched_positions:
                    if not (end_idx <= pos[0] or start >= pos[1]):
                        is_overlapping = True
                        break
                
                if not is_overlapping and standard_name in self.word2id:
                    word_id = self.word2id[standard_name]
                    level_id = self.hierarchy_map.get(standard_name, 0)
                    type_id = self.type_map.get(standard_name, 0)
                    coords = self._get_approx_coords(standard_name)
                    
                    results.append((start, end, standard_name, word_id, level_id, type_id, coords))
                    matched_positions.add((start, end_idx))
        
        # ========== 阶段2：精确匹配（标准优先级）==========
        exact_matches = self.match(text)
        for start, end, word, word_id in exact_matches:
            end_idx = end + 1
            
            # 检查是否已被别名匹配覆盖
            is_overlapping = False
            for pos in matched_positions:
                if not (end_idx <= pos[0] or start >= pos[1]):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                level_id = self.hierarchy_map.get(word, 0)
                type_id = self.type_map.get(word, 0)
                coords = self._get_approx_coords(word)
                
                # 如果是别名，使用标准名的属性
                if word in self.alias_map:
                    standard_name = self.alias_map[word]
                    if standard_name in self.hierarchy_map:
                        level_id = self.hierarchy_map[standard_name]
                    if standard_name in self.type_map:
                        type_id = self.type_map[standard_name]
                    if standard_name in self.coord_map:
                        coords = self.coord_map[standard_name]
                
                results.append((start, end, word, word_id, level_id, type_id, coords))
                matched_positions.add((start, end_idx))
        
        # 按起始位置排序
        results.sort(key=lambda x: x[0])
        
        return results
    
    def _build_alias_trie(self):
        """构建别名Trie，加速别名匹配"""
        self.alias_trie = Trie()
        # 只添加长度≥2的别名（避免单字过度匹配）
        filtered_aliases = [(alias, standard) for alias, standard in self.alias_map.items() if len(alias) >= 2]
        # 按长度降序添加，确保长别名优先匹配
        for alias, standard in sorted(filtered_aliases, key=lambda x: -len(x[0])):
            self.alias_trie.insert(alias, (alias, standard))
        print(f"[LexiconMatcher] 别名Trie构建完成，共 {len(filtered_aliases)} 个别名")
    
    def _match_alias_trie(self, text):
        """使用别名Trie进行匹配，返回(start, end, alias, standard_name)"""
        matched = []
        n = len(text)
        i = 0
        while i < n:
            node = self.alias_trie.root
            j = i
            last_match = None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_end and node.word_id is not None:
                    alias, standard_name = node.word_id
                    last_match = (i, j - 1, alias, standard_name)
            if last_match:
                matched.append(last_match)
                i = last_match[1] + 1
            else:
                i += 1
        return matched
    
    def __len__(self):
        return len(self.lexicon)
    
    def __contains__(self, word):
        return word in self.lexicon

# ========================== 实体频率统计函数 ==========================

def extract_entities_from_item(item):
    """从数据项中提取实体"""
    entities = []
    if 'text' in item and 'ner_tags' in item:
        text = item['text']
        tags = item['ner_tags']
        n = len(text)
        i = 0
        while i < n:
            if tags[i].startswith('B-'):
                entity_type = tags[i][2:]
                start = i
                i += 1
                while i < n and tags[i].startswith('I-' + entity_type):
                    i += 1
                entities.append({
                    'text': text[start:i],
                    'type': entity_type,
                    'start': start,
                    'end': i
                })
            else:
                i += 1
    return entities

def build_entity_frequency(data):
    """统计数据集中实体出现频率"""
    freq = defaultdict(int)
    for item in data:
        entities = extract_entities_from_item(item)
        for ent in entities:
            freq[ent['text']] += 1
    return freq

def get_frequency_level(freq_dict, entity_text):
    """获取实体频率等级"""
    count = freq_dict.get(entity_text, 0)
    if count > 50:
        return 'high'
    elif 10 <= count <= 50:
        return 'medium'
    else:
        return 'low'

def split_data_by_frequency(data, freq_dict):
    """按实体频率分层数据"""
    high_freq_data = []
    medium_freq_data = []
    low_freq_data = []
    
    for item in data:
        entities = extract_entities_from_item(item)
        if not entities:
            continue
        
        # 按主要实体的频率决定样本归属
        entity_freqs = [freq_dict.get(e['text'], 0) for e in entities]
        if entity_freqs:
            max_freq = max(entity_freqs)
            if max_freq > 50:
                high_freq_data.append(item)
            elif 10 <= max_freq <= 50:
                medium_freq_data.append(item)
            else:
                low_freq_data.append(item)
    
    return high_freq_data, medium_freq_data, low_freq_data


def compute_class_weights(train_data, label_type='ner'):
    """
    计算类别权重，用于处理类别不平衡
    
    Args:
        train_data: 训练数据
        label_type: 标签类型 ('ner', 'rel', 'event')
    
    Returns:
        class_weights: 类别权重列表
    """
    from collections import defaultdict
    
    label_counts = defaultdict(int)
    total_samples = 0
    
    if label_type == 'ner':
        for item in train_data:
            if 'ner_tags' in item:
                for tag in item['ner_tags']:
                    label_id = cfg.ner_label2id.get(tag, 0)
                    label_counts[label_id] += 1
                    total_samples += 1
        
        # 计算逆频率权重（带平滑和上限，避免极端值导致NaN）
        num_classes = len(cfg.ner_labels)
        class_weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            if count == 0:
                weight = 1.0
            else:
                # 平滑逆频率权重: total / (num_classes * (count + smoothing))
                # 并限制最大权重为10，避免极端值导致梯度爆炸
                smoothing = 0.1  # 平滑项，防止权重过大
                weight = total_samples / (num_classes * (count + smoothing * total_samples / num_classes))
                weight = min(weight, 10.0)  # 限制最大权重
            class_weights.append(weight)
    
    elif label_type == 'rel':
        # 关系抽取类别权重计算
        for item in train_data:
            if 'spo_list' in item:
                for spo in item['spo_list']:
                    relation = spo.get('relation', '无关系')
                    label_id = cfg.rel_label2id.get(relation, 0)
                    label_counts[label_id] += 1
                    total_samples += 1
        
        num_classes = len(cfg.rel_labels)
        class_weights = []
        smoothing = 0.1
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            if count == 0:
                weight = 1.0
            else:
                weight = total_samples / (num_classes * (count + smoothing * total_samples / num_classes))
                weight = min(weight, 10.0)
            class_weights.append(weight)
    
    elif label_type == 'event':
        # 事件抽取类别权重计算
        for item in train_data:
            if 'event_list' in item:
                for event in item['event_list']:
                    event_type = event.get('event_type', '无事件')
                    label_id = cfg.event_label2id.get(event_type, 0)
                    label_counts[label_id] += 1
                    total_samples += 1
        
        num_classes = len(cfg.event_labels)
        class_weights = []
        smoothing = 0.1
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            if count == 0:
                weight = 1.0
            else:
                weight = total_samples / (num_classes * (count + smoothing * total_samples / num_classes))
                weight = min(weight, 10.0)
            class_weights.append(weight)
    
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    
    return class_weights

# ========================== 采样和分割函数 ==========================

def sample_data(data, sample_ratio=1.0):
    """采样数据"""
    if sample_ratio >= 1.0:
        return data
    import random
    sample_size = max(10, int(len(data) * sample_ratio))
    return random.sample(data, min(sample_size, len(data)))

def split_val_set(train_data, val_ratio=0.1, min_samples=100, max_samples=1000):
    """分割验证集"""
    import random
    val_size = max(min_samples, min(int(len(train_data) * val_ratio), max_samples))
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    val_indices = set(indices[:val_size])
    return [train_data[i] for i in range(len(train_data)) if i not in val_indices], [train_data[i] for i in val_indices]

# ========================== 数据集加载函数 ==========================

def load_ner_datasets(sample_size=None, use_dev_as_test=False):
    """
    加载预处理后的NER数据集
    
    参数：
        sample_size: 限制合并后的总样本量（None表示加载全部）
        use_dev_as_test: 是否使用验证集作为测试集（用于测试集无标签的情况）
    
    返回：
        merged_train_data: 合并的训练集
        merged_dev_data: 合并的验证集
        test_datasets: 各个测试集的字典 {dataset_name: data}
    """
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    
    datasets = ['cluener', 'msra', 'weibo_ner', 'cmner']
    
    merged_train_data = []
    merged_dev_data = []
    test_datasets = {}
    
    # 保存各数据集的验证集（用于替换无标签的测试集）
    dev_datasets = {}
    
    for dataset_name in datasets:
        # 加载训练集
        train_path = os.path.join(processed_dir, dataset_name, 'train.json')
        if os.path.exists(train_path):
            train_data = load_json_lines(train_path)
        
        # 加载验证集
        dev_path = os.path.join(processed_dir, dataset_name, 'dev.json')
        if os.path.exists(dev_path):
            dev_data = load_json_lines(dev_path)
        
        # 加载测试集
        test_path = os.path.join(processed_dir, dataset_name, 'test.json')
        if os.path.exists(test_path):
            test_data = load_json_lines(test_path)
        
        # 检查并修复数据泄露
        train_texts = set(item['text'] for item in train_data)
        dev_texts = set(item['text'] for item in dev_data)
        test_texts = set(item['text'] for item in test_data)
        
        train_dev_overlap = train_texts.intersection(dev_texts)
        train_test_overlap = train_texts.intersection(test_texts)
        dev_test_overlap = dev_texts.intersection(test_texts)
        
        total_overlap = len(train_dev_overlap) + len(train_test_overlap) + len(dev_test_overlap)
        if total_overlap > 0:
            print(f"检测到 {dataset_name} 数据泄露: train-dev={len(train_dev_overlap)}, train-test={len(train_test_overlap)}, dev-test={len(dev_test_overlap)}")
            print(f"正在自动修复数据泄露...")
            
            train_data = [item for item in train_data if item['text'] not in dev_texts and item['text'] not in test_texts]
            dev_data = [item for item in dev_data if item['text'] not in train_texts and item['text'] not in test_texts]
            test_data = [item for item in test_data if item['text'] not in train_texts and item['text'] not in dev_texts]
            
            print(f"修复完成: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
        
        merged_train_data.extend(train_data)
        merged_dev_data.extend(dev_data)
        dev_datasets[dataset_name] = dev_data
        
        print(f"加载 {dataset_name} 训练集: {len(train_data)} 条")
        print(f"加载 {dataset_name} 验证集: {len(dev_data)} 条")
        
        # 检查测试数据是否有有效标签
        has_valid_labels = any(any(tag != 'O' for tag in item.get('ner_tags', [])) for item in test_data)
        
        if use_dev_as_test or not has_valid_labels:
            # 使用验证集替换测试集
            if dataset_name in dev_datasets:
                print(f"注意: {dataset_name} 测试集无有效标签，使用验证集代替")
                test_datasets[dataset_name] = dev_datasets[dataset_name]
            else:
                test_datasets[dataset_name] = test_data
        else:
            test_datasets[dataset_name] = test_data
        
        # 统计测试集标签分布
        if len(test_datasets[dataset_name]) > 0:
            all_tags = []
            for item in test_datasets[dataset_name]:
                all_tags.extend(item.get('ner_tags', []))
            o_ratio = sum(1 for t in all_tags if t == 'O') / len(all_tags) if all_tags else 0
            if o_ratio > 0.95:
                print(f"警告: {dataset_name} 测试集O标签比例高达 {o_ratio*100:.1f}%，可能影响评估结果")
        
        print(f"加载 {dataset_name} 测试集: {len(test_datasets[dataset_name])} 条")
    
    # 在合并后应用样本量限制
    if sample_size:
        merged_train_data = merged_train_data[:sample_size]
        merged_dev_data = merged_dev_data[:sample_size//2] if sample_size > 1 else merged_dev_data
        for key in test_datasets:
            test_datasets[key] = test_datasets[key][:sample_size//2] if sample_size > 1 else test_datasets[key]
    
    print(f"\n合并后(应用样本量限制): 训练集 {len(merged_train_data)} 条, 验证集 {len(merged_dev_data)} 条")
    print(f"测试集: {list(test_datasets.keys())}")
    
    return merged_train_data, merged_dev_data, test_datasets

def load_duie_dataset(split='train'):
    """加载预处理后的DuIE数据集"""
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    file_path = os.path.join(processed_dir, 'duie', f'{split}.json')
    
    if not os.path.exists(file_path):
        print(f"DuIE {split} 文件不存在: {file_path}")
        return []
    
    data = load_json_lines(file_path)
    print(f"加载 DuIE {split}: {len(data)} 条")
    return data

def load_duee_dataset(split='train'):
    """加载预处理后的DuEE数据集"""
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    file_path = os.path.join(processed_dir, 'duee', f'{split}.json')
    
    if not os.path.exists(file_path):
        print(f"DuEE {split} 文件不存在: {file_path}")
        return []
    
    data = load_json_lines(file_path)
    print(f"加载 DuEE {split}: {len(data)} 条")
    return data

def load_json_lines(file_path):
    """加载JSON Lines格式文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ========================== 类别权重计算 ==========================

def calculate_class_weights(dataset, label2id, method='inverse_freq', min_weight=0.1, max_weight=10.0):
    """
    根据训练数据自动计算类别权重，提升Macro-F1
    
    Args:
        dataset: 训练数据集
        label2id: 标签到ID的映射
        method: 权重计算方法
            - 'inverse_freq': 逆频率权重 w = total / (num_classes * count)
            - 'sqrt_inverse': 平方根逆频率 w = sqrt(total / (num_classes * count))
            - 'manual': 使用配置中的手动权重
        min_weight: 最小权重限制
        max_weight: 最大权重限制
    
    Returns:
        class_weights: 类别权重列表
    """
    if method == 'manual':
        return None
    
    class_counts = defaultdict(int)
    total_labels = 0
    
    for item in dataset.data:
        ner_tags = item.get('ner_tags', [])
        for tag in ner_tags:
            if tag in label2id:
                class_counts[label2id[tag]] += 1
                total_labels += 1
    
    num_classes = len(label2id)
    class_weights = []
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        if method == 'inverse_freq':
            weight = total_labels / (num_classes * count)
        elif method == 'sqrt_inverse':
            weight = np.sqrt(total_labels / (num_classes * count))
        else:
            weight = 1.0
        
        weight = max(min_weight, min(max_weight, weight))
        class_weights.append(weight)
    
    print(f"[类别权重] 计算完成: method={method}, counts={dict(class_counts)}")
    print(f"[类别权重] 权重值: {class_weights}")
    
    return class_weights

# ========================== DataLoader 创建 ==========================

def create_dataloader(dataset, batch_size, shuffle=True, collate_fn=None):
    """
    创建DataLoader
    Args:
        dataset: Dataset对象
        batch_size: 批次大小
        shuffle: 是否打乱顺序
        collate_fn: 自定义的collate函数
    Returns:
        DataLoader对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn if collate_fn else default_collate_fn,
        num_workers=cfg.num_workers
    )

def collate_fn(batch):
    """
    collate_fn 别名，保持与 train.py 的兼容性
    """
    return default_collate_fn(batch)


def default_collate_fn(batch):
    """默认的collate函数"""
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.array(values))
        else:
            result[key] = values
    
    return result

# ========================== MultiTaskDataset 类 ==========================

class MultiTaskDataset(Dataset):
    """
    多任务数据集类（预编码优化版）
    支持：NER、关系抽取、事件抽取
    
    优化策略：
    1. 在 __init__ 中一次性完成所有文本的 tokenize 和词典匹配
    2. __getitem__ 只做索引取值，避免重复计算
    """
    def __init__(self, data, tokenizer, max_seq_length=128, lexicon_matcher=None, is_train=True):
        """
        Args:
            data: 预处理后的数据列表
            tokenizer: 分词器
            max_seq_length: 最大序列长度
            lexicon_matcher: 词典匹配器（用于词汇增强）
            is_train: 是否为训练模式
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.lexicon_matcher = lexicon_matcher
        self.is_train = is_train
        
        # 加载词汇映射
        self.lexicon_vocab = self._load_lexicon_vocab()
        
        # 预编码所有数据
        print(f"  - Pre-encoding {len(data)} samples...")
        self.encoded_data = self._pre_encode_all()
        print(f"  - Pre-encoding completed")
    
    def _load_lexicon_vocab(self):
        """加载词汇到ID的映射"""
        vocab_path = os.path.join(cfg.data_dir, 'processed', 'lexicon_vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
                vocab['id_to_word'] = {int(k): v for k, v in vocab['id_to_word'].items()}
                return vocab
        return {'word_to_id': {}, 'id_to_word': {}, 'size': 1}
    
    def _pre_encode_all(self):
        """预编码所有数据（一次性完成tokenize和词典匹配）"""
        texts = [item['text'] for item in self.data]
        
        encodings = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        offset_mappings = encodings['offset_mapping']
        
        encoded = []
        
        for i, item in enumerate(self.data):
            text = item['text']
            offset_mapping = offset_mappings[i]
            ner_tags = item.get('ner_tags', [])
            
            labels = self._align_labels(ner_tags, offset_mapping)
            
            lexicon_indices, lexicon_levels, lexicon_types, lexicon_coords = self._match_lexicon(text, offset_mapping)
            
            rel_labels = self._get_rel_labels(item, offset_mapping)
            event_labels = self._get_event_labels(item, offset_mapping)
            
            encoded.append({
                'input_ids': input_ids[i],
                'attention_mask': attention_mask[i],
                'labels': labels,
                'ner_tags': labels,
                'lexicon_indices': lexicon_indices,
                'lexicon_levels': lexicon_levels,
                'lexicon_types': lexicon_types,
                'lexicon_coords': lexicon_coords,
                'text': text,
                'rel_labels': rel_labels,
                'event_labels': event_labels,
                'spo_list': item.get('spo_list', []),
                'event_list': item.get('event_list', [])
            })
        
        return encoded
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        """获取单个样本（直接返回预编码数据）"""
        return self.encoded_data[idx]
    
    def _align_labels(self, ner_tags, offset_mapping):
        """
        将原始标签对齐到tokenized后的位置
        Args:
            ner_tags: 原始字符级标签列表
            offset_mapping: token偏移映射
        Returns:
            对齐后的标签张量
        """
        labels = torch.full((self.max_seq_length,), -100, dtype=torch.long)
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                continue  # 特殊token
            if start >= len(ner_tags):
                break
            
            # 获取原始标签
            original_tag = ner_tags[start]
            labels[i] = cfg.ner_label2id.get(original_tag, 0)
        
        return labels
    
    def _match_lexicon(self, text, offset_mapping):
        """
        词典匹配，返回每个token对应的词汇ID、层级ID、类型ID和坐标
        Args:
            text: 原始文本
            offset_mapping: token偏移映射
        Returns:
            lexicon_indices: 词汇ID列表（0表示padding/无匹配）
            lexicon_levels: 层级ID列表（0表示padding/未知）
            lexicon_types: 类型ID列表（0表示padding/未知）
            lexicon_coords: 坐标特征列表（[0,0]表示无坐标）
        """
        lexicon_indices = torch.zeros((self.max_seq_length,), dtype=torch.long)
        lexicon_levels = torch.zeros((self.max_seq_length,), dtype=torch.long)
        lexicon_types = torch.zeros((self.max_seq_length,), dtype=torch.long)
        lexicon_coords = torch.zeros((self.max_seq_length, 2), dtype=torch.float)
        
        if self.lexicon_matcher is None:
            return lexicon_indices, lexicon_levels, lexicon_types, lexicon_coords
        
        # 使用 match_with_attributes 获取带属性的匹配结果
        matches = self.lexicon_matcher.match_with_attributes(text)
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                continue
            
            for match_start, match_end, word, word_id, level_id, type_id, coords in matches:
                if start >= match_start and end <= match_end + 1:
                    lexicon_indices[i] = word_id
                    lexicon_levels[i] = level_id
                    lexicon_types[i] = type_id
                    if coords is not None:
                        # 归一化坐标：纬度/90，经度/180
                        lexicon_coords[i][0] = coords[0] / 90.0
                        lexicon_coords[i][1] = coords[1] / 180.0
                    break
        
        return lexicon_indices, lexicon_levels, lexicon_types, lexicon_coords
    
    def _get_rel_labels(self, item, offset_mapping):
        """
        获取关系抽取标签
        Args:
            item: 数据项
            offset_mapping: token偏移映射
        Returns:
            rel_labels: 关系标签矩阵 [seq_len x seq_len]
        """
        rel_labels = torch.full((self.max_seq_length, self.max_seq_length), -100, dtype=torch.long)
        
        if 'spo_list' not in item:
            return rel_labels
        
        spo_list = item['spo_list']
        
        for spo in spo_list:
            subj_start = spo.get('subject_start', -1)
            subj_end = spo.get('subject_end', -1)
            obj_start = spo.get('object_start', -1)
            obj_end = spo.get('object_end', -1)
            relation = spo.get('relation', '无关系')
            
            # 将字符位置对齐到token位置
            subj_token_start = self._char_to_token(offset_mapping, subj_start)
            subj_token_end = self._char_to_token(offset_mapping, subj_end)
            obj_token_start = self._char_to_token(offset_mapping, obj_start)
            obj_token_end = self._char_to_token(offset_mapping, obj_end)
            
            if subj_token_start != -1 and obj_token_start != -1:
                rel_label = cfg.rel_label2id.get(relation, 0)
                rel_labels[subj_token_start:subj_token_end+1, obj_token_start:obj_token_end+1] = rel_label
        
        return rel_labels
    
    def _get_event_labels(self, item, offset_mapping):
        """
        获取事件抽取标签（多标签分类）
        Args:
            item: 数据项
            offset_mapping: token偏移映射
        Returns:
            event_labels: 事件标签矩阵 [seq_len x num_event_types]
            使用 0/1 表示是否为该事件类型，配合 mask 使用
        """
        num_event_types = len(cfg.event_labels)
        # 使用 0 初始化（无事件），0/1 表示是否为该事件类型
        event_labels = torch.zeros((self.max_seq_length, num_event_types), dtype=torch.float)
        
        if 'event_list' not in item:
            return event_labels
        
        event_list = item['event_list']
        
        for event in event_list:
            event_type = event.get('event_type', '无事件')
            trigger_start = event.get('trigger_start', -1)
            trigger_end = event.get('trigger_end', -1)
            
            trigger_token_start = self._char_to_token(offset_mapping, trigger_start)
            trigger_token_end = self._char_to_token(offset_mapping, trigger_end)
            
            if trigger_token_start != -1:
                event_idx = cfg.event_label2id.get(event_type, 0)
                # 设置触发词位置为该事件类型（1表示属于该事件）
                event_labels[trigger_token_start:trigger_token_end+1, event_idx] = 1
        
        return event_labels
    
    def _char_to_token(self, offset_mapping, char_pos):
        """
        将字符位置转换为token位置
        Args:
            offset_mapping: token偏移映射
            char_pos: 字符位置
        Returns:
            token位置（-1表示未找到）
        """
        if char_pos == -1:
            return -1
        
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_pos < end:
                return i
        return -1
