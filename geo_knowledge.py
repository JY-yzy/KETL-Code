#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
地理知识图谱数据处理模块
支持地名别名、层级关系、地理坐标等知识的加载和匹配
"""
import os
import json
import re
from collections import defaultdict

class GeoKnowledgeGraph:
    """地理知识图谱类（扩展版）"""
    
    TYPE_MAPPING = {
        'PER': 3,
        'ORG': 2,
        'GEO': 1,
        'ADMIN': 1,
        'LANDMARK': 1,
        'MT': 1,
        'MTS': 1,
        'STM': 1,
        'LAKE': 1,
        'PASS': 1,
        'PK': 1,
        'VAL': 1,
        'ISL': 1,
        '城市': 1,
        '省': 1,
        '自治区': 1,
        '直辖市': 1,
        '特别行政区': 1,
    }
    
    LEVEL_MAPPING = {
        '省级': 2,
        '省': 2,
        '自治区': 2,
        '直辖市': 2,
        '特别行政区': 2,
        '地级市': 3,
        '自治州': 3,
        '地区': 3,
        '盟': 3,
        '县级市': 4,
        '市辖区': 4,
        '县': 4,
        '自治县': 4,
        '旗': 4,
        '镇': 5,
        '乡': 5,
        '村': 6,
    }
    
    def __init__(self, data_dir, cache_dir=None, load_ownthink=False, skip_cache=False):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, 'processed')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.lexicon = []
        self.alias_map = {}
        self.hierarchy_map = {}
        self.type_map = {}
        self.coord_map = {}
        self.level_names = ['Unknown', 'Country', 'Province', 'City', 'District', 'Town', 'Village']
        
        self._load_all_knowledge(load_ownthink, skip_cache)
    
    def _load_all_knowledge(self, load_ownthink=False, skip_cache=False):
        """加载所有地理知识文件（优化版）"""
        print("Loading geographic knowledge files...")
        
        # 缓存文件名包含筛选参数指纹，确保参数变化时使用新缓存（不污染旧缓存）
        if load_ownthink:
            _mo = os.environ.get('OWNTHINK_MAX_ORG', '200000')
            _mp = os.environ.get('OWNTHINK_MAX_PER', '200000')
            _mn = os.environ.get('OWNTHINK_MIN_LEN', '2')
            _mx = os.environ.get('OWNTHINK_MAX_LEN', '15')
            cache_filename = f'geo_knowledge_cache_ownthink_org{_mo}_per{_mp}_min{_mn}_max{_mx}.json'
        else:
            cache_filename = 'geo_knowledge_cache.json'
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if not skip_cache and os.path.exists(cache_path):
            print(f"  - Loading from cache: {cache_path}")
            try:
                with open(cache_path, 'r', encoding='utf-8-sig') as f:
                    cached = json.load(f)
                    self.lexicon = cached['lexicon']
                    self.alias_map = cached['alias_map']
                    self.hierarchy_map = cached['hierarchy_map']
                    self.type_map = cached['type_map']
                    self.coord_map = cached['coord_map']
                print(f"  - Loaded {len(self.lexicon)} entries from cache")
                return
            except Exception as e:
                print(f"  - Cache load failed ({e}), reloading from files...")
        elif skip_cache:
            print(f"  - Skipping cache (force reload), will process raw data files...")
        
        self._load_geo_lexicon()
        self._load_city_information()
        self._load_records_json()
        self._load_cn_txt()
        
        if load_ownthink:
            print(f"\n  - Loading ownthink_v2.csv (this may take several minutes)...")
            self._load_ownthink_full()
        
        self._generate_alias_map()
        
        self._save_to_cache(cache_filename)
        
        print(f"\n[GeoKnowledgeGraph] 加载完成:")
        print(f"  - 词典大小: {len(self.lexicon)}")
        print(f"  - 别名映射: {len(self.alias_map)}")
        print(f"  - 层级信息: {len(self.hierarchy_map)}")
        print(f"  - 类型信息: {len(self.type_map)}")
        print(f"  - 坐标信息: {len(self.coord_map)}")
    
    def _save_to_cache(self, filename='geo_knowledge_cache.json'):
        """保存到缓存"""
        cache_path = os.path.join(self.cache_dir, filename)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'lexicon': self.lexicon,
                    'alias_map': self.alias_map,
                    'hierarchy_map': self.hierarchy_map,
                    'type_map': self.type_map,
                    'coord_map': self.coord_map,
                }, f, ensure_ascii=False)
            print(f"  - Saved to cache: {cache_path}")
        except Exception as e:
            print(f"  - Cache save failed ({e}), skipping")
    
    def _is_valid_geo_name(self, name):
        """GEO 实体名质量过滤（应用于所有 GEO 源数据）

        实验逻辑：地名识别为主，需保证 GEO 数据质量
        - 长度 1-20 字（允许单字地名缩写如"京"、"沪"等）
        - 必须含中文
        - 无 HTML/URL/特殊符号
        """
        if not name or len(name) < 1 or len(name) > 20:
            return False
        # 必须包含至少一个中文字符
        if not any('\u4e00' <= c <= '\u9fff' for c in name):
            return False
        # 不能包含特殊字符
        import re as _re
        if _re.search(r'[<>"\'\\\[\]{}()=;|!?@#$%^*+~`\n\r\t]', name):
            return False
        if _re.search(r'http|www|\.com|\.cn|\.org|\.net', name):
            return False
        # 不能以方括号、问号结尾（之前发现的GEO质量问题）
        if name.endswith('[') or name.endswith(']') or name.endswith('?'):
            return False
        return True

    def _load_geo_lexicon(self):
        """加载基础词典（带质量过滤）"""
        lexicon_path = os.path.join(self.data_dir, 'geo_lexicon.txt')
        if os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8-sig') as f:
                raw_names = [line.strip() for line in f if line.strip()]
            # 应用质量过滤
            names = [n for n in raw_names if self._is_valid_geo_name(n)]
            filtered_count = len(raw_names) - len(names)
            self.lexicon.extend(names)
            for name in names:
                self.type_map[name] = 1
            print(f"  - Loaded {len(names)} entries from geo_lexicon.txt (filtered out {filtered_count} low-quality)")
        else:
            print(f"  - Warning: geo_lexicon.txt not found at: {lexicon_path}")
    
    def _load_city_information(self):
        """加载城市信息.csv（带质量过滤）"""
        city_path = os.path.join(self.data_dir, 'China_main_city_information', 'graph', '城市信息.csv')
        if os.path.exists(city_path):
            import csv
            count = 0
            filtered_count = 0
            with open(city_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    city_name = row.get('城市名', '').strip()
                    alias = row.get('别名', '').strip()
                    admin_level = row.get('行政级别', '').strip()
                    province = row.get('省份', '').strip()
                    
                    if city_name and self._is_valid_geo_name(city_name):
                        if city_name not in self.type_map:
                            self.lexicon.append(city_name)
                            self.type_map[city_name] = 1
                        level = self._get_level_from_admin(admin_level)
                        self.hierarchy_map[city_name] = level
                        count += 1
                    elif city_name:
                        filtered_count += 1
                    
                    if alias and alias != 'nan':
                        alias_parts = [a.strip() for a in alias.split('/') if a.strip()]
                        for a in alias_parts:
                            if a and a not in self.alias_map:
                                self.alias_map[a] = city_name
                    
                    if province and self._is_valid_geo_name(province) and province not in self.type_map:
                        self.lexicon.append(province)
                        self.type_map[province] = 1
                        self.hierarchy_map[province] = 2
            print(f"  - Loaded {count} cities from 城市信息.csv (filtered out {filtered_count} low-quality)")
        else:
            print(f"  - Warning: 城市信息.csv not found at: {city_path}")
    
    def _load_records_json(self):
        """加载records.json（补充城市信息）"""
        records_path = os.path.join(self.data_dir, 'China_main_city_information', 'html', 'data', 'records.json')
        print(f"  - Trying to load records.json from: {records_path}")
        if os.path.exists(records_path):
            try:
                with open(records_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                print(f"  - records.json loaded successfully, total items: {len(data)}")
                
                count = 0
                skipped = 0
                filtered_count = 0
                for item in data:
                    if isinstance(item, dict) and 'p' in item:
                        p = item['p']
                        start = p.get('start', {})
                        props = start.get('properties', {})
                        city_name = props.get('name', '').strip()
                        another_name = props.get('anothername', '').strip()
                        
                        if city_name and self._is_valid_geo_name(city_name):
                            if city_name not in self.type_map:
                                self.lexicon.append(city_name)
                                self.type_map[city_name] = 1
                                self.hierarchy_map[city_name] = 3
                                count += 1
                            else:
                                skipped += 1
                        elif city_name:
                            filtered_count += 1
                        
                        if another_name and another_name != 'nan' and another_name not in self.alias_map:
                            self.alias_map[another_name] = city_name
                print(f"  - Loaded {count} cities from records.json (skipped {skipped} duplicates, filtered {filtered_count} low-quality)")
            except Exception as e:
                print(f"  - Skipped records.json (error: {type(e).__name__}: {str(e)})")
        else:
            print(f"  - Warning: records.json not found at: {records_path}")
    
    def _load_cn_txt(self):
        """加载CN.txt（自然地理实体，带质量过滤）"""
        cn_path = os.path.join(self.data_dir, 'CN.txt')
        if os.path.exists(cn_path):
            count = 0
            filtered_count = 0
            with open(cn_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        names_str = parts[3]
                        lat = parts[4]
                        lon = parts[5]
                        feature_type = parts[7]
                        
                        names = [n.strip() for n in names_str.split(',') if n.strip()]
                        type_id = self.TYPE_MAPPING.get(feature_type, 1)
                        
                        for name in names:
                            if self._is_valid_geo_name(name) and name not in self.type_map:
                                self.lexicon.append(name)
                                self.type_map[name] = type_id
                                count += 1
                            elif name and name not in self.type_map:
                                filtered_count += 1
                        
                        if names and lat.replace('.', '').isdigit() and lon.replace('.', '').isdigit():
                            primary_name = names[0]
                            try:
                                self.coord_map[primary_name] = (float(lat), float(lon))
                            except:
                                pass
            print(f"  - Loaded {count} geographic entities from CN.txt (filtered out {filtered_count} low-quality)")
        else:
            print(f"  - Warning: CN.txt not found at: {cn_path}")
    
    def _load_ownthink_full(self):
        """加载ownthink_v2.csv（基于实验逻辑的智能筛选版）

        实验逻辑：
        1. NER 任务需识别 GEO/ADMIN/LANDMARK/ORG/PER 五类实体
        2. GEO/ADMIN/LANDMARK 已由 geo_lexicon.txt、城市信息.csv、records.json、CN.txt 完整覆盖
        3. ownthink 仅补充 ORG（组织机构）和 PER（人物）实体，避免类型重复

        智能筛选策略（替代粗暴截断）：
        A. 质量过滤：实体名长度 2-15 字（NER 文本合理范围）、必须含中文、无特殊字符
        B. 类型严格判定：PER 必须有 per_attrs，ORG 必须有 org_keywords 或 org_attrs
        C. 属性丰富度统计：有越多属性/坐标/别名的实体越重要
        D. 类型配额筛选：ORG/PER 各限 20 万，按属性丰富度排序后取 Top-K
        E. 真正去重：跨 chunk 去重 + 别名不进入 lexicon（仅入 alias_map）
        """
        kg_path = os.path.join(self.data_dir, 'GeoKG', 'ownthink_v2.csv')
        if not os.path.exists(kg_path):
            print(f"  - ownthink_v2.csv not found at: {kg_path}")
            print(f"  - Skipping ownthink knowledge loading")
            return

        print("  - Loading ownthink_v2.csv (8GB, intelligent filtering: ORG/PER only)...")
        print(f"  - Existing lexicon before ownthink: {len(self.lexicon)} entries")

        import pandas as pd
        import gc
        import re as _re

        # 筛选参数：从环境变量或使用默认值（避免重新实例化Config影响随机种子）
        # 这些值在 config_ner.py 中也可配置，但此处提供合理默认值
        max_org = int(os.environ.get('OWNTHINK_MAX_ORG', '200000'))
        max_per = int(os.environ.get('OWNTHINK_MAX_PER', '200000'))
        min_len = int(os.environ.get('OWNTHINK_MIN_LEN', '2'))
        max_len = int(os.environ.get('OWNTHINK_MAX_LEN', '15'))
        print(f"  - Filter config: min_len={min_len}, max_len={max_len}, max_org={max_org}, max_per={max_per}")
        print(f"    (set env OWNTHINK_MAX_ORG/OWNTHINK_MAX_PER/OWNTHINK_MIN_LEN/OWNTHINK_MAX_LEN to override)")

        # ORG 关键词：组织机构类实体
        org_keywords = {'公司', '集团', '企业', '机构', '组织', '大学', '学院', '医院',
                        '银行', '基金', '协会', '委员会', '局', '部', '厅', '院', '所',
                        '站', '台', '政府', '馆', '报社', '出版社', '电台', '电视台',
                        '研究所', '研究院', '法院', '检察院', '司令部', '部队'}

        # PER 相关属性：人物类实体（强信号）
        per_attrs = {'中文名', '外文名', '姓名', '国籍', '出生日期', '出生地',
                     '逝世日期', '逝世地', '职业', '性别', '民族', '籍贯',
                     '毕业院校', '配偶', '子女', '父母', '兄弟姐妹'}

        # ORG 相关属性：组织机构类实体（强信号）
        org_attrs = {'成立日期', '成立时间', '总部地点', '总部', '创始人',
                     '员工数', '营业收入', '注册资本', '法定代表人'}

        # 特殊字符过滤正则：保留中文、字母、数字、连接符；剔除 HTML/URL/特殊符号
        invalid_char_pattern = _re.compile(
            r'[<>"\'\\\[\]{}()=;|!?@#$%^*+~`]'
            r'|http|https|www|\.com|\.cn|\.org|\.net'
            r'|&[a-z]+;|\n|\r|\t'
        )

        def is_valid_entity_name(name):
            """质量过滤：实体名有效性检查"""
            if not name or len(name) < min_len or len(name) > max_len:
                return False
            # 必须包含至少一个中文字符
            if not any('\u4e00' <= c <= '\u9fff' for c in name):
                return False
            # 不能包含特殊字符
            if invalid_char_pattern.search(name):
                return False
            # 不能是纯数字或纯标点
            chinese_count = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
            if chinese_count < 1:
                return False
            return True

        # ============ 第一阶段：扫描 + 质量过滤 + 属性丰富度统计 ============
        # entity_attrs: {entity_name: set(attrs)} - 记录每个实体的属性集合
        # entity_type_hint: {entity_name: int} - 0=未确定, 2=ORG, 3=PER
        entity_attrs = defaultdict(set)
        entity_type_hint = {}
        # 同步收集坐标和别名（用于评估重要性）
        entity_coords = {}   # entity -> (lat, lon)
        entity_aliases = defaultdict(list)  # entity -> [alias1, alias2, ...]

        chunk_num = 0
        chunksize = 200000
        valid_count = 0
        filtered_quality = 0
        filtered_type = 0

        try:
            tqdm = __import__('tqdm').tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        print("  - Phase 1: Scanning and quality filtering...")
        for chunk in pd.read_csv(kg_path, chunksize=chunksize, usecols=['实体', '属性', '值'],
                                 dtype=str, encoding='utf-8-sig', keep_default_na=False):
            chunk_num += 1

            chunk['实体'] = chunk['实体'].astype(str).str.strip()
            chunk['属性'] = chunk['属性'].astype(str).str.strip()
            chunk['值'] = chunk['值'].astype(str).str.strip()

            # 基础过滤：非空
            mask_valid = (chunk['实体'] != '') & (chunk['属性'] != '')
            chunk = chunk[mask_valid]

            # 质量过滤：实体名长度、中文、无特殊字符
            mask_quality = chunk['实体'].apply(is_valid_entity_name)
            filtered_quality += (~mask_quality).sum()
            chunk = chunk[mask_quality]

            if len(chunk) == 0:
                del chunk
                continue

            # 类型判定：必须明确为 ORG 或 PER
            # ORG 信号：实体名含 org_keywords 或属性属 org_attrs
            # PER 信号：属性属 per_attrs
            mask_org_entity = chunk['实体'].str.contains('|'.join(org_keywords), regex=True)
            mask_per_attr = chunk['属性'].isin(per_attrs)
            mask_org_attr = chunk['属性'].isin(org_attrs)
            mask_is_org = mask_org_entity | mask_org_attr
            mask_is_per = mask_per_attr
            mask_typed = mask_is_org | mask_is_per
            filtered_type += (~mask_typed).sum()
            chunk = chunk[mask_typed]

            if len(chunk) == 0:
                del chunk
                continue

            # 跳过已在 type_map 中的实体（GEO/ADMIN/LANDMARK 已覆盖）
            chunk_new = chunk[~chunk['实体'].isin(self.type_map)]
            skipped_dup_in_geo = (len(chunk) - len(chunk_new)) // 2  # 近似去重计数
            chunk = chunk_new

            if len(chunk) == 0:
                del chunk
                continue

            # 累积每个实体的属性集合和类型提示
            for entity, attr, value in zip(chunk['实体'], chunk['属性'], chunk['值']):
                # 类型判定（ORG 关键词优先，避免学校/医院等被误判为PER）
                # 实验逻辑：以地名识别为主，ORG（含设施名）比PER更重要
                is_org_entity = any(kw in entity for kw in org_keywords)
                if is_org_entity:
                    # 实体名含"大学/医院/公司"等ORG关键词，强制判定为ORG
                    # （即使有"毕业院校"属性，也是指该机构的毕业生，而非机构本身是人物）
                    entity_type_hint[entity] = 2
                elif attr in per_attrs:
                    # 仅当实体名不含ORG关键词，且属性是PER强信号时，才判为PER
                    entity_type_hint[entity] = 3
                elif attr in org_attrs:
                    # ORG相关属性（成立日期、创始人等）
                    if entity_type_hint.get(entity, 0) != 3:
                        entity_type_hint[entity] = 2
                elif entity not in entity_type_hint:
                    entity_type_hint[entity] = 0

                # 属性集合
                entity_attrs[entity].add(attr)
                valid_count += 1

                # 坐标
                if attr in ['地理坐标', '经纬度', '坐标'] and entity not in entity_coords:
                    try:
                        coord_clean = value.replace('(', '').replace(')', '').strip()
                        parts = coord_clean.split(',')
                        if len(parts) >= 2:
                            lat = float(parts[0].strip())
                            lon = float(parts[1].strip())
                            entity_coords[entity] = (lat, lon)
                    except (ValueError, IndexError):
                        pass

                # 别名
                if attr == '别名' and value:
                    aliases = [a.strip() for a in value.split('、') if a.strip()]
                    for alias in aliases:
                        if alias and alias[0] >= '\u4e00' and alias[0] <= '\u9fff':
                            if alias not in entity_aliases[entity]:
                                entity_aliases[entity].append(alias)

            if chunk_num % 20 == 0:
                print(f"    Phase1 Chunk {chunk_num}: candidates={len(entity_attrs)}, quality_filtered={filtered_quality}, type_filtered={filtered_type}")

            del chunk, mask_valid, mask_quality, mask_typed
            gc.collect()

        print(f"  - Phase 1 complete: {len(entity_attrs)} candidate entities")
        print(f"    Filtered out: quality={filtered_quality}, type_undetermined={filtered_type}")

        # ============ 第二阶段：类型严格判定 + 属性丰富度排序 + 配额筛选 ============
        print("  - Phase 2: Type determination and importance ranking...")

        org_candidates = []   # (entity, importance_score)
        per_candidates = []
        skipped_no_type = 0

        for entity, attrs in entity_attrs.items():
            type_id = entity_type_hint.get(entity, 0)
            # 严格类型判定：必须有明确类型信号
            if type_id == 0:
                skipped_no_type += 1
                continue

            # 重要性评分：属性数 * 1 + 有坐标 * 3 + 有别名 * 2
            score = len(attrs)
            if entity in entity_coords:
                score += 3
            if entity in entity_aliases and len(entity_aliases[entity]) > 0:
                score += 2

            if type_id == 2:
                org_candidates.append((entity, score))
            elif type_id == 3:
                per_candidates.append((entity, score))

        # 按重要性评分降序排序，取 Top-K
        org_candidates.sort(key=lambda x: x[1], reverse=True)
        per_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_org = org_candidates[:max_org]
        selected_per = per_candidates[:max_per]

        print(f"  - Type determination: ORG candidates={len(org_candidates)}, PER candidates={len(per_candidates)}")
        print(f"    Skipped (no clear type): {skipped_no_type}")
        print(f"  - After quota: ORG={len(selected_org)} (limit={max_org}), PER={len(selected_per)} (limit={max_per})")
        if org_candidates:
            print(f"    ORG score range: [{selected_org[-1][1]}, {org_candidates[0][1]}]")
        if per_candidates:
            print(f"    PER score range: [{selected_per[-1][1]}, {per_candidates[0][1]}]")

        # ============ 第三阶段：写入词典 ============
        print("  - Phase 3: Writing to lexicon...")

        added_count = 0
        coord_count = 0
        alias_count = 0

        for entity, _ in selected_org:
            if entity not in self.type_map:  # 双重去重检查
                self.lexicon.append(entity)
                self.type_map[entity] = 2
                added_count += 1
                if entity in entity_coords and entity not in self.coord_map:
                    self.coord_map[entity] = entity_coords[entity]
                    coord_count += 1
                if entity in entity_aliases:
                    for alias in entity_aliases[entity]:
                        if alias not in self.alias_map:
                            self.alias_map[alias] = entity
                            alias_count += 1

        for entity, _ in selected_per:
            if entity not in self.type_map:
                self.lexicon.append(entity)
                self.type_map[entity] = 3
                added_count += 1
                if entity in entity_coords and entity not in self.coord_map:
                    self.coord_map[entity] = entity_coords[entity]
                    coord_count += 1
                if entity in entity_aliases:
                    for alias in entity_aliases[entity]:
                        if alias not in self.alias_map:
                            self.alias_map[alias] = entity
                            alias_count += 1

        print(f"  - Loaded {added_count} ORG/PER entities from ownthink_v2.csv")
        print(f"    (coords: {coord_count}, aliases: {alias_count})")
        print(f"  - Total lexicon after ownthink: {len(self.lexicon)} entries")

        # 释放中间数据
        del entity_attrs, entity_type_hint, entity_coords, entity_aliases
        del org_candidates, per_candidates, selected_org, selected_per
        gc.collect()
    
    def _infer_type_from_entity(self, entity):
        """仅从实体名推断类型（用于向量化处理）"""
        if any(kw in entity for kw in ['公司', '集团', '企业', '机构', '组织', '大学', '学院', '医院']):
            return 2
        if any(kw in entity for kw in ['省', '市', '区', '县', '镇', '乡', '村', '自治区', '直辖市']):
            return 1
        if any(kw in entity for kw in ['山', '河', '湖', '海', '江', '峰', '岭', '高原', '平原', '盆地']):
            return 1
        return 1
    
    def _infer_type_from_ownthink(self, entity, attr, value):
        """从ownthink数据推断实体类型"""
        if any(kw in entity for kw in ['公司', '集团', '企业', '机构', '组织', '大学', '学院', '医院']):
            return 2
        if any(kw in entity for kw in ['省', '市', '区', '县', '镇', '乡', '村', '自治区', '直辖市']):
            return 1
        if any(kw in entity for kw in ['山', '河', '湖', '海', '江', '峰', '岭', '高原', '平原', '盆地']):
            return 1
        if attr in ['中文名', '外文名', '姓名', '国籍', '出生日期']:
            return 3
        if any(kw in attr for kw in ['公司', '企业', '机构']):
            return 2
        return 1
    
    def _get_level_from_admin(self, admin_level):
        """从行政级别字符串获取层级ID"""
        for key, level in self.LEVEL_MAPPING.items():
            if key in admin_level:
                return level
        return 0
    
    def _generate_alias_map(self):
        """自动生成别名映射"""
        print("  - Generating alias map...")
        name_variants = defaultdict(list)
        
        # 常见地名别名模式
        alias_patterns = [
            (r'省$', ''),      # 去掉省字
            (r'市$', ''),      # 去掉市字
            (r'区$', ''),      # 去掉区字
            (r'县$', ''),      # 去掉县字
            (r'镇$', ''),      # 去掉镇字
            (r'乡$', ''),      # 去掉乡字
            (r'村$', ''),      # 去掉村字
            (r'自治区$', ''),  # 去掉自治区
            (r'自治州$', ''),  # 去掉自治州
            (r'特别行政区$', ''),  # 去掉特别行政区
            (r'市辖区$', ''),  # 去掉市辖区
        ]
        
        for name in self.lexicon:
            original_name = name
            name_variants[original_name].append(original_name)
            
            # 生成变体
            for pattern, replacement in alias_patterns:
                variant = re.sub(pattern, replacement, name)
                if variant != name and variant:
                    name_variants[original_name].append(variant)
            
            # 拼音简写（如北京→BJ）
            pinyin_short = ''.join([c[0] for c in name if '\u4e00' <= c <= '\u9fff'][:3])
            if pinyin_short:
                name_variants[original_name].append(pinyin_short)
        
        # 构建别名→标准名映射
        for standard_name, variants in name_variants.items():
            for variant in variants:
                if variant != standard_name:
                    self.alias_map[variant] = standard_name
    
    def _load_coordinates_from_knowledge_graph(self):
        """从知识图谱加载坐标信息"""
        kg_path = os.path.join(self.data_dir, 'GeoKG', 'ownthink_v2.csv')
        if os.path.exists(kg_path):
            print("  - Loading coordinates from knowledge graph...")
            import csv
            with open(kg_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        entity, relation, value = row[0], row[1], row[2]
                        if relation in ['地理坐标', '经纬度', '坐标']:
                            coords = value.strip()
                            if coords:
                                try:
                                    # 解析坐标格式：纬度,经度 或 (纬度,经度)
                                    coords = coords.replace('(', '').replace(')', '')
                                    lat, lon = coords.split(',')
                                    self.coord_map[entity] = (float(lat.strip()), float(lon.strip()))
                                except:
                                    pass
    
    def get_entity_info(self, name):
        """获取实体的完整信息"""
        # 先检查别名
        standard_name = self.alias_map.get(name, name)
        
        return {
            'standard_name': standard_name,
            'level': self.hierarchy_map.get(standard_name, 0),
            'level_name': self.level_names[self.hierarchy_map.get(standard_name, 0)] if self.hierarchy_map.get(standard_name) else 'Unknown',
            'type': self.type_map.get(standard_name, 0),
            'coordinates': self.coord_map.get(standard_name, None)
        }
    
    def match_text(self, text):
        """在文本中匹配地名实体"""
        matches = []
        
        # 按长度降序排列词典，优先匹配长地名
        sorted_names = sorted(self.lexicon, key=lambda x: -len(x))
        
        for name in sorted_names:
            start = 0
            while start < len(text):
                idx = text.find(name, start)
                if idx == -1:
                    break
                info = self.get_entity_info(name)
                matches.append({
                    'text': name,
                    'start': idx,
                    'end': idx + len(name),
                    'standard_name': info['standard_name'],
                    'level': info['level'],
                    'level_name': info['level_name'],
                    'type': info['type'],
                    'coordinates': info['coordinates']
                })
                start = idx + len(name)
        
        # 也检查别名匹配
        for alias, standard_name in self.alias_map.items():
            start = 0
            while start < len(text):
                idx = text.find(alias, start)
                if idx == -1:
                    break
                info = self.get_entity_info(standard_name)
                matches.append({
                    'text': alias,
                    'start': idx,
                    'end': idx + len(alias),
                    'standard_name': info['standard_name'],
                    'level': info['level'],
                    'level_name': info['level_name'],
                    'type': info['type'],
                    'coordinates': info['coordinates']
                })
                start = idx + len(alias)
        
        # 去重并按位置排序
        matches = sorted(matches, key=lambda x: (x['start'], -len(x['text'])))
        final_matches = []
        last_end = -1
        for match in matches:
            if match['start'] >= last_end:
                final_matches.append(match)
                last_end = match['end']
        
        return final_matches

def build_enhanced_lexicon_matcher(knowledge_graph, lexicon_size=500000):
    """构建增强版词典匹配器"""
    lexicon_list = knowledge_graph.lexicon[:lexicon_size]
    
    # 创建ID映射
    name_to_id = {name: i + 1 for i, name in enumerate(lexicon_list)}
    level_ids = [knowledge_graph.hierarchy_map.get(name, 0) for name in lexicon_list]
    type_ids = [knowledge_graph.type_map.get(name, 0) for name in lexicon_list]
    
    # 坐标嵌入（如果有坐标）
    coord_embeddings = []
    for name in lexicon_list:
        coords = knowledge_graph.coord_map.get(name)
        if coords:
            coord_embeddings.append([coords[0] / 90.0, coords[1] / 180.0])
        else:
            coord_embeddings.append([0.0, 0.0])
    
    return {
        'lexicon': lexicon_list,
        'name_to_id': name_to_id,
        'level_ids': level_ids,
        'type_ids': type_ids,
        'coord_embeddings': coord_embeddings,
        'alias_map': knowledge_graph.alias_map
    }

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    kg = GeoKnowledgeGraph(data_dir)
    
    # 测试匹配
    test_text = "我从北京出发，经过上海浦东新区，最后到达广州天河区"
    matches = kg.match_text(test_text)
    print(f"\nTest text: {test_text}")
    print("Matched entities:")
    for match in matches:
        print(f"  - {match['text']} (type: {match['level_name']}, level: {match['level']})")