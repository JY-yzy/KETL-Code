#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本
功能：
1. 读取 CLUENER、MSRA、Weibo NER、CMNER 原始数据，统一标签为 GEO、ADMIN、LANDMARK、ORG、PER
2. 转换为 BIO 格式，保存为预处理后的 JSON 文件
3. 读取 DuIE 2.0 和 DuEE 1.0，转换为统一格式
4. 构建地理词典 geo_lexicon.txt（从 GeoNames 和城市知识图谱提取）
5. 生成词汇到 ID 的映射文件 lexicon_vocab.json
"""
import os
import json
import re
from collections import defaultdict


# ========================== 配置 ==========================
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# 标签映射
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

# ========================== NER数据处理 ==========================

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

# -------------------------- CLUENER --------------------------

def load_cluener_raw(file_path):
    """加载原始CLUENER数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            labels = item.get('label', {})
            data.append({'text': text, 'labels': labels})
    return data

def cluener_to_bio(data):
    """将CLUENER数据转换为BIO格式"""
    bio_data = []
    for item in data:
        text = item['text']
        labels = item['labels']
        bio_tags = ['O'] * len(text)
        for entity_type, entities in labels.items():
            unified_type = DATASET_LABEL_MAP['cluener'].get(entity_type, None)
            if unified_type is None:
                continue
            for entity_name, spans in entities.items():
                for start, end in spans:
                    if start < len(text) and end <= len(text) and start < end:
                        bio_tags[start] = f'B-{unified_type}'
                        for i in range(start + 1, end):
                            if i < len(text):
                                bio_tags[i] = f'I-{unified_type}'
        bio_data.append({'text': text, 'ner_tags': bio_tags})
    return bio_data

# -------------------------- MSRA --------------------------

def load_msra_raw(file_path):
    """加载原始MSRA数据"""
    sentences = []
    current_sentence = []
    current_tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
                    current_sentence = []
                    current_tags = []
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                current_sentence.append(parts[0])
                current_tags.append(unify_tag(parts[1], 'msra'))
    if current_sentence:
        sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
    return sentences

# -------------------------- Weibo NER --------------------------

def load_weibo_ner_raw(file_path):
    """加载原始Weibo NER数据"""
    sentences = []
    current_sentence = []
    current_tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
                    current_sentence = []
                    current_tags = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                current_sentence.append(parts[0])
                current_tags.append(unify_tag(parts[1], 'weibo'))
    if current_sentence:
        sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
    return sentences

# -------------------------- CMNER --------------------------

def load_cmner_raw(file_path):
    """加载原始CMNER数据"""
    sentences = []
    current_sentence = []
    current_tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
                    current_sentence = []
                    current_tags = []
                continue
            if line.startswith('WID:'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                current_sentence.append(parts[0])
                current_tags.append(unify_tag(parts[1], 'cmner'))
    if current_sentence:
        sentences.append({'text': ''.join(current_sentence), 'ner_tags': current_tags.copy()})
    return sentences

# ========================== DuIE处理 ==========================

def load_duie_raw(file_path):
    """加载原始DuIE数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item['text']
            spo_list = item.get('spo_list', [])
            
            processed_spo = []
            for spo in spo_list:
                subject = spo.get('subject', '')
                predicate = spo.get('predicate', '')
                obj_data = spo.get('object', {})
                obj_value = obj_data.get('@value', '') if isinstance(obj_data, dict) else str(obj_data)
                
                subject_type = spo.get('subject_type', '')
                object_type = spo.get('object_type', {})
                obj_type_value = object_type.get('@value', '') if isinstance(object_type, dict) else str(object_type)
                
                processed_spo.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj_value,
                    'subject_type': subject_type,
                    'object_type': obj_type_value
                })
            
            data.append({
                'text': text,
                'spo_list': processed_spo
            })
    return data

# ========================== DuEE处理 ==========================

def load_duee_raw(file_path):
    """加载原始DuEE数据（支持JSON格式和CSV格式）"""
    data = []
    
    # 尝试JSON格式
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip().startswith('[') or content.strip().startswith('{'):
                items = json.loads(content)
                for item in items:
                    processed = process_duee_json(item)
                    if processed:
                        data.append(processed)
                return data
    except:
        pass
    
    # 尝试CSV格式（旧版DuEE）
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                sentence, relation, head = parts[0], parts[1], parts[2]
                head_offset = int(parts[3]) if parts[3].isdigit() else 0
                tail, tail_offset = parts[4], int(parts[5]) if parts[5].isdigit() else 0
                
                # 确定事件类型
                event_type_map = {
                    '出生地': 'SocialEvent', '出生日期': 'SocialEvent', '国籍': 'PoliticalEvent', 
                    '民族': 'SocialEvent', '祖籍': 'SocialEvent', '朝代': 'HistoricalEvent',
                    '身高': 'HealthEvent', '体重': 'HealthEvent', '妻子': 'SocialEvent',
                    '丈夫': 'SocialEvent', '父亲': 'SocialEvent', '母亲': 'SocialEvent',
                    '主演': 'CulturalEvent', '导演': 'CulturalEvent', '歌手': 'CulturalEvent',
                    '作曲': 'CulturalEvent', '作词': 'CulturalEvent', '作者': 'CulturalEvent',
                    '出版社': 'EconomicEvent', '连载网站': 'CulturalEvent', '出品公司': 'EconomicEvent',
                    '成立日期': 'EconomicEvent', '总部地点': 'EconomicEvent', '海拔': 'NaturalDisaster',
                    '面积': 'NaturalDisaster', '首都': 'PoliticalEvent', '官方语言': 'PoliticalEvent',
                    '人口数量': 'SocialEvent', '气候': 'NaturalDisaster', '毕业院校': 'SocialEvent',
                    '专业': 'SocialEvent'
                }
                event_type = event_type_map.get(relation, 'SocialEvent')
                
                # 查找触发词在文本中的位置
                trigger_word = head if relation in ['主演', '导演', '歌手', '作曲', '作词', '作者'] else tail
                trigger_start = sentence.find(trigger_word)
                trigger_end = trigger_start + len(trigger_word) if trigger_start != -1 else -1
                
                data.append({
                    'text': sentence,
                    'relation': relation,
                    'trigger_word': trigger_word,
                    'trigger_start': trigger_start,
                    'trigger_end': trigger_end,
                    'event_type': event_type,
                    'head': head,
                    'head_offset': head_offset,
                    'tail': tail,
                    'tail_offset': tail_offset,
                    'event_list': [{
                        'event_type': event_type,
                        'trigger': trigger_word,
                        'trigger_start': trigger_start,
                        'trigger_end': trigger_end,
                        'arguments': [{
                            'role': 'subject',
                            'text': head,
                            'start': head_offset,
                            'end': head_offset + len(head)
                        }, {
                            'role': 'object',
                            'text': tail,
                            'start': tail_offset,
                            'end': tail_offset + len(tail)
                        }]
                    }]
                })
    except Exception as e:
        print(f"加载DuEE数据失败: {e}")
    
    return data


def process_duee_json(item):
    """处理DuEE JSON格式数据"""
    try:
        text = item.get('text', '')
        event_list = item.get('event_list', [])
        
        processed_events = []
        for event in event_list:
            event_type = event.get('event_type', '')
            trigger = event.get('trigger', '')
            trigger_start = text.find(trigger) if trigger else -1
            trigger_end = trigger_start + len(trigger) if trigger_start != -1 else -1
            
            arguments = []
            for arg in event.get('arguments', []):
                arg_text = arg.get('argument', '')
                arg_start = arg.get('start', -1)
                arg_end = arg.get('end', -1)
                if arg_start == -1 and arg_text:
                    arg_start = text.find(arg_text)
                    arg_end = arg_start + len(arg_text) if arg_start != -1 else -1
                arguments.append({
                    'role': arg.get('role', ''),
                    'text': arg_text,
                    'start': arg_start,
                    'end': arg_end
                })
            
            processed_events.append({
                'event_type': event_type,
                'trigger': trigger,
                'trigger_start': trigger_start,
                'trigger_end': trigger_end,
                'arguments': arguments
            })
        
        if not processed_events:
            return None
        
        return {
            'text': text,
            'event_list': processed_events
        }
    except Exception as e:
        print(f"处理DuEE JSON数据失败: {e}")
        return None

# ========================== 地理词典构建 ==========================

def build_geo_lexicon():
    """从GeoNames和城市知识图谱构建地理词典"""
    lexicon = set()
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    # 从CN.txt (GeoNames) 提取地名
    cn_path = os.path.join(DATA_DIR, 'CN.txt')
    if os.path.exists(cn_path):
        print(f"正在从 {cn_path} 提取地名...")
        # 尝试多种编码读取，处理非UTF-8文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        content = None
        for enc in encodings:
            try:
                with open(cn_path, 'r', encoding=enc) as f:
                    content = f.read()
                    break
            except:
                continue
        
        if content is not None:
            for line in content.split('\n'):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    name = parts[1].strip()
                    if chinese_pattern.search(name) and 1 < len(name) <= 20:
                        lexicon.add(name)
                if len(parts) >= 3:
                    asciiname = parts[2].strip()
                    if chinese_pattern.search(asciiname) and 1 < len(asciiname) <= 20:
                        lexicon.add(asciiname)
                if len(parts) >= 4:
                    altname = parts[3].strip()
                    if altname:
                        matches = chinese_pattern.findall(altname)
                        for match in matches:
                            if 1 < len(match) <= 20:
                                lexicon.add(match)
        else:
            print(f"警告：无法读取文件 {cn_path}")
    
    # 从城市知识图谱提取地名
    city_json_path = os.path.join(DATA_DIR, 'China_main_city_information', 'graph', 'city.json')
    if os.path.exists(city_json_path):
        print(f"正在从 {city_json_path} 提取地名...")
        with open(city_json_path, 'r', encoding='utf-8-sig') as f:
            city_data = json.load(f)
            for item in city_data:
                if 'n' in item and 'properties' in item['n']:
                    props = item['n']['properties']
                    # 城市名称
                    if 'name' in props:
                        name = props['name']
                        if chinese_pattern.search(name) and 1 < len(name) <= 20:
                            lexicon.add(name)
                    # 别名
                    if 'anothername' in props:
                        anothernames = props['anothername']
                        if anothernames and anothernames != '暂无数据':
                            # 处理多个别名
                            aliases = anothernames.replace('、', '，').replace('、', ',').split('，')
                            for alias in aliases:
                                alias = alias.strip()
                                if chinese_pattern.search(alias) and 1 < len(alias) <= 20:
                                    lexicon.add(alias)
                    # 省份
                    if 'province' in props:
                        province = props['province']
                        if chinese_pattern.search(province) and 1 < len(province) <= 20:
                            lexicon.add(province)
    
    # 从现有的geo_lexicon.txt加载
    existing_lexicon_path = os.path.join(DATA_DIR, 'geo_lexicon.txt')
    if os.path.exists(existing_lexicon_path):
        print(f"正在从 {existing_lexicon_path} 加载现有词典...")
        with open(existing_lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and 1 < len(word) <= 20:
                    lexicon.add(word)
    
    return lexicon

def save_lexicon_and_vocab(lexicon):
    """保存词典和词汇映射"""
    lexicon_path = os.path.join(PROCESSED_DIR, 'geo_lexicon.txt')
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        for word in sorted(lexicon):
            f.write(word + '\n')
    print(f"地理词典已保存到: {lexicon_path}，共 {len(lexicon)} 个词汇")
    
    lexicon_list = sorted(lexicon)
    word_to_id = {word: idx + 1 for idx, word in enumerate(lexicon_list)}
    id_to_word = {idx + 1: word for idx, word in enumerate(lexicon_list)}
    
    vocab_path = os.path.join(PROCESSED_DIR, 'lexicon_vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'size': len(lexicon) + 1
        }, f, ensure_ascii=False, indent=2)
    print(f"词汇映射已保存到: {vocab_path}")
    
    return lexicon_list


# ========================== 地理层级、别名、类型文件生成 ==========================

def extract_geo_hierarchy():
    """
    从GeoNames和城市知识图谱提取地理层级信息
    
    层级编码方案（共7级，含0占位）：
    - 0: padding/未知
    - 1: 省级（省、自治区、直辖市、特别行政区）
    - 2: 市级（地级市、自治州、盟）
    - 3: 区级（市辖区、县级市、县、自治县）
    - 4: 乡级（乡、镇、街道）
    - 5: 村级（村、居委会）
    - 6: 其他/地标
    
    Returns:
        hierarchy_dict: dict - 地名到层级ID的映射
    """
    hierarchy_dict = {}
    
    # 从城市知识图谱提取层级信息
    city_json_path = os.path.join(DATA_DIR, 'China_main_city_information', 'graph', 'city.json')
    if os.path.exists(city_json_path):
        print(f"正在从 {city_json_path} 提取层级信息...")
        try:
            with open(city_json_path, 'r', encoding='utf-8-sig') as f:
                city_data = json.load(f)
                for item in city_data:
                    if 'n' in item and 'properties' in item['n']:
                        props = item['n']['properties']
                        name = props.get('name', '')
                        if name:
                            # 根据行政级别字段确定层级
                            level = props.get('level', '')
                            if level in ['province', '省', '自治区', '直辖市', '特别行政区']:
                                hierarchy_dict[name] = 1
                            elif level in ['city', '地级市', '自治州', '盟']:
                                hierarchy_dict[name] = 2
                            elif level in ['district', '区', '县', '县级市', '自治县']:
                                hierarchy_dict[name] = 3
                            elif level in ['town', '镇', '乡', '街道']:
                                hierarchy_dict[name] = 4
                            elif level in ['village', '村', '居委会']:
                                hierarchy_dict[name] = 5
                            
                            # 添加别名
                            if 'anothername' in props and props['anothername'] and props['anothername'] != '暂无数据':
                                aliases = props['anothername'].replace('、', '，').replace(';', '，').split('，')
                                for alias in aliases:
                                    alias = alias.strip()
                                    if alias and alias not in hierarchy_dict:
                                        hierarchy_dict[alias] = hierarchy_dict[name]
        except Exception as e:
            print(f"警告：读取城市知识图谱失败: {e}")
    else:
        print(f"警告：城市知识图谱文件不存在: {city_json_path}")
    
    # 从GeoNames CN.txt提取层级信息
    cn_path = os.path.join(DATA_DIR, 'CN.txt')
    if os.path.exists(cn_path):
        print(f"正在从 {cn_path} 提取层级信息...")
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        content = None
        for enc in encodings:
            try:
                with open(cn_path, 'r', encoding=enc) as f:
                    content = f.read()
                    break
            except:
                continue
        
        if content is not None:
            for line in content.split('\n'):
                parts = line.strip().split('\t')
                if len(parts) >= 18:
                    name = parts[1].strip()
                    feature_class = parts[7].strip()
                    feature_code = parts[8].strip()
                    
                    if name and name not in hierarchy_dict:
                        # 根据feature code推断层级
                        if feature_code in ['ADM1', 'PRI']:
                            hierarchy_dict[name] = 1
                        elif feature_code in ['ADM2', 'PPLA', 'PPLA2', 'PPLA3']:
                            hierarchy_dict[name] = 2
                        elif feature_code in ['ADM3', 'PPLA4']:
                            hierarchy_dict[name] = 3
                        elif feature_code in ['ADM4']:
                            hierarchy_dict[name] = 4
                        elif feature_code in ['ADM5']:
                            hierarchy_dict[name] = 5
                        elif feature_class in ['P', 'S']:
                            hierarchy_dict[name] = 3
                        else:
                            hierarchy_dict[name] = 6
        else:
            print(f"警告：无法读取CN.txt文件")
    else:
        print(f"警告：GeoNames CN.txt文件不存在: {cn_path}")
    
    # 手动添加常见地标（层级6）
    landmarks = ['天安门', '故宫', '长城', '颐和园', '天坛', '兵马俑', '西湖', 
                 '黄山', '泰山', '华山', '九寨沟', '张家界', '桂林山水', '兵马俑',
                 '布达拉宫', '兵马俑', '大雁塔', '小雁塔', '少林寺', '寒山寺']
    for landmark in landmarks:
        if landmark not in hierarchy_dict:
            hierarchy_dict[landmark] = 6
    
    print(f"共提取到 {len(hierarchy_dict)} 个地名的层级信息")
    return hierarchy_dict


def extract_geo_aliases(hierarchy_dict):
    """
    生成地名别名映射
    
    Args:
        hierarchy_dict: dict - 地名到层级的映射
    
    Returns:
        alias_map: dict - 别名到标准名称的映射
    """
    alias_map = {}
    
    # 读取手工整理的别名表
    manual_alias_path = os.path.join(DATA_DIR, 'geo_aliases_manual.txt')
    if os.path.exists(manual_alias_path):
        print(f"正在从 {manual_alias_path} 加载手工别名表...")
        try:
            with open(manual_alias_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            alias = parts[0].strip()
                            standard_name = parts[1].strip()
                            if standard_name in hierarchy_dict:
                                alias_map[alias] = standard_name
        except Exception as e:
            print(f"警告：读取手工别名表失败: {e}")
    else:
        print(f"警告：手工别名表不存在: {manual_alias_path}，将自动生成别名")
    
    # 自动生成别名（基于常见简称）
    print("正在自动生成别名...")
    province_abbrevs = {
        '北京市': ['北京', '京'],
        '上海市': ['上海', '沪'],
        '天津市': ['天津', '津'],
        '重庆市': ['重庆', '渝'],
        '河北省': ['河北', '冀'],
        '山西省': ['山西', '晋'],
        '辽宁省': ['辽宁', '辽'],
        '吉林省': ['吉林', '吉'],
        '黑龙江省': ['黑龙江', '黑'],
        '江苏省': ['江苏', '苏'],
        '浙江省': ['浙江', '浙'],
        '安徽省': ['安徽', '皖'],
        '福建省': ['福建', '闽'],
        '江西省': ['江西', '赣'],
        '山东省': ['山东', '鲁'],
        '河南省': ['河南', '豫'],
        '湖北省': ['湖北', '鄂'],
        '湖南省': ['湖南', '湘'],
        '广东省': ['广东', '粤'],
        '海南省': ['海南', '琼'],
        '四川省': ['四川', '川', '蜀'],
        '贵州省': ['贵州', '贵', '黔'],
        '云南省': ['云南', '云', '滇'],
        '陕西省': ['陕西', '陕', '秦'],
        '甘肃省': ['甘肃', '甘', '陇'],
        '青海省': ['青海', '青'],
        '台湾省': ['台湾', '台'],
        '内蒙古自治区': ['内蒙古'],
        '广西壮族自治区': ['广西'],
        '西藏自治区': ['西藏'],
        '宁夏回族自治区': ['宁夏'],
        '新疆维吾尔自治区': ['新疆'],
        '香港特别行政区': ['香港', '港'],
        '澳门特别行政区': ['澳门', '澳']
    }
    
    for standard_name, aliases in province_abbrevs.items():
        if standard_name in hierarchy_dict:
            for alias in aliases:
                if alias not in alias_map and alias != standard_name:
                    alias_map[alias] = standard_name
    
    # 添加城市常见别名
    city_aliases = {
        '广州市': ['广州', '穗'],
        '深圳市': ['深圳', '鹏城'],
        '成都市': ['成都', '蓉城'],
        '杭州市': ['杭州', '杭'],
        '南京市': ['南京', '宁'],
        '武汉市': ['武汉', '江城'],
        '西安市': ['西安', '长安'],
        '苏州市': ['苏州', '苏'],
        '郑州市': ['郑州', '商都'],
        '长沙市': ['长沙', '星城'],
        '青岛市': ['青岛', '岛城'],
        '沈阳市': ['沈阳', '盛京'],
        '大连市': ['大连', '滨城'],
        '厦门市': ['厦门', '鹭岛'],
        '宁波市': ['宁波', '甬'],
        '合肥市': ['合肥', '庐州'],
        '佛山市': ['佛山', '禅'],
        '东莞市': ['东莞'],
        '无锡市': ['无锡'],
        '济南市': ['济南', '泉城'],
        '哈尔滨市': ['哈尔滨', '冰城'],
        '长春市': ['长春', '春城'],
        '石家庄市': ['石家庄'],
        '南宁市': ['南宁', '邕城'],
        '南昌市': ['南昌', '洪城'],
        '福州市': ['福州', '榕城'],
        '太原市': ['太原', '龙城'],
        '贵阳市': ['贵阳', '筑城'],
        '昆明市': ['昆明', '春城'],
        '烟台市': ['烟台'],
        '常州市': ['常州'],
        '南通市': ['南通'],
        '泉州市': ['泉州'],
        '绍兴市': ['绍兴'],
        '嘉兴市': ['嘉兴'],
        '徐州市': ['徐州'],
        '温州市': ['温州'],
        '金华市': ['金华'],
        '惠州市': ['惠州'],
        '珠海市': ['珠海'],
        '中山市': ['中山'],
        '江门市': ['江门'],
        '湛江市': ['湛江'],
        '汕头市': ['汕头'],
        '揭阳市': ['揭阳'],
        '潮州市': ['潮州'],
        '肇庆市': ['肇庆'],
        '清远市': ['清远'],
        '韶关市': ['韶关'],
        '梅州市': ['梅州'],
        '河源市': ['河源'],
        '汕尾市': ['汕尾'],
        '阳江市': ['阳江'],
        '茂名市': ['茂名'],
        '云浮市': ['云浮'],
        '潮州市': ['潮州']
    }
    
    for standard_name, aliases in city_aliases.items():
        if standard_name in hierarchy_dict:
            for alias in aliases:
                if alias not in alias_map and alias != standard_name:
                    alias_map[alias] = standard_name
    
    print(f"共生成 {len(alias_map)} 个别名映射")
    return alias_map


def extract_geo_types(hierarchy_dict):
    """
    生成地名类型映射
    
    类型编码方案：
    - 0: padding/未知
    - 1: 行政区划（省、市、区、县等）
    - 2: 自然地理（山脉、河流、湖泊等）
    - 3: 人工地标（建筑、景点等）
    - 4: 道路/交通设施
    - 5: 区域/边界
    
    Args:
        hierarchy_dict: dict - 地名到层级的映射
    
    Returns:
        type_map: dict - 地名到类型ID的映射
    """
    type_map = {}
    
    # 从GeoNames提取类型信息
    cn_path = os.path.join(DATA_DIR, 'CN.txt')
    if os.path.exists(cn_path):
        print(f"正在从 {cn_path} 提取类型信息...")
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        content = None
        for enc in encodings:
            try:
                with open(cn_path, 'r', encoding=enc) as f:
                    content = f.read()
                    break
            except:
                continue
        
        if content is not None:
            for line in content.split('\n'):
                parts = line.strip().split('\t')
                if len(parts) >= 18:
                    name = parts[1].strip()
                    feature_class = parts[7].strip()
                    feature_code = parts[8].strip()
                    
                    if name and name in hierarchy_dict and name not in type_map:
                        # 根据feature class/code确定类型
                        if feature_class == 'A' or feature_code.startswith('ADM'):
                            type_map[name] = 1
                        elif feature_class in ['T', 'H', 'V', 'W', 'L']:
                            type_map[name] = 2
                        elif feature_class in ['S', 'P']:
                            type_map[name] = 3
                        elif feature_code in ['RD', 'HW', 'BR', 'RA']:
                            type_map[name] = 4
                        else:
                            type_map[name] = 1
    else:
        print(f"警告：GeoNames CN.txt文件不存在: {cn_path}")
    
    # 从城市知识图谱提取类型
    city_json_path = os.path.join(DATA_DIR, 'China_main_city_information', 'graph', 'city.json')
    if os.path.exists(city_json_path):
        print(f"正在从 {city_json_path} 提取类型信息...")
        try:
            with open(city_json_path, 'r', encoding='utf-8-sig') as f:
                city_data = json.load(f)
                for item in city_data:
                    if 'n' in item and 'properties' in item['n']:
                        props = item['n']['properties']
                        name = props.get('name', '')
                        if name and name in hierarchy_dict and name not in type_map:
                            level = props.get('level', '')
                            if level in ['province', 'city', 'district', 'town', 'village']:
                                type_map[name] = 1
                            else:
                                type_map[name] = 3
        except Exception as e:
            print(f"警告：读取城市知识图谱失败: {e}")
    
    # 默认类型为行政区划
    for name in hierarchy_dict:
        if name not in type_map:
            type_map[name] = 1
    
    print(f"共生成 {len(type_map)} 个类型映射")
    return type_map


def save_geo_hierarchy_files(hierarchy_dict, alias_map, type_map):
    """
    保存地理层级、别名、类型文件
    
    Args:
        hierarchy_dict: dict - 地名到层级ID的映射
        alias_map: dict - 别名到标准名称的映射
        type_map: dict - 地名到类型ID的映射
    """
    # 保存层级文件 (格式: 地名\t层级编号)
    hierarchy_path = os.path.join(PROCESSED_DIR, 'geo_lexicon_hierarchy.txt')
    with open(hierarchy_path, 'w', encoding='utf-8') as f:
        for name, level in sorted(hierarchy_dict.items()):
            f.write(f"{name}\t{level}\n")
    print(f"地理层级文件已保存到: {hierarchy_path}")
    
    # 保存别名映射文件
    alias_path = os.path.join(PROCESSED_DIR, 'geo_alias_map.json')
    with open(alias_path, 'w', encoding='utf-8') as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)
    print(f"别名映射文件已保存到: {alias_path}")
    
    # 保存类型映射文件
    type_path = os.path.join(PROCESSED_DIR, 'geo_type_map.json')
    with open(type_path, 'w', encoding='utf-8') as f:
        json.dump(type_map, f, ensure_ascii=False, indent=2)
    print(f"类型映射文件已保存到: {type_path}")


def build_geo_knowledge_files():
    """
    构建地理知识文件（层级、别名、类型）
    """
    print("\n" + "=" * 60)
    print("构建地理知识文件")
    print("=" * 60)
    
    # 先构建词典获取地名列表
    lexicon = build_geo_lexicon()
    
    # 提取层级信息
    print("\n1. 提取地理层级信息...")
    hierarchy_dict = extract_geo_hierarchy()
    
    # 提取别名映射
    print("\n2. 提取地名别名...")
    alias_map = extract_geo_aliases(hierarchy_dict)
    
    # 提取类型信息
    print("\n3. 提取地名类型...")
    type_map = extract_geo_types(hierarchy_dict)
    
    # 保存文件
    print("\n4. 保存地理知识文件...")
    save_geo_hierarchy_files(hierarchy_dict, alias_map, type_map)
    
    print("\n地理知识文件构建完成！")

# ========================== 主处理函数 ==========================

def process_ner_datasets():
    """处理所有NER数据集"""
    datasets = {
        'cluener': {
            'train': os.path.join(DATA_DIR, 'cluener_public', 'train.json'),
            'dev': os.path.join(DATA_DIR, 'cluener_public', 'dev.json'),
            'test': os.path.join(DATA_DIR, 'cluener_public', 'test.json'),
            'loader': load_cluener_raw,
            'converter': cluener_to_bio
        },
        'msra': {
            'train': os.path.join(DATA_DIR, 'MSRA', 'train.txt'),
            'dev': os.path.join(DATA_DIR, 'MSRA', 'train.txt'),  # MSRA没有单独的dev，用train的一部分
            'test': os.path.join(DATA_DIR, 'MSRA', 'test.txt'),
            'loader': load_msra_raw,
            'converter': lambda x: x  # 已经是BIO格式
        },
        'weibo_ner': {
            'train': os.path.join(DATA_DIR, 'Weibo NER', 'train.txt'),
            'dev': os.path.join(DATA_DIR, 'Weibo NER', 'dev.txt'),
            'test': os.path.join(DATA_DIR, 'Weibo NER', 'test.txt'),
            'loader': load_weibo_ner_raw,
            'converter': lambda x: x  # 已经是BIO格式
        },
        'cmner': {
            'train': os.path.join(DATA_DIR, 'CMNER', 'text', 'train.txt'),
            'dev': os.path.join(DATA_DIR, 'CMNER', 'text', 'dev.txt'),
            'test': os.path.join(DATA_DIR, 'CMNER', 'text', 'test.txt'),
            'loader': load_cmner_raw,
            'converter': lambda x: x  # 已经是BIO格式
        }
    }
    
    for dataset_name, paths in datasets.items():
        print(f"\n处理 {dataset_name} 数据集...")
        
        for split in ['train', 'dev', 'test']:
            file_path = paths[split]
            if not os.path.exists(file_path):
                print(f"  {split} 文件不存在: {file_path}")
                continue
            
            # 加载数据
            data = paths['loader'](file_path)
            
            # 转换格式
            data = paths['converter'](data)
            
            # 保存处理后的数据
            save_path = os.path.join(PROCESSED_DIR, dataset_name, f'{split}.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"  {split}: {len(data)} 条数据 -> {save_path}")

def process_relation_datasets():
    """处理关系抽取数据集"""
    # DuIE
    print("\n处理 DuIE 数据集...")
    duie_paths = {
        'train': os.path.join(DATA_DIR, 'DuIE2.0', 'duie_train.json', 'duie_train.json'),
        'dev': os.path.join(DATA_DIR, 'DuIE2.0', 'duie_dev.json', 'duie_dev.json'),
        'test': os.path.join(DATA_DIR, 'DuIE2.0', 'duie_test2.json', 'duie_test2.json')
    }
    
    for split, file_path in duie_paths.items():
        if not os.path.exists(file_path):
            print(f"  DuIE {split} 文件不存在: {file_path}")
            continue
        
        data = load_duie_raw(file_path)
        
        save_path = os.path.join(PROCESSED_DIR, 'duie', f'{split}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  DuIE {split}: {len(data)} 条数据 -> {save_path}")
    
    # DuEE
    print("\n处理 DuEE 数据集...")
    duee_paths = {
        'train': os.path.join(DATA_DIR, 'DuEE1.0', 'train.csv'),
        'dev': os.path.join(DATA_DIR, 'DuEE1.0', 'dev.csv')
    }
    
    for split, file_path in duee_paths.items():
        if not os.path.exists(file_path):
            print(f"  DuEE {split} 文件不存在: {file_path}")
            continue
        
        data = load_duee_raw(file_path)
        
        save_path = os.path.join(PROCESSED_DIR, 'duee', f'{split}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  DuEE {split}: {len(data)} 条数据 -> {save_path}")

def main():
    """主函数"""
    print("=" * 80)
    print("数据预处理脚本")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"输出目录: {PROCESSED_DIR}")
    
    # 1. 处理NER数据集
    print("\n" + "=" * 60)
    print("步骤1: 处理NER数据集（CLUENER、MSRA、Weibo NER、CMNER）")
    print("=" * 60)
    process_ner_datasets()
    
    # 2. 处理关系/事件数据集
    print("\n" + "=" * 60)
    print("步骤2: 处理关系和事件数据集（DuIE、DuEE）")
    print("=" * 60)
    process_relation_datasets()
    
    # 3. 构建地理词典
    print("\n" + "=" * 60)
    print("步骤3: 构建地理词典")
    print("=" * 60)
    lexicon = build_geo_lexicon()
    save_lexicon_and_vocab(lexicon)
    
    # 4. 构建地理知识文件（层级、别名、类型）
    build_geo_knowledge_files()
    
    print("\n" + "=" * 80)
    print("数据预处理完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
