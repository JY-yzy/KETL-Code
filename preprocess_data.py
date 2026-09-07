#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
地理知识数据预处理脚本
独立于主程序运行，一次性处理所有原始数据并保存缓存
主程序 ner_experiment.py 直接从缓存加载，无需每次重新处理

使用方法：
    python preprocess_data.py              # 常规预处理（从缓存加载，不存在则处理）
    python preprocess_data.py --force      # 强制重新处理所有数据
    python preprocess_data.py --no-ownthink # 不加载ownthink_v2.csv（快速模式）

处理流程：
    1. 加载 geo_lexicon.txt（基础词典，GEO/ADMIN/LANDMARK）
    2. 加载 城市信息.csv（行政级别、别名，GEO/ADMIN）
    3. 加载 records.json（补充城市信息，GEO/ADMIN）
    4. 加载 CN.txt（自然地理实体+坐标，GEO）
    5. 加载 ownthink_v2.csv（仅补充 ORG/PER 实体，跳过已覆盖的 GEO/ADMIN）
    6. 生成别名映射
    7. 保存缓存到 data/processed/geo_knowledge_cache*.json

优化说明：
    ownthink_v2.csv（8GB）中的 GEO/ADMIN 实体与前4个数据源高度重复，
    因此仅从中提取 ORG（组织机构）和 PER（人物）实体，大幅减少数据量。
    预计缓存文件从 678MB 缩减至 50-100MB。
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 同步 ownthink 智能筛选参数到环境变量（必须在 import geo_knowledge 之前）
# 这样 geo_knowledge.py 中的 _load_ownthink_full 才能读取到 config_ner.py 的配置
from config_ner import cfg_ner as _cfg_pre
os.environ.setdefault('OWNTHINK_MAX_ORG', str(getattr(_cfg_pre, 'ownthink_max_org', 200000)))
os.environ.setdefault('OWNTHINK_MAX_PER', str(getattr(_cfg_pre, 'ownthink_max_per', 200000)))
os.environ.setdefault('OWNTHINK_MIN_LEN', str(getattr(_cfg_pre, 'ownthink_min_entity_len', 2)))
os.environ.setdefault('OWNTHINK_MAX_LEN', str(getattr(_cfg_pre, 'ownthink_max_entity_len', 15)))
del _cfg_pre

from geo_knowledge import GeoKnowledgeGraph

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def get_cache_filename(load_ownthink):
    """生成与 GeoKnowledgeGraph 内部一致的缓存文件名（含筛选参数指纹）"""
    if load_ownthink:
        _mo = os.environ.get('OWNTHINK_MAX_ORG', '200000')
        _mp = os.environ.get('OWNTHINK_MAX_PER', '200000')
        _mn = os.environ.get('OWNTHINK_MIN_LEN', '2')
        _mx = os.environ.get('OWNTHINK_MAX_LEN', '15')
        return f'geo_knowledge_cache_ownthink_org{_mo}_per{_mp}_min{_mn}_max{_mx}.json'
    return 'geo_knowledge_cache.json'


def validate_cache(cache_path, expected_min_entries=None):
    """验证缓存文件是否有效"""
    if not os.path.exists(cache_path):
        return False, "缓存文件不存在"
    
    try:
        with open(cache_path, 'r', encoding='utf-8-sig') as f:
            cached = json.load(f)
        
        required_keys = ['lexicon', 'alias_map', 'hierarchy_map', 'type_map', 'coord_map']
        for key in required_keys:
            if key not in cached:
                return False, f"缺少必需字段: {key}"
            if not cached[key]:
                return False, f"字段 {key} 为空"
        
        lexicon_size = len(cached['lexicon'])
        if expected_min_entries and lexicon_size < expected_min_entries:
            return False, f"词典条目不足: {lexicon_size} < {expected_min_entries}"
        
        print(f"  ✓ 缓存验证通过")
        print(f"    - 词典: {lexicon_size} 条")
        print(f"    - 别名: {len(cached['alias_map'])} 条")
        print(f"    - 层级: {len(cached['hierarchy_map'])} 条")
        print(f"    - 类型: {len(cached['type_map'])} 条")
        print(f"    - 坐标: {len(cached['coord_map'])} 条")
        
        return True, "验证通过"
        
    except Exception as e:
        return False, f"缓存加载失败: {type(e).__name__}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='地理知识数据预处理')
    parser.add_argument('--force', action='store_true', help='强制重新处理所有数据')
    parser.add_argument('--no-ownthink', action='store_true', help='不加载ownthink_v2.csv（快速模式）')
    parser.add_argument('--min-entries', type=int, default=10000, help='词典最小条目数验证阈值')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"地理知识数据预处理脚本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    load_ownthink = not args.no_ownthink
    cache_filename = get_cache_filename(load_ownthink)
    cache_path = os.path.join(DEFAULT_DATA_DIR, 'processed', cache_filename)
    
    # 显示筛选参数（仅 ownthink 模式）
    if load_ownthink:
        print(f"\n配置参数:")
        print(f"  - 数据目录: {DEFAULT_DATA_DIR}")
        print(f"  - 加载 ownthink: {load_ownthink}")
        print(f"  - 缓存文件: {cache_filename}")
        print(f"  - 强制重处理: {args.force}")
        print(f"  - 智能筛选参数:")
        print(f"    * ORG 配额: {os.environ.get('OWNTHINK_MAX_ORG', '200000')}")
        print(f"    * PER 配额: {os.environ.get('OWNTHINK_MAX_PER', '200000')}")
        print(f"    * 实体名长度: [{os.environ.get('OWNTHINK_MIN_LEN', '2')}, {os.environ.get('OWNTHINK_MAX_LEN', '15')}]")
    else:
        print(f"\n配置参数:")
        print(f"  - 数据目录: {DEFAULT_DATA_DIR}")
        print(f"  - 加载 ownthink: {load_ownthink}")
        print(f"  - 缓存文件: {cache_filename}")
        print(f"  - 强制重处理: {args.force}")
    
    if not args.force:
        print(f"\n验证现有缓存...")
        valid, msg = validate_cache(cache_path, args.min_entries)
        if valid:
            print(f"\n{'='*60}")
            print(f"缓存有效，跳过预处理")
            print(f"下次运行 ner_experiment.py 时将直接从缓存加载")
            print(f"如需重新处理，请使用: python preprocess_data.py --force")
            print(f"{'='*60}")
            return
        else:
            print(f"  ✗ 缓存无效: {msg}")
            print(f"    将重新处理数据...")
    
    print(f"\n{'='*60}")
    print(f"开始数据预处理...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        kg = GeoKnowledgeGraph(
            DEFAULT_DATA_DIR, 
            load_ownthink=load_ownthink, 
            skip_cache=True
        )
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n{'='*60}")
        print(f"预处理完成!")
        print(f"耗时: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"{'='*60}")
        print(f"数据统计:")
        print(f"  - 词典大小: {len(kg.lexicon):,}")
        print(f"  - 别名映射: {len(kg.alias_map):,}")
        print(f"  - 层级信息: {len(kg.hierarchy_map):,}")
        print(f"  - 类型信息: {len(kg.type_map):,}")
        print(f"  - 坐标信息: {len(kg.coord_map):,}")
        print(f"{'='*60}")
        
        print(f"\n验证生成的缓存...")
        valid, msg = validate_cache(cache_path, args.min_entries)
        if valid:
            print(f"\n{'='*60}")
            print(f"✓ 预处理成功!")
            print(f"缓存已保存至: {cache_path}")
            print(f"下次运行 ner_experiment.py 将直接从缓存加载")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"✗ 缓存验证失败: {msg}")
            print(f"请检查数据源文件是否完整")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ 预处理失败!")
        print(f"错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == '__main__':
    main()