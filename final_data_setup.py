import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join('data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
GEOGLUE_DIR = os.path.join(DATA_DIR, 'GeoGLUE')

GEOETA_TAG_MAP = {
    'prov': 'ADMIN',
    'city': 'ADMIN',
    'district': 'ADMIN',
    'town': 'ADMIN',
    'community': 'ADMIN',
    'devzone': 'ADMIN',
    'road': 'LANDMARK',
    'poi': 'LANDMARK',
    'subpoi': 'LANDMARK',
    'roadno': 'LANDMARK',
    'houseno': 'LANDMARK',
}

def convert_bies_to_bio(tokens, tags):
    bio_tags = []
    for tag in tags:
        if tag == 'O':
            bio_tags.append('O')
        else:
            prefix, entity_type = tag.split('-', 1)
            unified_type = GEOETA_TAG_MAP.get(entity_type, 'GEO')
            
            if prefix == 'B':
                bio_tags.append(f'B-{unified_type}')
            elif prefix == 'I':
                bio_tags.append(f'I-{unified_type}')
            elif prefix == 'E':
                bio_tags.append(f'I-{unified_type}')
            elif prefix == 'S':
                bio_tags.append(f'B-{unified_type}')
            else:
                bio_tags.append('O')
    return bio_tags

def extract_geoglue_samples():
    print("\n从GeoGLUE提取ADMIN/LANDMARK样本...")
    augmented_data = []
    geoeta_path = os.path.join(GEOGLUE_DIR, 'GeoETA', 'train.json')
    
    if os.path.exists(geoeta_path):
        with open(geoeta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                tokens = item.get('tokens', [])
                tags = item.get('ner_tags', [])
                
                if len(tokens) != len(tags):
                    continue
                
                bio_tags = convert_bies_to_bio(tokens, tags)
                text = ''.join(tokens)
                
                has_admin = any(t.startswith('B-ADMIN') for t in bio_tags)
                has_landmark = any(t.startswith('B-LANDMARK') for t in bio_tags)
                
                if has_admin or has_landmark:
                    augmented_data.append({
                        'text': text,
                        'ner_tags': bio_tags
                    })
    
    print(f"提取了 {len(augmented_data)} 条样本")
    return augmented_data

def rebuild_cluener_train(augmented_data):
    print("\n重建cluener训练集...")
    
    original_path = os.path.join('data', 'cluener_public', 'train.json')
    save_path = os.path.join(PROCESSED_DIR, 'cluener', 'train.json')
    
    if os.path.exists(original_path):
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = []
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item['text']
                    labels = item.get('label', {})
                    
                    bio_tags = ['O'] * len(text)
                    for entity_type, entities in labels.items():
                        if entity_type in ['address', 'location']:
                            unified_type = 'GEO'
                        elif entity_type in ['company', 'organization', 'game', 'movie', 'book']:
                            unified_type = 'ORG'
                        elif entity_type in ['person', 'name', 'position']:
                            unified_type = 'PER'
                        elif entity_type == 'government':
                            unified_type = 'ADMIN'
                        elif entity_type == 'scene':
                            unified_type = 'LANDMARK'
                        else:
                            continue
                        
                        for entity_name, spans in entities.items():
                            for start, end in spans:
                                if start < len(text) and end <= len(text) and start < end:
                                    bio_tags[start] = f'B-{unified_type}'
                                    for i in range(start + 1, end):
                                        if i < len(text):
                                            bio_tags[i] = f'I-{unified_type}'
                    original_data.append({'text': text, 'ner_tags': bio_tags})
    
    print(f"原始cluener训练集: {len(original_data)} 条")
    
    combined_data = original_data + augmented_data
    print(f"合并后: {len(combined_data)} 条")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"已保存到 {save_path}")

def remove_data_leakage():
    print("\n移除数据泄露...")
    
    datasets = ['weibo_ner', 'cmner']
    
    for dataset_name in datasets:
        dev_path = os.path.join(PROCESSED_DIR, dataset_name, 'dev.json')
        test_path = os.path.join(PROCESSED_DIR, dataset_name, 'test.json')
        
        if not os.path.exists(dev_path) or not os.path.exists(test_path):
            continue
        
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = [json.loads(line) for line in f if line.strip()]
        
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f if line.strip()]
        
        dev_texts = set(item['text'] for item in dev_data)
        filtered_test = [item for item in test_data if item['text'] not in dev_texts]
        
        leaked_count = len(test_data) - len(filtered_test)
        
        if leaked_count > 0:
            with open(test_path, 'w', encoding='utf-8') as f:
                for item in filtered_test:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        
        print(f"  {dataset_name}: 移除了 {leaked_count} 条泄露数据，测试集剩余 {len(filtered_test)} 条")

def expand_lexicon_with_entities():
    """从训练数据中提取ORG/PER实体，扩展词典覆盖范围"""
    print("\n扩展词典覆盖范围（提取ORG/PER实体）...")

    datasets = ['cluener', 'msra', 'weibo_ner', 'cmner']
    org_entities = set()
    per_entities = set()

    for dataset_name in datasets:
        train_path = os.path.join(PROCESSED_DIR, dataset_name, 'train.json')
        if not os.path.exists(train_path):
            continue

        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get('text', '')
                tags = item.get('ner_tags', [])

                if len(text) != len(tags):
                    continue

                # 提取实体文本
                i = 0
                while i < len(tags):
                    tag = tags[i]
                    if tag.startswith('B-'):
                        entity_type = tag[2:]
                        j = i + 1
                        while j < len(tags) and tags[j].startswith('I-') and tags[j][2:] == entity_type:
                            j += 1
                        entity_text = text[i:j]
                        if len(entity_text) >= 2:  # 过滤单字实体
                            if entity_type == 'ORG':
                                org_entities.add(entity_text)
                            elif entity_type == 'PER':
                                per_entities.add(entity_text)
                        i = j
                    else:
                        i += 1

    print(f"  提取ORG实体: {len(org_entities)} 个")
    print(f"  提取PER实体: {len(per_entities)} 个")

    # 读取现有词典
    lexicon_path = os.path.join(PROCESSED_DIR, 'geo_lexicon.txt')
    existing_words = set()
    if os.path.exists(lexicon_path):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            existing_words = set(line.strip() for line in f if line.strip())

    print(f"  现有词典: {len(existing_words)} 条")

    # 找出新增实体
    new_org = org_entities - existing_words
    new_per = per_entities - existing_words
    print(f"  新增ORG实体: {len(new_org)} 个")
    print(f"  新增PER实体: {len(new_per)} 个")

    # 追加到词典文件
    with open(lexicon_path, 'a', encoding='utf-8') as f:
        for word in sorted(new_org):
            f.write(word + '\n')
        for word in sorted(new_per):
            f.write(word + '\n')

    # 更新类型映射
    type_map_path = os.path.join(PROCESSED_DIR, 'geo_type_map.json')
    type_map = {}
    if os.path.exists(type_map_path):
        with open(type_map_path, 'r', encoding='utf-8') as f:
            type_map = json.load(f)

    # type_id: 1=GEO, 2=ORG, 3=PER
    for word in org_entities:
        type_map[word] = 2
    for word in per_entities:
        type_map[word] = 3

    with open(type_map_path, 'w', encoding='utf-8') as f:
        json.dump(type_map, f, ensure_ascii=False, indent=2)

    total = len(existing_words) + len(new_org) + len(new_per)
    print(f"  扩展后词典: {total} 条")
    print(f"  类型映射已更新: ORG→type_id=2, PER→type_id=3")

def rebuild_lexicon_vocab():
    print("\n重建词汇映射...")

    lexicon_path = os.path.join(PROCESSED_DIR, 'geo_lexicon.txt')
    vocab_path = os.path.join(PROCESSED_DIR, 'lexicon_vocab.json')

    if os.path.exists(lexicon_path):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]

        words = sorted(set(words))

        word_to_id = {word: idx + 1 for idx, word in enumerate(words)}
        id_to_word = {idx + 1: word for idx, word in enumerate(words)}

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word_to_id': word_to_id,
                'id_to_word': id_to_word,
                'size': len(words) + 1
            }, f, ensure_ascii=False, indent=2)

        print(f"  词汇数: {len(words)}")
        print(f"  已保存到 {vocab_path}")

        # 自动更新 config.py 中的 lexicon_size
        import re
        config_path = os.path.join('config.py')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_size = len(words) + 1000  # 留余量
            content = re.sub(r'self\.lexicon_size\s*=\s*\d+', f'self.lexicon_size = {new_size}', content)
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  已更新 config.py: lexicon_size = {new_size}")

def recalculate_weights():
    print("\n重新计算类别权重...")
    
    from data_utils import load_ner_datasets
    
    train_data, _, _ = load_ner_datasets()
    
    tag_counts = {}
    for item in train_data:
        tags = item.get('ner_tags', [])
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    non_o_counts = {k: v for k, v in tag_counts.items() if k != 'O'}
    total_non_o = sum(non_o_counts.values())
    
    label_map = {
        0: 'O',
        1: 'B-GEO',
        2: 'I-GEO',
        3: 'B-ADMIN',
        4: 'I-ADMIN',
        5: 'B-LANDMARK',
        6: 'I-LANDMARK',
        7: 'B-ORG',
        8: 'I-ORG',
        9: 'B-PER',
        10: 'I-PER',
    }
    
    weights = []
    for i in range(11):
        tag = label_map[i]
        if tag == 'O':
            weights.append(0.1)
        else:
            count = tag_counts.get(tag, 1)
            weight = total_non_o / (10 * count)
            weight = max(0.5, min(3.0, weight))
            weights.append(round(weight, 2))
    
    print(f"\n计算的类别权重:")
    for i in range(11):
        print(f"  {i}: {label_map[i]} -> {weights[i]}")
    
    config_path = os.path.join('config_ner.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    content = re.sub(r'self\.ner_class_weights = \[.*?\]', f'self.ner_class_weights = {weights}', content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n已更新 {config_path}")

def verify_data():
    print("\n验证数据质量...")
    
    from data_utils import load_ner_datasets
    
    train_data, dev_data, _ = load_ner_datasets()
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    
    tag_counts = {}
    entity_starts = []
    errors = 0
    
    for item in train_data:
        text = item.get('text', '')
        tags = item.get('ner_tags', [])
        
        if len(text) != len(tags):
            errors += 1
            continue
        
        tag_counts.update({t: tag_counts.get(t, 0) + 1 for t in tags})
        
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith('B-'):
                entity_starts.append(tag[2:])
                j = i + 1
                while j < len(tags) and tags[j].startswith('I-') and tags[j][2:] == tag[2:]:
                    j += 1
                i = j
            elif tag.startswith('I-'):
                errors += 1
                i += 1
            else:
                i += 1
    
    print(f"错误样本数: {errors}")
    print(f"实体总数: {len(entity_starts)}")
    
    print("\n标签分布:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

def main():
    print("=" * 80)
    print("最终数据设置")
    print("=" * 80)
    
    augmented_data = extract_geoglue_samples()
    
    rebuild_cluener_train(augmented_data)
    
    remove_data_leakage()

    expand_lexicon_with_entities()

    rebuild_lexicon_vocab()
    
    recalculate_weights()
    
    verify_data()
    
    print("\n" + "=" * 80)
    print("最终数据设置完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
