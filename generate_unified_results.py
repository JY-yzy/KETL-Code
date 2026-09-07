#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一结果汇总脚本
将分散在各文件中的实验结果合并到一个CSV文件，方便写论文时直接引用。

用法: python generate_unified_results.py

输出文件: output_ner/results/unified_results.csv
"""
import os
import csv
import re

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'output_ner', 'results')


def read_ner_results():
    """读取NER实验结果"""
    path = os.path.join(RESULT_DIR, 'ner_results.csv')
    results = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get('model_name', '')
                results[model] = {
                    'best_dev_f1': float(row.get('best_dev_f1', 0)),
                    'avg_micro_f1': float(row.get('avg_micro_f1', 0)),
                    'cluener_micro_f1': float(row.get('cluener_micro_f1', 0)),
                    'cluener_macro_f1': float(row.get('cluener_macro_f1', 0)),
                    'cmner_micro_f1': float(row.get('cmner_micro_f1', 0)),
                    'cmner_macro_f1': float(row.get('cmner_macro_f1', 0)),
                    'msra_micro_f1': float(row.get('msra_micro_f1', 0)),
                    'msra_macro_f1': float(row.get('msra_macro_f1', 0)),
                    'weibo_ner_micro_f1': float(row.get('weibo_ner_micro_f1', 0)),
                    'weibo_ner_macro_f1': float(row.get('weibo_ner_macro_f1', 0)),
                }
    return results


def read_geoglue_results():
    """读取每个模型的GeoGLUE结果"""
    results = {}
    for fname in os.listdir(RESULT_DIR):
        if not fname.startswith('geoglue_') or not fname.endswith('.csv'):
            continue
        # 跳过汇总文件（非单个模型的结果）
        if fname == 'geoglue_results.csv':
            continue
        # 从文件名提取模型名
        model_name = fname[len('geoglue_'):-len('.csv')].replace('_', '+', 1)
        # 修复模型名: RoBERTaBiLSTMAttentionCRF+Lexicon 等
        if '+' in model_name:
            base, suffix = model_name.split('+', 1)
            model_name = f"{base}+{suffix}"
        else:
            model_name = model_name

        path = os.path.join(RESULT_DIR, fname)
        metrics = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get('Task', '')
                metric = row.get('Metric', '')
                value = row.get('Value', '0')
                try:
                    value = float(value)
                except ValueError:
                    continue
                key = f"{task}_{metric}"
                metrics[key] = value
        results[model_name] = metrics
    return results


def read_longtail_results():
    """读取每个模型的长尾实体识别结果"""
    results = {}
    for fname in os.listdir(RESULT_DIR):
        if not fname.startswith('longtail_') or not fname.endswith('.csv'):
            continue
        # 从文件名提取模型名
        model_name = fname[len('longtail_'):-len('.csv')].replace('_', '+', 1)
        if '+' in model_name:
            base, suffix = model_name.split('+', 1)
            model_name = f"{base}+{suffix}"

        path = os.path.join(RESULT_DIR, fname)
        regex_results = {}
        freq_results = {}
        current_section = None

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if 'Regex-based' in line:
                    current_section = 'regex'
                    continue
                elif 'Frequency-based' in line:
                    current_section = 'freq'
                    continue
                elif line.startswith('---') or line.startswith('===') or not line:
                    continue

                parts = line.split(',')
                if current_section == 'regex' and len(parts) >= 4:
                    category = parts[0]
                    try:
                        total = int(parts[1])
                        correct = int(parts[2])
                        precision = float(parts[3])
                        regex_results[category] = {'total': total, 'correct': correct, 'precision': precision}
                    except (ValueError, IndexError):
                        pass
                elif current_section == 'freq' and len(parts) >= 5:
                    level = parts[0]
                    try:
                        total = int(parts[1])
                        evaluated = int(parts[2])
                        correct = int(parts[3])
                        precision = float(parts[4])
                        freq_results[level] = {'total': total, 'evaluated': evaluated, 'correct': correct, 'precision': precision}
                    except (ValueError, IndexError):
                        pass

        results[model_name] = {
            'regex': regex_results,
            'freq': freq_results
        }
    return results


def generate_unified_csv():
    """生成统一结果CSV"""
    ner_results = read_ner_results()
    geoglue_results = read_geoglue_results()
    longtail_results = read_longtail_results()

    # 收集所有模型名
    all_models = set()
    all_models.update(ner_results.keys())
    all_models.update(geoglue_results.keys())
    all_models.update(longtail_results.keys())

    # 按逻辑顺序排列模型
    model_order = [
        'RoBERTaBiLSTMAttentionCRF',
        'RoBERTaBiLSTMAttentionCRF+Lexicon',
        'RoBERTaBiLSTMAttentionCRF+DynamicLexicon',
        'RoBERTaBiGRUAttentionCRF',
        'RoBERTaBiGRUAttentionCRF+Lexicon',
        'RoBERTaBiGRUAttentionCRF+DynamicLexicon',
    ]
    # 添加不在预设顺序中的模型
    for m in sorted(all_models):
        if m not in model_order:
            model_order.append(m)

    # 过滤掉不存在的模型
    model_order = [m for m in model_order if m in all_models]

    # 定义CSV列
    fieldnames = [
        'Model',
        # NER结果
        'NER_Best_Dev_F1',
        'NER_Avg_Micro_F1',
        'CLUENER_Micro_F1', 'CLUENER_Macro_F1',
        'CMNER_Micro_F1', 'CMNER_Macro_F1',
        'MSRA_Micro_F1', 'MSRA_Macro_F1',
        'Weibo_Micro_F1', 'Weibo_Macro_F1',
        # GeoGLUE结果（仅NER子任务：GeoETA / GeoCPA / GeoWWC，百分制）
        'GeoETA_Micro_F1', 'GeoETA_Macro_F1',
        'GeoCPA_Micro_F1', 'GeoCPA_Macro_F1',
        'GeoWWC_Micro_F1', 'GeoWWC_Accuracy',
        'GeoGLUE_Overall_Score',
        # 长尾-正则分类
        'LT_County_Level_Prec', 'LT_Historical_Prec', 'LT_Natural_Geo_Prec',
        'LT_Administrative_Prec', 'LT_Town_Village_Prec', 'LT_POI_Prec',
        'LT_Regex_Avg_Prec',
        # 长尾-频率分组
        'LT_High_Freq_Prec', 'LT_Medium_Freq_Prec', 'LT_Low_Freq_Prec',
    ]

    output_path = os.path.join(RESULT_DIR, 'unified_results.csv')
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in model_order:
            row = {'Model': model}

            # NER结果
            ner = ner_results.get(model, {})
            row['NER_Best_Dev_F1'] = f"{ner.get('best_dev_f1', 0):.4f}" if ner else ''
            row['NER_Avg_Micro_F1'] = f"{ner.get('avg_micro_f1', 0):.4f}" if ner else ''
            row['CLUENER_Micro_F1'] = f"{ner.get('cluener_micro_f1', 0):.4f}" if ner else ''
            row['CLUENER_Macro_F1'] = f"{ner.get('cluener_macro_f1', 0):.4f}" if ner else ''
            row['CMNER_Micro_F1'] = f"{ner.get('cmner_micro_f1', 0):.4f}" if ner else ''
            row['CMNER_Macro_F1'] = f"{ner.get('cmner_macro_f1', 0):.4f}" if ner else ''
            row['MSRA_Micro_F1'] = f"{ner.get('msra_micro_f1', 0):.4f}" if ner else ''
            row['MSRA_Macro_F1'] = f"{ner.get('msra_macro_f1', 0):.4f}" if ner else ''
            row['Weibo_Micro_F1'] = f"{ner.get('weibo_ner_micro_f1', 0):.4f}" if ner else ''
            row['Weibo_Macro_F1'] = f"{ner.get('weibo_ner_macro_f1', 0):.4f}" if ner else ''

            # GeoGLUE结果（仅NER子任务：GeoETA / GeoCPA / GeoWWC）
            # 注意：CSV中保存的是百分制值（0-100），直接显示
            geo = geoglue_results.get(model, {})
            row['GeoETA_Micro_F1'] = f"{geo.get('GeoETA_micro_f1', 0):.2f}" if geo else ''
            row['GeoETA_Macro_F1'] = f"{geo.get('GeoETA_macro_f1', 0):.2f}" if geo else ''
            row['GeoCPA_Micro_F1'] = f"{geo.get('GeoCPA_micro_f1', 0):.2f}" if geo else ''
            row['GeoCPA_Macro_F1'] = f"{geo.get('GeoCPA_macro_f1', 0):.2f}" if geo else ''
            row['GeoWWC_Micro_F1'] = f"{geo.get('GeoWWC_micro_f1', 0):.2f}" if geo else ''
            row['GeoWWC_Accuracy'] = f"{geo.get('GeoWWC_accuracy', 0):.2f}" if geo else ''
            row['GeoGLUE_Overall_Score'] = f"{geo.get('_summary_overall_score', geo.get('_summary_avg_micro_f1', 0)):.2f}" if geo else ''

            # 长尾结果
            lt = longtail_results.get(model, {})
            regex = lt.get('regex', {})
            freq = lt.get('freq', {})

            row['LT_County_Level_Prec'] = f"{regex.get('County_Level', {}).get('precision', 0):.2f}" if regex else ''
            row['LT_Historical_Prec'] = f"{regex.get('Historical', {}).get('precision', 0):.2f}" if regex else ''
            row['LT_Natural_Geo_Prec'] = f"{regex.get('Natural_Geo', {}).get('precision', 0):.2f}" if regex else ''
            row['LT_Administrative_Prec'] = f"{regex.get('Administrative', {}).get('precision', 0):.2f}" if regex else ''
            row['LT_Town_Village_Prec'] = f"{regex.get('Town_Village', {}).get('precision', 0):.2f}" if regex else ''
            row['LT_POI_Prec'] = f"{regex.get('POI', {}).get('precision', 0):.2f}" if regex else ''

            # 计算正则分类平均准确率
            if regex:
                regex_precisions = [v['precision'] for v in regex.values()]
                row['LT_Regex_Avg_Prec'] = f"{sum(regex_precisions) / len(regex_precisions):.2f}"
            else:
                row['LT_Regex_Avg_Prec'] = ''

            row['LT_High_Freq_Prec'] = f"{freq.get('High_Freq_>50', {}).get('precision', 0):.2f}" if freq else ''
            row['LT_Medium_Freq_Prec'] = f"{freq.get('Medium_Freq_10-50', {}).get('precision', 0):.2f}" if freq else ''
            row['LT_Low_Freq_Prec'] = f"{freq.get('Low_Freq_<10', {}).get('precision', 0):.2f}" if freq else ''

            writer.writerow(row)

    print(f"统一结果已保存到: {output_path}")
    print(f"共 {len(model_order)} 个模型, {len(fieldnames)-1} 个指标")

    # 同时输出一个便于论文使用的Markdown表格
    generate_markdown_table(model_order, ner_results, geoglue_results, longtail_results)


def generate_markdown_table(model_order, ner_results, geoglue_results, longtail_results):
    """生成Markdown格式的表格，方便直接粘贴到论文中"""
    md_path = os.path.join(RESULT_DIR, 'unified_results.md')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 实验结果汇总\n\n")

        # 表1: NER F1结果
        f.write("## 表1: NER模型对比 (Micro-F1 %)\n\n")
        f.write("| Model | CLUENER | MSRA | Weibo | CMNER | Avg |\n")
        f.write("|-------|---------|------|-------|-------|-----|\n")
        for model in model_order:
            ner = ner_results.get(model, {})
            if not ner:
                continue
            cluener = ner.get('cluener_micro_f1', 0) * 100
            msra = ner.get('msra_micro_f1', 0) * 100
            weibo = ner.get('weibo_ner_micro_f1', 0) * 100
            cmner = ner.get('cmner_micro_f1', 0) * 100
            avg = ner.get('avg_micro_f1', 0) * 100
            f.write(f"| {model} | {cluener:.2f} | {msra:.2f} | {weibo:.2f} | {cmner:.2f} | {avg:.2f} |\n")
        f.write("\n")

        # 表2: GeoGLUE结果（仅NER子任务，百分制显示）
        f.write("## 表2: GeoGLUE地名识别评测结果 (NER子任务, 百分制)\n\n")
        f.write("| Model | GeoETA(分) | GeoCPA(分) | GeoWWC(分) | 综合分(分) | 等级 |\n")
        f.write("|-------|-----------|-----------|-----------|-----------|------|\n")
        for model in model_order:
            geo = geoglue_results.get(model, {})
            if not geo:
                continue
            # CSV中保存的是百分制值，直接使用
            eta_micro = geo.get('GeoETA_micro_f1', 0)
            cpa_micro = geo.get('GeoCPA_micro_f1', 0)
            wwc_micro = geo.get('GeoWWC_micro_f1', 0)
            overall = geo.get('_summary_overall_score', geo.get('_summary_avg_micro_f1', 0))
            if overall >= 90:
                grade = "A"
            elif overall >= 80:
                grade = "B"
            elif overall >= 70:
                grade = "C"
            elif overall >= 60:
                grade = "D"
            else:
                grade = "F"
            f.write(f"| {model} | {eta_micro:.1f} | {cpa_micro:.1f} | {wwc_micro:.1f} | {overall:.1f} | {grade} |\n")
        f.write("\n")

        # 表3: 长尾实体识别-正则分类
        f.write("## 表3: 长尾地名实体识别-正则分类准确率\n\n")
        f.write("| Model | County | Historical | Natural | Admin | Town | POI | Avg |\n")
        f.write("|-------|--------|------------|---------|-------|------|-----|-----|\n")
        for model in model_order:
            lt = longtail_results.get(model, {})
            regex = lt.get('regex', {})
            if not regex:
                continue
            county = regex.get('County_Level', {}).get('precision', 0)
            hist = regex.get('Historical', {}).get('precision', 0)
            natural = regex.get('Natural_Geo', {}).get('precision', 0)
            admin = regex.get('Administrative', {}).get('precision', 0)
            town = regex.get('Town_Village', {}).get('precision', 0)
            poi = regex.get('POI', {}).get('precision', 0)
            avg = (county + hist + natural + admin + town + poi) / 6
            f.write(f"| {model} | {county:.2f} | {hist:.2f} | {natural:.2f} | {admin:.2f} | {town:.2f} | {poi:.2f} | {avg:.2f} |\n")
        f.write("\n")

        # 表4: 长尾实体识别-频率分组
        f.write("## 表4: 长尾地名实体识别-频率分组准确率\n\n")
        f.write("| Model | High Freq | Medium Freq | Low Freq |\n")
        f.write("|-------|-----------|-------------|----------|\n")
        for model in model_order:
            lt = longtail_results.get(model, {})
            freq = lt.get('freq', {})
            if not freq:
                continue
            high = freq.get('High_Freq_>50', {}).get('precision', 0)
            medium = freq.get('Medium_Freq_10-50', {}).get('precision', 0)
            low = freq.get('Low_Freq_<10', {}).get('precision', 0)
            f.write(f"| {model} | {high:.2f} | {medium:.2f} | {low:.2f} |\n")

    print(f"Markdown表格已保存到: {md_path}")


if __name__ == '__main__':
    generate_unified_csv()
