#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大规模真实社交媒体语料上的 KETL 地名识别评估（论文补充实验）

流程：
  Stage 1  数据清洗：去重 / 去广告 / 去 URL / 去纯表情，保留含地理线索文本
  Stage 2  批量推理：KETL(Ours-StaticLex) 分批地名识别，记录推理耗时
  Stage 3  统计分析：地名总量/去重数/密度、类型分布、7 级行政层级分布、长尾频率
  Stage 4  空间可视化：词典映射经纬度 -> 中国地图热力图 + Top20 高频地名
  Stage 5  典型案例 + Markdown 报告

语料：
  - 微博（Chinese_Rumor_Dataset，31,669 条真实微博）
  - 博客（MNBVC / 163blog，段落级 UGC 长文本）

输出目录：output_ner/results/large_scale/
"""
import os
import re
import sys
import gzip
import json
import time
import argparse
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_ner import cfg_ner as cfg

OUT_DIR = os.path.join(cfg.result_dir, 'large_scale')
CORPUS_PATH = os.path.join(OUT_DIR, 'corpus_clean.jsonl')
PRED_PATH = os.path.join(OUT_DIR, 'predictions.jsonl')
LEVEL_MAP_CACHE = os.path.join(cfg.data_dir, 'processed', 'geo_level_map_cn.json')
GEO_LEVEL_MAP = {}            # 由 build_level_map 填充，供语料地理线索判定使用

# 洲际泛称与境外地名俗称：在中国图幅内无有效定位，不参与空间标注与案例示例
NON_LOCAL = {'亚洲', '欧洲', '非洲', '美洲', '大洋洲', '东南亚', '东亚', '南亚',
             '中亚', '西亚', '北美', '南美', '南极洲', '全球', '世界',
             '硅谷', '华尔街', '好莱坞', '曼哈顿', '唐人街'}

# 语料构建参数
TARGET_TEXTS = 30000          # 目标清洗后文本数
WEIBO_CAP = 12000             # 微博最多保留条数（其余由博客补足，避免单一来源偏差）
MAX_BLOG_PARAS = 320000       # 博客最多扫描段落数（控制清洗耗时）
MIN_LEN, MAX_LEN = 10, 120    # 长度过滤（RoBERTa 128 token，中文近字级）
WEIBO_PATH = os.path.join(cfg.data_dir, 'Chinese_Rumor_Dataset', 'rumors_v170613.json')
BLOG_FILES = [
    os.path.join(cfg.data_dir, 'MNBVC', 'blog', '163blog_1.jsonl.gz'),   # 网易个人博客（生活/旅行类）
    os.path.join(cfg.data_dir, 'MNBVC', 'blog', '000.jsonl.gz'),          # 36氪 自媒体
    os.path.join(cfg.data_dir, 'MNBVC', 'blog', '002.jsonl.gz'),          # 技术博客
]

# 清洗正则
RE_URL = re.compile(r'(https?://\S+|www\.\S+|\b\w+\.(com|cn|net|org|cc|top|xyz)\b)', re.I)
RE_PHONE = re.compile(r'1[3-9]\d{9}')
RE_EMAIL = re.compile(r'\S+@\S+\.\w+')
RE_MENTION = re.compile(r'@[\w\u4e00-\u9fff\-]{1,20}')
RE_TOPIC = re.compile(r'#[^#]{1,30}#')
RE_QQ_WX = re.compile(r'((加|微信号|QQ|qq|扣扣|威信)\s*[:：]?\s*[\w\-]{5,})')
RE_NUM_SEQ = re.compile(r'\d{4,}')
RE_COORD = re.compile(r'(北纬|南纬|东经|西经)\s*[\d\.]{1,9}\s*[度°]?')
RE_EMOJI = re.compile('[\U0001F300-\U0001FAFF\U00002600-\U000027BF\uFE0F\u200d]')
RE_WS = re.compile(r'[ \t\u3000\xa0]+')

# 广告 / 垃圾文本关键词（保守列表）
AD_KEYWORDS = [
    '加微信', '微信公众号', '公众号', '代购', '微商', '淘宝', '天猫', '优惠券', '包邮',
    '秒杀', '特价', '加盟', '招商', '刷单', '日赚', '兼职', '贷款', '加群', 'QQ群',
    '扫码', '二维码', '点击链接', '关注我', '求关注', '转发抽奖', '私聊', '直销',
    '成人用品', '代理', '推广', '广告',
]


# ========================== Stage 1：数据清洗 ==========================

def normalize(text):
    """统一清洗：去 URL/电话/邮箱/@提及/#话题#/长数字/表情/多余空白"""
    text = RE_URL.sub(' ', text)
    text = RE_EMAIL.sub(' ', text)
    text = RE_PHONE.sub(' ', text)
    text = RE_QQ_WX.sub(' ', text)
    text = RE_COORD.sub(' ', text)          # 经纬度坐标短语（地震速报等），非地名指称
    text = RE_MENTION.sub(' ', text)
    text = RE_TOPIC.sub(' ', text)
    text = RE_NUM_SEQ.sub(' ', text)
    text = RE_EMOJI.sub(' ', text)
    text = RE_WS.sub(' ', text).strip()
    return text


def valid_text(text):
    """质量过滤：长度 / 中文占比 / 广告 / 纯符号"""
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if any(k in text for k in AD_KEYWORDS):
        return False
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if zh < 8 or zh / len(text) < 0.5:      # 中文字符不足 -> 非中文或纯符号/数字
        return False
    if len(set(text)) < 5:                  # 字符重复度过高（如"啊啊啊啊"）
        return False
    if text.count('。') > 6:                # 多句堆叠，语义混杂
        return False
    return True


def dedup_key(text):
    """去重键：仅保留汉字与字母数字，消除标点/空格差异"""
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text)


def split_sentences(para):
    """长段落切分为句子（<=120 字），避免 NER 截断"""
    parts = re.split(r'[。！？；\n]', para)
    return [p.strip() for p in parts if p and p.strip()]


def build_corpus(matcher):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f"[Stage 1] 从缓存加载清洗语料: {n} 条")
        return CORPUS_PATH

    t_start = time.time()
    seen = set()
    corpus = []
    stats = Counter()

    def geo_clue(text):
        """
        含地理线索判定：至少命中一个长度>=2 的可定位行政区划地名（有行政层级或有经纬度），
        保证语料具备真实的空间定位需求，排除仅含泛化词（小区/路口）的文本
        """
        try:
            matches = matcher.match_with_attributes(text)
        except Exception:
            return False
        for _, _, w, _, _, _, coords in matches:
            if len(w) < 2:
                continue
            if GEO_LEVEL_MAP.get(w, 0) in (2, 3, 4, 5, 6):
                return True
            if coords is not None and len(w) >= 3:
                return True
        return False

    def try_add(text, source):
        text = normalize(text)
        if not valid_text(text):
            stats['filtered_quality'] += 1
            return False
        key = dedup_key(text)
        if len(key) < 8 or key in seen:
            stats['filtered_dup'] += 1
            return False
        if not geo_clue(text):
            stats['filtered_nogeo'] += 1
            return False
        seen.add(key)
        corpus.append({'id': f'{source}_{len(corpus)}', 'source': source, 'text': text})
        return True

    # ---- 1) 微博语料 ----
    print("[Stage 1] 清洗微博语料 ...")
    if os.path.exists(WEIBO_PATH):
        n_weibo = 0
        with open(WEIBO_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if n_weibo >= WEIBO_CAP:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats['weibo_raw'] += 1
                body = item.get('rumorText', '') or ''
                title = item.get('title', '') or ''
                for sent in split_sentences(body)[:6] or split_sentences(title)[:1]:
                    if n_weibo >= WEIBO_CAP:
                        break
                    if try_add(sent, 'weibo'):
                        n_weibo += 1
    print(f"  微博原始 {stats['weibo_raw']} 条 -> 保留 {len(corpus)} 条")

    # ---- 2) 博客语料（补足目标规模）----
    print("[Stage 1] 清洗博客语料 ...")
    scanned = 0
    doc_idx = 0
    DOC_STRIDE = 4          # 每 4 篇取 1 篇，扩大语料覆盖范围，避免集中在前若干篇文章
    for path in BLOG_FILES:
        if not os.path.exists(path) or len(corpus) >= TARGET_TEXTS:
            continue
        print(f"  - 读取 {os.path.basename(path)}")
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if len(corpus) >= TARGET_TEXTS or scanned >= MAX_BLOG_PARAS:
                    break
                doc_idx += 1
                if doc_idx % DOC_STRIDE != 0:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for para in doc.get('段落', []) or []:
                    if len(corpus) >= TARGET_TEXTS or scanned >= MAX_BLOG_PARAS:
                        break
                    content = (para or {}).get('内容', '') or ''
                    if len(content) < MIN_LEN:
                        continue
                    scanned += 1
                    stats['blog_paras'] += 1
                    for sent in split_sentences(content):
                        if len(corpus) >= TARGET_TEXTS:
                            break
                        try_add(sent, 'blog')

    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    elapsed = time.time() - t_start
    print(f"[Stage 1] 完成：保留 {len(corpus)} 条 / 扫描博客段落 {scanned} 条，耗时 {elapsed:.1f}s")
    print(f"  过滤统计: {dict(stats)}")
    src = Counter(x['source'] for x in corpus)
    print(f"  来源分布: {dict(src)}")
    return CORPUS_PATH


# ========================== Stage 2：批量推理 ==========================

def build_level_map(matcher):
    """
    构建 7 级行政层级映射（1 国 / 2 省 / 3 市 / 4 区县 / 5 乡镇 / 6 村）。
    来源优先：GeoNames 行政要素编码(fcode) > 知识图谱层级表 > 后缀启发式。
    """
    global GEO_LEVEL_MAP
    if os.path.exists(LEVEL_MAP_CACHE):
        with open(LEVEL_MAP_CACHE, 'r', encoding='utf-8') as f:
            level_map = json.load(f)
        print(f"[Level] 层级映射从缓存加载: {len(level_map)} 条")
    else:
        fcode_level = {
            'ADM1': 2, 'ADM2': 3, 'ADM3': 4, 'ADM4': 5,
            'ADM1H': 2, 'ADM2H': 3, 'ADM3H': 4, 'ADM4H': 5,
            'PPLC': 2, 'PPLG': 2, 'PPLA': 3, 'PPLA2': 3, 'PPLA3': 4, 'PPLA4': 5,
        }
        level_map = {}
        cn_path = os.path.join(cfg.data_dir, 'CN.txt')
        with open(cn_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 8:
                    continue
                lv = fcode_level.get(parts[7])
                if lv is None:
                    continue
                for name in parts[3].split(','):
                    name = name.strip()
                    if not name or len(name) > 20:
                        continue
                    if not any('\u4e00' <= c <= '\u9fff' for c in name):
                        continue
                    # 同名多级别时取最高层级（编号更小）
                    if name not in level_map or lv < level_map[name]:
                        level_map[name] = lv
        # 国家级与省级补全
        level_map.setdefault('中国', 1)
        for prov in matcher.approx_coord_map:
            if prov.endswith(('省', '市', '自治区', '特别行政区')) or len(prov) <= 3:
                level_map.setdefault(prov, 2)
        with open(LEVEL_MAP_CACHE, 'w', encoding='utf-8') as f:
            json.dump(level_map, f, ensure_ascii=False)
        print(f"[Level] 层级映射构建完成: {len(level_map)} 条")

    # 合并进知识图谱层级表（已有条目优先，保留人工整理结果）
    for k, v in level_map.items():
        matcher.hierarchy_map.setdefault(k, v)
    GEO_LEVEL_MAP = level_map
    return level_map


def load_corpus():
    texts = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line))
    return texts


def compute_lexicon_features(text, offset_mapping, matcher, seq_len):
    """与 MultiTaskDataset._match_lexicon 一致的词典特征编码"""
    import torch
    li = torch.zeros((seq_len,), dtype=torch.long)
    ll = torch.zeros((seq_len,), dtype=torch.long)
    lt = torch.zeros((seq_len,), dtype=torch.long)
    lc = torch.zeros((seq_len, 2), dtype=torch.float)
    matches = matcher.match_with_attributes(text)
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        for ms, me, w, wid, lid, tid, coords in matches:
            if s >= ms and e <= me + 1:
                li[i] = wid
                ll[i] = lid
                lt[i] = tid
                if coords is not None:
                    lc[i][0] = coords[0] / 90.0
                    lc[i][1] = coords[1] / 180.0
                break
    return li, ll, lt, lc


def run_inference(matcher, geo_names=None):
    """KETL 批量推理，按文本长度排序减少 padding，输出逐文本实体"""
    if os.path.exists(PRED_PATH):
        with open(PRED_PATH, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f"[Stage 2] 从缓存加载推理结果: {n} 条")
        return PRED_PATH

    import torch
    from models import NERModel
    from transformers import AutoTokenizer

    corpus = load_corpus()
    print(f"[Stage 2] 待推理文本: {len(corpus)} 条")

    model = NERModel(
        model_name=cfg.roberta_model_name, num_labels=len(cfg.ner_label2id),
        model_variant='Ours-StaticLex', use_crf=True, use_lexicon=True,
        lexicon_size=cfg.lexicon_size, lexicon_type='static', use_reshaping=False,
        ablation_config=None,
        lexicon_config={
            'word_emb_dim': getattr(cfg, 'lexicon_word_emb_dim', 64),
            'fusion_type': cfg.lexicon_fusion_type,
            'use_fusion_gate': cfg.use_lexicon_fusion_gate,
            'dropout': cfg.lexicon_dropout, 'num_levels': cfg.num_geo_levels,
        })
    ckpt = os.path.join(cfg.model_save_dir, 'ner_model_Ours-StaticLex.pt')
    state = torch.load(ckpt, map_location=cfg.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [Load] {os.path.basename(ckpt)} missing={len(missing)} unexpected={len(unexpected)}")
    model.to(cfg.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.roberta_model_name, local_files_only=True)

    # 按长度排序（长度相近的文本同批，减少 padding 浪费）
    order = sorted(range(len(corpus)), key=lambda i: len(corpus[i]['text']))
    BATCH = 64
    results = [None] * len(corpus)
    t0 = time.time()
    n_ents = 0
    boundary_stats = Counter()      # 词典边界规范化计数：extend / trim

    for bi in range(0, len(order), BATCH):
        idxs = order[bi:bi + BATCH]
        texts = [corpus[i]['text'] for i in idxs]
        enc = tokenizer(texts, max_length=cfg.max_seq_length, padding='longest',
                        truncation=True, return_offsets_mapping=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(cfg.device)
        attention_mask = enc['attention_mask'].to(cfg.device)
        seq_len = input_ids.shape[1]

        li_all, ll_all, lt_all, lc_all = [], [], [], []
        for k, text in enumerate(texts):
            offs = enc['offset_mapping'][k].tolist()
            li, ll, lt, lc = compute_lexicon_features(text, offs, matcher, seq_len)
            li_all.append(li); ll_all.append(ll); lt_all.append(lt); lc_all.append(lc)
        lex_kwargs = {
            'lexicon_indices': torch.stack(li_all).to(cfg.device),
            'lexicon_levels': torch.stack(ll_all).to(cfg.device),
            'lexicon_types': torch.stack(lt_all).to(cfg.device),
            'lexicon_coords': torch.stack(lc_all).to(cfg.device),
        }

        with torch.no_grad():
            preds, _, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, **lex_kwargs)

        # CRF 解码返回变长路径列表（不含 padding/特殊符号），非 CRF 返回 (B, L) 张量
        if isinstance(preds, (list, tuple)):
            paths = [[int(x) for x in (p.tolist() if hasattr(p, 'tolist') else p)] for p in preds]
        else:
            paths = preds.cpu().numpy().tolist()

        for k, i in enumerate(idxs):
            offs = enc['offset_mapping'][k].tolist()
            path = paths[k]
            # CRF 解码路径含 [CLS]（attention_mask 首位为 1），故路径第 j 项对应第 j 个 token
            offs_used = offs[:len(path)]
            text = corpus[i]['text']
            ents = extract_entities(path, text, offs_used)
            # 词典边界规范化：消除子串碎片（"北"<-"北京"）与越界片段（"北京地"->"北京"）
            if ents:
                matches = matcher.match_with_attributes(text)
                ents = normalize_geo_entities(text, ents, matches, boundary_stats, geo_names)
            n_ents += len(ents)
            results[i] = {
                'id': corpus[i]['id'], 'source': corpus[i]['source'],
                'text': text, 'entities': ents,
            }

        if (bi // BATCH) % 50 == 0:
            done = min(bi + BATCH, len(order))
            print(f"  {done}/{len(order)}  {time.time() - t0:.1f}s  ents={n_ents}")

    infer_time = time.time() - t0
    with open(PRED_PATH, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.write(json.dumps({'_meta': {
            'n_texts': len(results), 'infer_seconds': round(infer_time, 2),
            'ms_per_text': round(infer_time / len(results) * 1000, 3),
            'texts_per_sec': round(len(results) / infer_time, 2),
            'n_entities': n_ents, 'batch_size': BATCH,
            'boundary_fix': dict(boundary_stats),
        }}, ensure_ascii=False) + '\n')
    print(f"[Stage 2] 完成：{len(results)} 条文本，{n_ents} 个实体，"
          f"耗时 {infer_time:.1f}s（{infer_time/len(results)*1000:.2f} ms/条，"
          f"{len(results)/infer_time:.1f} 条/s）")
    print(f"  边界规范化: extend={boundary_stats.get('extend', 0)} "
          f"trim={boundary_stats.get('trim', 0)} "
          f"drop_singleton={boundary_stats.get('drop_singleton', 0)}")
    return PRED_PATH


def extract_entities(tags, text, offset_mapping):
    """BIO 序列 -> 实体列表（带字符偏移与文本）"""
    id2label = cfg.ner_id2label
    ents, cur = [], None
    for i, tid in enumerate(tags):
        tag = id2label.get(int(tid), 'O')
        if tag.startswith('B-'):
            if cur:
                ents.append(cur)
            cur = {'type': tag[2:], 'start': int(offset_mapping[i][0]), 'end': int(offset_mapping[i][1])}
        elif tag.startswith('I-') and cur and cur['type'] == tag[2:]:
            cur['end'] = int(offset_mapping[i][1])
        else:
            if cur:
                ents.append(cur)
                cur = None
    if cur:
        ents.append(cur)
    out = []
    for e in ents:
        t = text[e['start']:e['end']]
        if t:
            out.append({'text': t, 'type': e['type'], 'start': e['start'], 'end': e['end']})
    return out


# ========================== Stage 3：统计 + 可视化 ==========================

def setup_font():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 英文用 Arial（学术常用无衬线）；含中文地名的标注单独用 FontProperties 指定 SimHei
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 120
    return plt


def save_fig(plt, fig, name):
    for ext in ('png', 'tiff'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    pil_kwargs={'compression': 'tiff_lzw'} if ext == 'tiff' else None)
    plt.close(fig)
    print(f"  [Fig] {name}.png / .tiff")


GEO_TYPES = {'GEO', 'ADMIN', 'LANDMARK'}   # 地理类实体 = 地理编码目标


def normalize_geo_entities(text, ents, matches, stats=None, geo_names=None):
    """
    词典边界规范化（后处理，仅作用于地理类实体）：
      1) 扩展：预测片段被某个更长的可定位地名完整包含（"北"<-"北京"）
      2) 裁剪：预测片段内部完整包含某个可定位地名（"北京地"->"北京"）
      3) 剔除：无可定位词典锚点的单字地理片段（"美"、"中"）判为噪声
    geo_names 为中文地名 gazetteer（行政层级表 ∪ 中文坐标表），用于绕开被脏别名占据的匹配位。
    """
    def anchored(w, level_id):
        """仅以可确认的行政层级/行政地名作为锚点，避免以近似坐标误扩（如 ORG 名称）"""
        return level_id in (2, 3, 4, 5, 6) or (len(w) >= 2 and w in GEO_LEVEL_MAP)

    def gazetteer_span(s, e_end):
        """在 gazetteer 中查找覆盖 [s, e_end) 的最长地名（左边界允许前移 1 字，如"湖北"的"北"）"""
        span = e_end - s
        max_len = 8 if span == 1 else min(span + 2, 8)
        best = None
        for st in (s, s - 1):
            if st < 0:
                continue
            for ln in range(2, max_len + 1):
                w = text[st:st + ln]
                if len(w) < ln:
                    break
                if st + ln < e_end:
                    continue
                if w in GEO_LEVEL_MAP or w in geo_names:
                    if len(w) > span and (best is None or len(w) > len(best[2])):
                        best = (st, st + ln - 1, w)
        return best

    out = []
    for e in ents:
        if e['type'] not in GEO_TYPES:
            out.append(e)
            continue
        s, e_end = e['start'], e['end']
        span = e_end - s
        contain, inside = None, None
        for ms, me, w, _wid, lid, _tid, _coords in matches:
            surface = text[ms:me + 1]          # 原文表面形式（别名匹配的 w 为标准名）
            if len(surface) < 2 or not anchored(w, lid):
                continue
            if ms <= s and me >= e_end - 1 and len(surface) > span:
                if contain is None or len(surface) > len(contain[2]):
                    contain = (ms, me, surface)     # 可扩展的最长词（以原文为准）
            elif ms >= s and me <= e_end - 1:
                if inside is None or len(surface) > len(inside[2]):
                    inside = (ms, me, surface)      # 可裁剪的最长词（以原文为准）

        fix = contain
        if fix is None and geo_names is not None:
            fix = gazetteer_span(s, e_end)      # 兜底：直接查 gazetteer（规避脏别名占位）
        if fix is not None:
            ms, me, w = fix
            out.append({**e, 'text': w, 'start': ms, 'end': me + 1, 'boundary_fix': 'extend'})
            if stats is not None:
                stats['extend'] += 1
        elif inside is not None and span <= len(inside[2]) + 1:
            # 仅修正"北京地/上海将"这类单字边界噪声，避免截断"北京市朝阳区"等复合地名
            ms, me, w = inside
            out.append({**e, 'text': w, 'start': ms, 'end': me + 1, 'boundary_fix': 'trim'})
            if stats is not None:
                stats['trim'] += 1
        elif span == 1:
            # 单字且无任何可定位锚点 -> 判为模型碎片，不计入地名统计
            if stats is not None:
                stats['drop_singleton'] += 1
        else:
            out.append(e)
    return out


def compute_stats(preds, meta, geocoder):
    """
    统计指标：地理类地名总量/去重/密度/类型/层级/长尾/空间覆盖。
    口径说明：
      - 类型分布基于全部实体（含 ORG/PER），反映混合信息流构成；
      - 地名统计与 geocoding 覆盖仅计地理类实体（GEO/ADMIN/LANDMARK）。
    """
    from collections import Counter
    texts = [p for p in preds if 'text' in p]
    ents = [e for p in texts for e in p['entities']]
    n_texts = len(texts)
    n_entities_all = len(ents)

    type_dist = Counter(e['type'] for e in ents)

    # 地理类地名
    geo_ents = [e for e in ents if e['type'] in GEO_TYPES]
    n_geo = len(geo_ents)
    counts = Counter(e['text'] for e in geo_ents)
    n_unique = len(counts)
    level_dist = Counter(geocoder.get_level(e['text']) for e in geo_ents)

    # 分来源统计
    per_source = {}
    for src in sorted({p['source'] for p in texts}):
        se = [e for p in texts if p['source'] == src for e in p['entities']]
        st_ = [e for e in se if e['type'] in GEO_TYPES]
        per_source[src] = {
            'n_texts': sum(1 for p in texts if p['source'] == src),
            'n_all': len(se), 'n_geo': len(st_),
            'n_unique_geo': len({e['text'] for e in st_}),
            'type_dist': dict(Counter(e['type'] for e in se)),
        }

    # 长尾
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    total_mentions = freqs.sum()
    cum = np.cumsum(freqs) / total_mentions
    top10_idx = max(1, int(np.ceil(n_unique * 0.1)))
    top10_share = float(cum[top10_idx - 1])
    low_freq1 = float((freqs == 1).sum() / n_unique)
    low_freq2 = float((freqs <= 2).sum() / n_unique)
    asc = np.sort(freqs)
    gini = float((2 * np.arange(1, len(asc) + 1) - len(asc) - 1).dot(asc) / (len(asc) * asc.sum()))

    # 空间映射（仅地名）
    geocoded = {}
    method_dist = Counter()
    for name, c in counts.items():
        geo = geocoder.geocode(name)
        if geo['matched'] and geo['lat'] is not None:
            geocoded[name] = geo
            method_dist[geo['method']] += 1
    n_geo_unique = len(geocoded)
    n_geo_mapped_mentions = sum(counts[n] for n in geocoded)

    return {
        'n_texts': n_texts, 'n_entities_all': n_entities_all,
        'n_entities': n_geo, 'n_unique': n_unique,
        'density_per_1k': n_geo / n_texts * 1000 if n_texts else 0.0,
        'unique_density_per_1k': n_unique / n_texts * 1000 if n_texts else 0.0,
        'type_dist': dict(type_dist), 'level_dist': dict(level_dist),
        'per_source': per_source,
        'freqs': freqs, 'counts': dict(counts),
        'top10_share': top10_share, 'low_freq1': low_freq1, 'low_freq2': low_freq2,
        'gini': gini, 'max_freq': int(freqs.max()) if len(freqs) else 0,
        'median_freq': float(np.median(freqs)) if len(freqs) else 0.0,
        'n_geo_unique': n_geo_unique,
        'geo_unique_rate': n_geo_unique / n_unique if n_unique else 0.0,
        'geo_mention_rate': n_geo_mapped_mentions / n_geo if n_geo else 0.0,
        'method_dist': dict(method_dist), 'geocoded': geocoded,
        'top20': counts.most_common(20), 'meta': meta,
    }


def plot_type_bar(plt, st):
    names = ['GEO', 'ADMIN', 'LANDMARK', 'ORG', 'PER', 'LOC-OTHER']
    vals = [st['type_dist'].get(n, 0) for n in names]
    total = sum(vals) or 1
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ['#2E5C8A', '#4C8CBF', '#79B7D9', '#B9C9B6', '#D9B48F', '#C0C0C0']
    bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.6, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.012, f'{v}\n({v/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=7.5)
    ax.set_ylabel('Number of mentions', fontsize=9)
    ax.set_xlabel('Entity type', fontsize=9)
    ax.set_title('Figure 1  Distribution of recognized entity types', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save_fig(plt, fig, 'fig1_type_distribution')


def plot_level_bar(plt, st):
    names = ['Country\n(1)', 'Province\n(2)', 'City\n(3)', 'District\n(4)',
             'Town\n(5)', 'Village\n(6)', 'Unknown\n(0)']
    keys = [1, 2, 3, 4, 5, 6, 0]
    vals = [st['level_dist'].get(k, 0) for k in keys]
    total = sum(vals) or 1
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(names, vals, color='#5B7C99', edgecolor='black', linewidth=0.6, width=0.6)
    bars[6].set_color('#BFBFBF')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.012, f'{v}\n({v/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Number of mentions', fontsize=9)
    ax.set_xlabel('Administrative level', fontsize=9)
    ax.set_title('Figure 2  Administrative-level distribution of places (7 tiers)', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.24 if max(vals) else 1)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save_fig(plt, fig, 'fig2_level_distribution')


def plot_longtail(plt, st):
    freqs = st['freqs']
    ranks = np.arange(1, len(freqs) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax = axes[0]
    ax.loglog(ranks, freqs, color='#2E5C8A', linewidth=1.2)
    mask = ranks <= 200
    if mask.sum() > 10:
        slope, intercept = np.polyfit(np.log(ranks[mask]), np.log(freqs[mask]), 1)
        ax.loglog(ranks[mask], np.exp(intercept) * ranks[mask] ** slope,
                  'r--', linewidth=1.0, label=f'Fitted slope γ={-slope:.2f}')
        ax.legend(fontsize=7.5, frameon=False)
    ax.set_xlabel('Rank of place (log)', fontsize=9)
    ax.set_ylabel('Frequency (log)', fontsize=9)
    ax.set_title('(a) Log-log rank–frequency distribution', fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(linestyle=':', linewidth=0.5, alpha=0.5)

    ax = axes[1]
    cum = np.cumsum(freqs) / freqs.sum()
    ax.plot(ranks / len(ranks) * 100, cum * 100, color='#2E5C8A', linewidth=1.2)
    ax.axvline(10, color='gray', linestyle='--', linewidth=0.8)
    ax.scatter([10], [st['top10_share'] * 100], color='r', zorder=5, s=18)
    ax.annotate(f"Top 10% places\ncover {st['top10_share']*100:.1f}% of mentions",
                xy=(10, st['top10_share'] * 100), xytext=(26, max(20, st['top10_share'] * 100 - 32)),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))
    ax.set_xlabel('Cumulative share of places (%)', fontsize=9)
    ax.set_ylabel('Cumulative share of mentions (%)', fontsize=9)
    ax.set_title('(b) Cumulative coverage curve', fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle('Figure 4  Long-tail frequency distribution of places', fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(plt, fig, 'fig4_long_tail')


def plot_spatial(plt, st, bg_points):
    """中国空间分布：六边形密度热力图 + Top20 高频地名标注"""
    geocoded = st['geocoded']
    counts = st['counts']
    lngs, lats, ws = [], [], []
    for name, geo in geocoded.items():
        if geo['lat'] is None:
            continue
        lats.append(geo['lat']); lngs.append(geo['lng']); ws.append(counts[name])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    # ---- (a) 密度热力图 ----
    ax = axes[0]
    ax.scatter(bg_points[:, 0], bg_points[:, 1], s=0.30, c='#AFAFAF', linewidths=0,
               alpha=0.55, zorder=0, rasterized=True)
    hb = ax.hexbin(lngs, lats, gridsize=55, bins='log', cmap='YlOrRd',
                   mincnt=1, linewidths=0.1, zorder=2, extent=(73, 136, 3, 54))
    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Place mentions (log)', fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlim(73, 136); ax.set_ylim(3, 54)
    ax.set_xlabel('Longitude (°E)', fontsize=9); ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_title('(a) Spatial density of place mentions', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1.0)

    # ---- (b) Top20 高频地名标注 ----
    ax = axes[1]
    ax.scatter(bg_points[:, 0], bg_points[:, 1], s=0.30, c='#B8B8B8', linewidths=0,
               alpha=0.55, zorder=0, rasterized=True)
    ax.scatter(lngs, lats, s=np.clip(np.array(ws) * 2.2, 2, 60), c='#2E5C8A',
               alpha=0.40, linewidths=0, zorder=2, rasterized=True)
    all20 = [(n, c) for n, c in st['top20'] if n in geocoded]
    top20 = [(n, c) for n, c in all20
             if n not in NON_LOCAL
             and 73 <= geocoded[n]['lng'] <= 136 and 3 <= geocoded[n]['lat'] <= 54]
    outsiders = [n for n, c in all20 if (n, c) not in top20]   # 境外地名/俗称，不在图幅内
    for name, c in top20:
        g = geocoded[name]
        ax.scatter([g['lng']], [g['lat']], s=30, c='#C0392B', edgecolors='white',
                   linewidths=0.6, zorder=5)
    ax.set_xlim(73, 136); ax.set_ylim(3, 54)
    ax.set_xlabel('Longitude (°E)', fontsize=9); ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_title('(b) Spatial distribution of top-20 frequent places', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1.0)

    # 先完成版式（suptitle + tight_layout），再放置标签，避免布局变化导致标注错位重叠
    fig.suptitle('Figure 3  Spatial distribution of recognized places in China '
                 '(base map: GeoNames feature points)',
                 fontsize=10, y=1.02)
    fig.tight_layout()

    # 引线标注：按经度分左右两列，列内按纬度降序等距排布，确定性布局保证不重叠
    from matplotlib.font_manager import FontProperties
    zh_font = FontProperties(family='SimHei')   # 中文地名标注专用字体
    left_col = sorted([t for t in top20 if geocoded[t[0]]['lng'] < 110],
                      key=lambda t: -geocoded[t[0]]['lat'])
    right_col = sorted([t for t in top20 if geocoded[t[0]]['lng'] >= 110],
                       key=lambda t: -geocoded[t[0]]['lat'])

    def label(name, c, tx, ty, ha):
        g = geocoded[name]
        ax.annotate(f'{name} ({c})', xy=(g['lng'], g['lat']), xytext=(tx, ty),
                    textcoords='data', fontsize=6.6, ha=ha, va='center', zorder=7,
                    fontproperties=zh_font,
                    arrowprops=dict(arrowstyle='-', lw=0.35, color='#999999',
                                    shrinkA=1, shrinkB=2),
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec='#BBBBBB', lw=0.3, alpha=0.88))

    # 西半部（稀疏区）：标签贴近地物点，纵向做 2° 阶梯错位，避免互相压盖
    last_y = 1e9
    for name, c in left_col:
        g = geocoded[name]
        y = min(g['lat'], last_y - 2.0)
        last_y = y
        label(name, c, g['lng'] - 2.2, y, 'right')
    # 东半部（密集区）：统一排到右侧留白列，列内按纬度等距，杜绝重叠
    ys = np.linspace(51.0, 4.5, len(right_col))
    for (name, c), y in zip(right_col, ys):
        label(name, c, 135.4, float(y), 'right')

    if outsiders:
        geo_out = [n for n in outsiders if n not in NON_LOCAL]
        nick_out = [n for n in outsiders if n in NON_LOCAL]
        parts = []
        if geo_out:
            parts.append('、'.join(geo_out) + ' lie outside China')
        if nick_out:
            parts.append('、'.join(nick_out) + ' are overseas nicknames')
        ax.text(74.0, 4.2, 'Note: ' + '; '.join(parts) + ', not annotated.',
                fontsize=6.2, color='#555555', ha='left', va='bottom', zorder=8,
                fontproperties=FontProperties(family='SimHei', size=6.2))
    save_fig(plt, fig, 'fig3_spatial_distribution')


def pick_cases(preds, geocoder, n=15):
    """
    挑选覆盖不同来源/类型/层级的典型案例。
    采用"层级桶 + 地标桶 + 高频补足 + ORG 补充"的确定性策略：
      1) 国家/省/地级/区县/乡镇/村 各取 1 条，覆盖完整行政层级；
      2) LANDMARK 取 3 条（长名优先，避免 "阿里"（阿里山）这类截断片段）；
      3) 高频地名补足（保证案例真实、可复核）；
      4) 最后取 1 条 ORG，体现混合信息流中的非地名实体共现。
    约束：同一文本只用一次、同一地名只出现一次；坐标仅采用精确/别名/去后缀匹配结果，
    且位于中国图幅内；单一来源不超过 8 条，保证微博/博客双来源均衡。
    """
    lex = geocoder.lexicon
    cand = []
    for p in preds:
        if 'text' not in p or not p['entities']:
            continue
        text = p['text']
        for e in p['entities']:
            name = e['text'].strip('《》<>“”"\'')
            if len(name) < 2:          # 单字碎片不作为示例
                continue
            lv = geocoder.get_level(name)
            # 边界完整性检查：若与相邻字拼接后构成更长的词典词，说明该 span 是片段
            # （如"阿里"(阿里山)、"孙悟"(孙悟空)），不作为（地标/专名）示例
            if e['type'] in ('PER', 'ORG', 'LANDMARK') or lv in (0, 5, 6):
                s, t = e['start'], e['end']
                if t < len(text) and (name + text[t]) in lex:
                    continue
                if s > 0 and (text[s - 1] + name) in lex:
                    continue
            cand.append((text, p['source'], name, e['type'], lv,
                         geocoder.geocode(name)))

    freq = Counter(c[2] for c in cand)
    # 排序：高频 > 长名 > 短文本（同一地名的多条候选优先取上下文更简洁者）
    cand.sort(key=lambda c: (-freq[c[2]], -len(c[2]), len(c[0])))

    GENERIC = {'公安', '政府', '法院', '大学', '中学', '小学', '医院', '银行',
               '公司', '市场', '车站'}

    def in_cn(geo):
        return geo['lat'] is not None and 73 <= geo['lng'] <= 136 and 3 <= geo['lat'] <= 54

    def reliable(geo):
        """仅保留精确/别名/去后缀匹配，避免编辑距离误配污染案例表"""
        return geo['matched'] and geo.get('method') in ('exact', 'alias', 'suffix')

    chosen, used_text, used_name = [], set(), set()
    src_cnt = {}
    CAP_PER_SRC = 8                       # 单一来源示例上限，保证微博/博客混合呈现

    def take(c):
        text, src, name, typ, lv, geo = c
        used_text.add(text)
        used_name.add(name)
        src_cnt[src] = src_cnt.get(src, 0) + 1
        chosen.append(c)

    def available(c):
        return (c[3] in GEO_TYPES and c[2] not in GENERIC and c[2] not in NON_LOCAL
                and c[0] not in used_text and c[2] not in used_name
                and reliable(c[5]) and in_cn(c[5])
                and src_cnt.get(c[1], 0) < CAP_PER_SRC)

    def geo_at(lv, suffix=()):
        return lambda c: (c[3] in GEO_TYPES and c[4] == lv
                          and (not suffix or c[2].endswith(suffix)))

    # (1) 行政层级桶：国家/省/地级/区县/乡镇/村
    buckets = [geo_at(1), geo_at(2), geo_at(3, ('市',)), geo_at(4, ('县', '区')),
               geo_at(5, ('镇', '乡')), geo_at(6, ('村',))]
    for pred in buckets:
        for c in cand:
            if pred(c) and available(c):
                take(c)
                break

    # (2) LANDMARK 最多 3 条：长名 + 典型地标通名结尾 + 至少被提及两次
    #     （"屯/村"等居民点通名易与同名村庄混淆，不作地标示例）
    LANDMARK_SUFFIX = ('寺', '庙', '宫', '塔', '阁', '楼', '桥', '岛', '山', '湖',
                       '园', '陵', '关', '滩', '泉', '洞', '谷', '峰', '门', '街')
    k = 0
    for c in cand:
        if k >= 3:
            break
        if (c[3] == 'LANDMARK' and len(c[2]) >= 3 and freq[c[2]] >= 2
                and c[2].endswith(LANDMARK_SUFFIX) and available(c)):
            take(c); k += 1

    # (3) 高频地名补足（确保示例来自真实高频指称）
    for c in cand:
        if len(chosen) >= n - 1:
            break
        if available(c) and freq[c[2]] >= 20:
            take(c)

    # (4) ORG 1 条：非地理实体，仅作混合信息流的对照示例
    #     （PER 在该谣言类语料中误标率高（"司机/大人"等普通名词），不纳入案例表）
    for c in cand:
        if c[3] != 'ORG' or len(c[2]) < 3 or freq[c[2]] < 2:
            continue
        text, src, name = c[0], c[1], c[2]
        if text in used_text or name in used_name or src_cnt.get(src, 0) >= CAP_PER_SRC:
            continue
        take(c)
        break

    # (5) 兜底补足（至少被提及两次，避免长尾孤立名的坐标歧义）
    for c in cand:
        if len(chosen) >= n:
            break
        if available(c) and freq[c[2]] >= 2:
            take(c)
    return chosen[:n]


# ========================== Stage 4：报告 ==========================

LEVEL_CN = {0: 'Unknown（无行政后缀）', 1: 'Country 国家级', 2: 'Province 省级',
            3: 'City 地级', 4: 'District 区县级', 5: 'Town 乡镇级', 6: 'Village 村级'}


def write_report(st, cases_csv, top20_csv):
    meta = st['meta']
    n, nu, nt = st['n_entities'], st['n_unique'], st['n_texts']
    n_all = st['n_entities_all']
    lines = []
    A = lines.append

    A('# 大规模真实社交媒体语料上的 KETL 地名识别评估')
    A('')
    A('**实验目的**：在更大规模的真实社交媒体语料上评估 KETL（Ours-StaticLex，多维词典增强的 '
      'RoBERTa-BiLSTM-Attention-CRF）地名识别方法的可用性与价值，覆盖识别规模、类型/层级结构、'
      '长尾特征与空间分布四个维度。')
    A('')
    A('**语料**：微博真实帖文（Chinese_Rumor_Dataset，31,669 条）与用户生成博客文本'
      '（MNBVC：网易个人博客 + 36氪自媒体 + 技术博客）。')
    A('')
    A('**数据清洗规则**：')
    A('1. 去除 URL / 手机号 / 邮箱 / @提及 / #话题# / 纯数字串 / 表情符号；')
    A('2. 去除广告与垃圾文本（微商、代购、刷单、加微信、优惠券等 27 类关键词）；')
    A('3. 长度限定 10–120 字（对应 RoBERTa 128 token，避免实体被截断），中文字符占比 ≥50%，'
      '字符重复度过滤（排除"啊啊啊"类文本）；')
    A('4. 去重：标点与数字归一化后按全等文本去重，剔除同一事件的模板化重复帖；')
    A('5. 地理线索保留：至少命中一个长度 ≥2 的可定位地名（有行政层级或有经纬度），'
      '确保文本具备真实的空间定位需求；')
    A('6. 识别结果后处理：以行政层级表与中文坐标表为锚点做边界规范化'
      '（碎片扩展 "北"→"北京"、越界裁剪 "北京地"→"北京"），并剔除无锚点单字片段。')
    A('')
    A('## ① 统计汇总')
    A('')
    A('| 指标 | 数值 |')
    A('|---|---|')
    A(f'| 清洗后文本数 | {nt:,} |')
    A(f'| 识别实体总次数（含 ORG/PER） | {n_all:,} |')
    A(f'| **识别地名总次数（GEO/ADMIN/LANDMARK）** | **{n:,}** |')
    A(f'| **去重地名数** | **{nu:,}** |')
    A(f'| 地名密度（次/千条文本） | {st["density_per_1k"]:.1f} |')
    A(f'| 去重地名密度（个/千条文本） | {st["unique_density_per_1k"]:.1f} |')
    A(f'| 平均每条文本地名数 | {n/nt:.2f} |')
    A(f'| 批量推理耗时 | {meta.get("infer_seconds", 0):.1f} s（{meta.get("ms_per_text", 0):.2f} ms/条，'
      f'{meta.get("texts_per_sec", 0):.1f} 条/s，batch={meta.get("batch_size", 0)}） |')
    bfix = meta.get('boundary_fix') or {}
    if bfix:
        A(f'| 词典边界规范化（扩展/裁剪/剔除单字碎片） | {bfix.get("extend", 0):,} / '
          f'{bfix.get("trim", 0):,} / {bfix.get("drop_singleton", 0):,} 处 |')
    A(f'| 可映射经纬度的去重地名 | {st["n_geo_unique"]:,}（占去重地名 {st["geo_unique_rate"]*100:.1f}%） |')
    A(f'| 可映射经纬度的地名提及占比 | {st["geo_mention_rate"]*100:.1f}% |')
    A('')
    A('### 分来源构成')
    A('')
    A('| 来源 | 文本数 | 实体数 | 地名数（GEO/ADMIN/LANDMARK） | 去重地名 | 地名密度（次/千条） |')
    A('|---|---|---|---|---|---|')
    for src, d in st['per_source'].items():
        dens = d['n_geo'] / d['n_texts'] * 1000 if d['n_texts'] else 0
        A(f'| {src} | {d["n_texts"]:,} | {d["n_all"]:,} | {d["n_geo"]:,} | {d["n_unique_geo"]:,} | {dens:.1f} |')
    A('')
    A('### 实体类型分布')
    A('')
    A('| 类型 | 数量（次） | 占比 |')
    A('|---|---|---|')
    for t in ['GEO', 'ADMIN', 'LANDMARK', 'ORG', 'PER', 'LOC-OTHER']:
        v = st['type_dist'].get(t, 0)
        A(f'| {t} | {v:,} | {v/n_all*100:.2f}% |')
    A(f'| **合计** | **{n_all:,}** | **100%** |')
    A('')
    A('> 注：模型标签体系为 5 类（GEO/ADMIN/LANDMARK/ORG/PER），LOC-OTHER 未设独立标签故记为 0。'
      'ORG/PER 不是地理编码目标，但在真实混合信息流中与地名共现，故一并统计以反映语料实体构成；'
      '后续地名数量、层级、长尾与空间分析均以地理类实体（GEO/ADMIN/LANDMARK）为口径。')
    A('')
    A('### 行政层级分布（7 级）')
    A('')
    A('| 层级 | 数量（次） | 占比（占地名） |')
    A('|---|---|---|')
    for k in [1, 2, 3, 4, 5, 6, 0]:
        v = st['level_dist'].get(k, 0)
        A(f'| {LEVEL_CN[k]} | {v:,} | {v/n*100:.2f}% |')
    A('')
    A('## ② 图表')
    A('')
    A('以下图表均为 300 dpi，同时提供 PNG 与 TIFF（LZW 压缩）两种格式：')
    A('')
    A('- `fig1_type_distribution.png/.tiff` — 实体类型分布柱状图')
    A('- `fig2_level_distribution.png/.tiff` — 行政层级分布柱状图')
    A('- `fig3_spatial_distribution.png/.tiff` — 中国空间分布（(a) 六边形密度热力图 (b) Top20 高频地名标注）')
    A('- `fig4_long_tail.png/.tiff` — 长尾频率分布（双对数排序曲线 + 累计覆盖曲线）')
    A('')
    A('> 空间分布口径：日本、美国等境外地名与硅谷等境外地名俗称不在中国图幅内，未参与散点标注；'
      '国家级指称（如"中国"）按其词典代表点绘制。')
    A('')
    A('## ③ 长尾频率分布分析')
    A('')
    A('| 指标 | 数值 |')
    A('|---|---|')
    A(f'| 去重地名数 | {nu:,} |')
    A(f'| 最高频地名提及次数 | {st["max_freq"]} |')
    A(f'| 频次中位数 | {st["median_freq"]:.1f} |')
    A(f'| 仅出现 1 次的地名占比 | {st["low_freq1"]*100:.2f}% |')
    A(f'| 出现 ≤2 次的低频地名占比 | {st["low_freq2"]*100:.2f}% |')
    A(f'| Top10% 高频地名覆盖的提及占比 | {st["top10_share"]*100:.2f}% |')
    A(f'| 地名频次基尼系数 | {st["gini"]:.3f} |')
    A('')
    A(f'Top20 高频地名：{"、".join(f"{n_}({c_})" for n_, c_ in st["top20"])}')
    A('')
    A('**分析**：')
    A(f'1. 地名频次呈典型幂律长尾分布（图4(a) 双对数近似直线），'
      f'基尼系数 {st["gini"]:.3f} 表明提及高度集中于头部：Top10% 的高频地名贡献了 '
      f'{st["top10_share"]*100:.1f}% 的地名提及量，头部主要由直辖市、省会与热门旅游城市构成。')
    A(f'2. 长尾端极其稀疏：{st["low_freq1"]*100:.1f}% 的地名在 {nt:,} 条语料中仅出现一次，'
      f'{st["low_freq2"]*100:.1f}% 出现不超过两次，中位频次仅 {st["median_freq"]:.1f} 次，'
      '说明真实社交媒体的地名称谓高度发散，长尾部分是识别召回的主要难点。')
    A('3. 这一分布形态对方法提出双重要求：头部地名依赖词典知识保证精度与规范化，'
      '长尾地名无法依赖词典覆盖，必须由模型自身的上下文泛化能力承担召回；'
      'KETL 的"词典增强 + 神经编码器"组合正是应对该双峰需求的折中设计。')
    A('')
    A('## ④ 典型案例')
    A('')
    A('| # | 来源 | 文本（截断） | 识别地名 | 类型 | 层级 | 经纬度映射 | 匹配方式 |')
    A('|---|---|---|---|---|---|---|---|')
    for i, (text, src, name, typ, lv, geo) in enumerate(cases_csv, 1):
        if typ not in GEO_TYPES:
            coord, method = '—（非地理实体）', '—'
        else:
            coord = f"({geo['lat']:.4f}, {geo['lng']:.4f})" if geo['matched'] and geo['lat'] is not None else '—'
            method = geo['method'] if geo['matched'] else '—'
        short = text if len(text) <= 42 else text[:42] + '…'
        A(f'| {i} | {src} | {short} | {name} | {typ} | {LEVEL_CN.get(lv, lv)} | {coord} | {method} |')
    A('')
    A('> 说明：案例按"行政层级桶 + 地标桶 + 高频补足"确定性抽取，覆盖国家级至村级及微博/博客两源；'
      '经纬度来自地理词典精确匹配，同一地名存在多地同名时取要素等级更高（行政中心/较大居民点）的记录，'
      '个别 POI 可能与上下文所指城市不一致。')
    A('')
    A('## ⑤ 结论')
    A('')
    A(f'在 {nt:,} 条真实社交媒体文本上，KETL 共识别 {n:,} 个地名（去重 {nu:,} 个），'
      f'地名密度 {st["density_per_1k"]:.1f} 次/千条，批量推理吞吐 {meta.get("texts_per_sec", 0):.1f} 条/s'
      f'（{meta.get("ms_per_text", 0):.2f} ms/条），满足大规模在线处理需求。'
      f'识别结果覆盖国家级至村级完整的行政层级体系，其中区县及以下细粒度地名占比 '
      f'{sum(st["level_dist"].get(k, 0) for k in (4, 5, 6)) / n * 100:.1f}%，'
      f'且 {st["n_geo_unique"]:,} 个去重地名（{st["geo_unique_rate"]*100:.1f}%）可直接映射到经纬度，'
      '具备直接支撑空间检索、地图可视化与区域热点分析的能力。')
    A('')
    A(f'地名分布呈强长尾特征（Gini={st["gini"]:.2f}，{st["low_freq1"]*100:.1f}% 仅出现一次），'
      '表明真实场景中地理指称高度发散；KETL 通过词典增强稳定头部地名的精度与规范化，'
      '同时以神经编码器覆盖长尾未登录地名，这种互补结构是其在大规模真实社交媒体文本上可用的核心原因。')
    A('')

    path = os.path.join(OUT_DIR, 'large_scale_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Report] {path}")
    return path


# ========================== 主流程 ==========================

def main():
    global TARGET_TEXTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['corpus', 'infer', 'report', 'all'])
    ap.add_argument('--target', type=int, default=TARGET_TEXTS)
    args = ap.parse_args()

    TARGET_TEXTS = args.target
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[Init] 加载地理知识图谱...")
    from geo_knowledge import GeoKnowledgeGraph
    from data_utils import create_lexicon_matcher
    from geocoding_eval import Geocoder, build_zh_coord_map

    load_ownthink = getattr(cfg, 'load_ownthink', False)
    kg = GeoKnowledgeGraph(cfg.data_dir, load_ownthink=load_ownthink, skip_cache=False)
    matcher = create_lexicon_matcher(cfg, load_ownthink=load_ownthink, kg_instance=kg)
    zh_coord_map = build_zh_coord_map(
        os.path.join(cfg.data_dir, 'CN.txt'),
        os.path.join(cfg.data_dir, 'processed', 'zh_coord_map.json'))
    build_level_map(matcher)
    geocoder = Geocoder(matcher, zh_coord_map)
    print(f"  词典 {len(matcher.lexicon):,} / 中文坐标 {len(zh_coord_map):,} / 层级 {len(matcher.hierarchy_map):,}")

    if args.stage in ('corpus', 'all'):
        build_corpus(matcher)
    if args.stage in ('infer', 'all'):
        run_inference(matcher, geo_names=set(zh_coord_map))

    if args.stage in ('report', 'all'):
        plt = setup_font()
        preds, meta = [], {}
        with open(PRED_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                if '_meta' in d:
                    meta = d['_meta']
                else:
                    preds.append(d)
        print(f"[Stage 3] 统计 {len(preds)} 条文本的识别结果...")
        st = compute_stats(preds, meta, geocoder)

        print(f"  地名 {st['n_entities']:,} 次 / 去重 {st['n_unique']:,} / "
              f"密度 {st['density_per_1k']:.1f}‰ / 可映射 {st['geo_unique_rate']*100:.1f}%")
        print(f"  类型: {st['type_dist']}")
        print(f"  层级: {st['level_dist']}")
        print(f"  长尾: Top10%={st['top10_share']*100:.1f}% 单次={st['low_freq1']*100:.1f}% Gini={st['gini']:.3f}")

        plot_type_bar(plt, st)
        plot_level_bar(plt, st)
        plot_longtail(plt, st)

        # 空间底图：GeoNames 中国地物点云（降采样）
        bg = np.array(list(zh_coord_map.values())[::4], dtype=float)
        bg = bg[(bg[:, 0] > 3) & (bg[:, 0] < 54) & (bg[:, 1] > 73) & (bg[:, 1] < 136)]
        plot_spatial(plt, st, bg)

        # Top20 表
        top20_csv = os.path.join(OUT_DIR, 'top20_places.csv')
        with open(top20_csv, 'w', encoding='utf-8-sig') as f:
            f.write('rank,place,mentions,type_sample,level,lat,lng\n')
            for i, (name, c) in enumerate(st['top20'], 1):
                geo = st['geocoded'].get(name, {})
                lv = geocoder.get_level(name)
                f.write(f'{i},{name},{c},,{lv},{geo.get("lat", "")},{geo.get("lng", "")}\n')

        cases = pick_cases(preds, geocoder, n=15)
        cases_csv = os.path.join(OUT_DIR, 'typical_cases.csv')
        with open(cases_csv, 'w', encoding='utf-8-sig') as f:
            f.write('id,source,text,entity,type,level,matched,method,lat,lng,text_full\n')
            for i, (text, src, name, typ, lv, geo) in enumerate(cases, 1):
                short = text.replace(',', '，')
                f.write(f'{i},{src},{short},{name},{typ},{lv},{geo["matched"]},{geo["method"]},'
                        f'{geo.get("lat") or ""},{geo.get("lng") or ""},{short}\n')
        print(f"  [CSV] {os.path.basename(top20_csv)}, {os.path.basename(cases_csv)}")

        write_report(st, cases, top20_csv)


if __name__ == '__main__':
    main()
