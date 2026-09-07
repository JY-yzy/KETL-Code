#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER实验配置文件
从config.py继承通用设置，但覆盖模型列表、数据路径、输出目录等
"""
import os
import sys
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

class NERConfig(Config):
    """NER实验专用配置"""
    
    def __init__(self):
        super().__init__()
        
        # ========== NER实验专用路径配置 ==========
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_ner')
        self.model_save_dir = os.path.join(self.output_dir, 'models')
        self.result_dir = os.path.join(self.output_dir, 'results')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        
        # 创建目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # ========== 实验配置 ==========
        self.experiment_name = 'ner_geo_experiments'
        
        # ========== 模型配置 ==========
        # 9个基线模型 + 4个增强模型（BiLSTM组2个 + BiGRU组2个）
        self.ner_experiment_models = [
            # ========== 9个基线模型 ==========
            {
                'name': 'BertNER',
                'model_name': self.bert_model_name,
                'use_crf': False,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': None
            },
            {
                'name': 'BertBiLSTMCRF',
                'model_name': self.bert_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm'
            },
            {
                'name': 'BertBiGRUCRF',
                'model_name': self.bert_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bigru'
            },
            {
                'name': 'BertBiLSTMAttention',
                'model_name': self.bert_model_name,
                'use_crf': False,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention'
            },
            {
                'name': 'BaselineRoBERTa',
                'model_name': self.roberta_model_name,
                'use_crf': False,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': None
            },
            {
                'name': 'RoBERTaBiLSTMCRF',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm'
            },
            {
                'name': 'RoBERTaBiLSTMAttention',
                'model_name': self.roberta_model_name,
                'use_crf': False,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention'
            },
            {
                'name': 'RoBERTaBiLSTMAttentionCRF',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention'
            },
            {
                'name': 'RoBERTaBiGRUAttentionCRF',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention'
            },
            # ========== 基于RoBERTaBiLSTMAttentionCRF的增强模型 ==========
            {
                'name': 'RoBERTaBiLSTMAttentionCRF+Lexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention'
            },
            {
                'name': 'RoBERTaBiLSTMAttentionCRF+DynamicLexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'dynamic',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention'
            },
            # ========== 基于RoBERTaBiGRUAttentionCRF的增强模型 ==========
            {
                'name': 'RoBERTaBiGRUAttentionCRF+Lexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention'
            },
            {
                'name': 'RoBERTaBiGRUAttentionCRF+DynamicLexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'dynamic',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention'
            }
        ]
        
        # ========== 消融实验配置 ==========
        # 分为两组：BiLSTM组 和 BiGRU组，每组3个模型
        self.ablation_models_bilstm = [
            {
                'name': 'RoBERTaBiLSTMAttentionCRF',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention',
                'group': 'BiLSTM'
            },
            {
                'name': 'RoBERTaBiLSTMAttentionCRF+Lexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention',
                'group': 'BiLSTM'
            },
            {
                'name': 'RoBERTaBiLSTMAttentionCRF+DynamicLexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'dynamic',
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention',
                'group': 'BiLSTM'
            }
        ]
        
        self.ablation_models_bigru = [
            {
                'name': 'RoBERTaBiGRUAttentionCRF',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': False,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention',
                'group': 'BiGRU'
            },
            {
                'name': 'RoBERTaBiGRUAttentionCRF+Lexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'static',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention',
                'group': 'BiGRU'
            },
            {
                'name': 'RoBERTaBiGRUAttentionCRF+DynamicLexicon',
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'dynamic',
                'use_reshaping': False,
                'middle_layer_type': 'bigru_attention',
                'group': 'BiGRU'
            }
        ]
        
        # 合并后的消融模型列表（兼容旧代码）
        self.ablation_models = self.ablation_models_bilstm + self.ablation_models_bigru

        # ========== Ours-StaticLex 消融实验配置 ==========
        # 共享 RoBERTa+BiLSTM+Attention+CRF 编码器，仅适配器不同
        # 训练流程：先全量训练 baseline，再以 strict=False 加载基线权重，冻结编码器/CRF/中间层，仅微调适配器
        # 共 6 个变体：baseline + w/o LexEmb + w/o Coord + w/o Mask + w/o Gate + Token Gate
        _full_ablation = {'use_level': True, 'use_type': True, 'use_coord': True,
                          'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}
        _lexonly_ablation = {'use_level': False, 'use_type': False, 'use_coord': False,
                              'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}
        _nocoord_ablation = {'use_level': True, 'use_type': True, 'use_coord': False,
                             'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}
        _nomask_ablation = {'use_level': True, 'use_type': True, 'use_coord': True,
                            'use_mask': False, 'use_global_gate': True, 'gate_type': 'global'}
        _nogate_ablation = {'use_level': True, 'use_type': True, 'use_coord': True,
                            'use_mask': True, 'use_global_gate': False, 'gate_type': 'global'}
        _tokengate_ablation = {'use_level': True, 'use_type': True, 'use_coord': True,
                               'use_mask': True, 'use_global_gate': True, 'gate_type': 'token'}

        def _make_staticlex(name, abl, finetune, epochs):
            return {
                'name': name,
                'model_name': self.roberta_model_name,
                'use_crf': True,
                'use_lexicon': True,
                'lexicon_type': 'static',          # 统一走 LexiconAdapter
                'use_reshaping': False,
                'middle_layer_type': 'bilstm_attention',
                'group': 'StaticLex',
                'ablation_config': abl,
                'adapter_only_finetune': finetune,
                'num_epochs_override': epochs,
            }

        # ========== 从头训练模式（消融效果最显著）==========
        # 方案说明：5 个消融变体从随机初始化训练 30 轮，不加载任何检查点。
        # 每个变体独立学习，移除的组件无法通过预训练权重补偿，消融差异最大。
        # 主模型 Ours-StaticLex 保持现有检查点不重训，仅加载评估 F1。
        self.ablation_models_staticlex = [
            # 1) Ours-StaticLex 基线（不重训，复用现有检查点）
            _make_staticlex('Ours-StaticLex', _full_ablation, False, None),
            # 2) w/o Lex Emb —— 仅保留 word_emb，移除 level/type/coord
            _make_staticlex('Ours-StaticLex_woLexEmb', _lexonly_ablation, False, 30),
            # 3) w/o Coord —— 移除坐标投影
            _make_staticlex('Ours-StaticLex_woCoord', _nocoord_ablation, False, 30),
            # 4) w/o Mask —— 在所有位置注入知识（不区分匹配/未匹配位置）
            _make_staticlex('Ours-StaticLex_woMask', _nomask_ablation, False, 30),
            # 5) w/o Gate —— 移除全局标量融合门控，直接残差
            _make_staticlex('Ours-StaticLex_woGate', _nogate_ablation, False, 30),
            # 6) Token Gate —— 全局标量门控替换为 token 级门控
            _make_staticlex('Ours-StaticLex_TokenGate', _tokengate_ablation, False, 30),
        ]

        # ========== StaticLex 适配器微调超参（保留兼容）==========
        self.staticlex_adapter_epochs = 10          # 消融变体适配器微调轮数
        self.staticlex_adapter_lr = 1e-3            # 适配器微调学习率（仅适配器参数，可较大）
        self.staticlex_adapter_patience = 3
        self.staticlex_freeze_keys = ['encoder', 'layers', 'classifier', 'crf_layer', 'dropout']
        # 复用现成 DynamicLexicon 检查点作为 Ours-StaticLex 起点（避免重复训练 30 轮）
        self.staticlex_reuse_dynamiclex_ckpt = True
        self.staticlex_dynamiclex_ckpt_name = 'ner_model_RoBERTaBiLSTMAttentionCRF_DynamicLexicon.pt'

        # ========== StaticLex Benchmark 配置（3 条）==========
        # 用于参数量、推理延迟、词典匹配耗时、P/R/F1 对比
        self.staticlex_benchmark_configs = [
            {
                'name': 'Base',  # RoBERTa+BiLSTM+Attn+CRF 无词典
                'model_name': self.roberta_model_name,
                'use_crf': True, 'use_lexicon': False, 'lexicon_type': 'static',
                'use_reshaping': False, 'middle_layer_type': 'bilstm_attention',
                'ablation_config': None,
                'ckpt': 'ner_model_RoBERTaBiLSTMAttentionCRF.pt',
            },
            {
                'name': 'Ours-StaticLex',  # 全多特征（与基线一致）
                'model_name': self.roberta_model_name,
                'use_crf': True, 'use_lexicon': True, 'lexicon_type': 'static',
                'use_reshaping': False, 'middle_layer_type': 'bilstm_attention',
                'ablation_config': _full_ablation,
                'ckpt': 'ner_model_Ours-StaticLex.pt',
            },
            {
                'name': 'Ours-StaticLex-LexEmbOnly',  # 仅 word_emb（对应 w/o Lex Emb 的结构）
                'model_name': self.roberta_model_name,
                'use_crf': True, 'use_lexicon': True, 'lexicon_type': 'static',
                'use_reshaping': False, 'middle_layer_type': 'bilstm_attention',
                'ablation_config': _lexonly_ablation,
                'ckpt': 'ner_model_Ours-StaticLex_woLexEmb.pt',
            },
        ]

        # ========== 网络文本测试用例路径 ==========
        self.network_text_cases_path = os.path.join(self.data_dir, 'network_text_cases.json')
        self.network_text_ground_truth_path = os.path.join(self.data_dir, 'network_text_ground_truth.json')
        
        # ========== 训练配置 ==========
        self.ner_batch_size = 8
        self.ner_eval_batch_size = 8
        self.ner_num_epochs = 30
        self.ner_learning_rate = 1e-5
        self.ner_lexicon_learning_rate = 1e-4  # 词典增强模块差异化学习率（10x编码器，加速适配器学习，避免梯度消失）
        
        # ========== 早停配置 ==========
        self.ner_patience = 5
        self.ner_early_stopping_metric = 'micro_f1'
        
        # ========== 词典配置 ==========
        self.use_geo_lexicon = True
        self.lexicon_fusion_type = 'gate'
        self.load_ownthink = True  # 是否加载ownthink_v2.csv（8GB，包含组织、人名等）
        self.lexicon_word_emb_dim = 64  # 词嵌入维度（64足够表达实体语义，平衡表达力与显存）
        # ownthink 智能筛选配置（基于实验逻辑，非粗暴截断）
        self.ownthink_max_org = 200000       # ORG 实体配额（按属性丰富度排序）
        self.ownthink_max_per = 200000       # PER 实体配额（按属性丰富度排序）
        self.ownthink_min_entity_len = 2    # 实体名最小长度（NER文本合理范围）
        self.ownthink_max_entity_len = 15   # 实体名最大长度（NER文本合理范围）
        
        # ========== 测试数据集 ==========
        self.test_datasets = ['cluener', 'msra', 'weibo_ner', 'cmner']
        
        # ========== 类别权重配置（以地名识别为主的实验逻辑）==========
        # 标签顺序: 0:O, 1:B-GEO, 2:I-GEO, 3:B-ADMIN, 4:I-ADMIN, 5:B-LANDMARK, 6:I-LANDMARK, 7:B-ORG, 8:I-ORG, 9:B-PER, 10:I-PER
        # 实验逻辑：地名识别为主 → GEO/ADMIN/LANDMARK 权重最高，ORG（含设施名）次之，PER 权重最低
        # - O: 非实体，极高频，权重极低（0.1）让模型聚焦实体识别
        # - GEO/ADMIN/LANDMARK: 地名核心任务，权重最高（2.0-4.0）
        # - ORG: 含设施/机构地名（如"北京大学"），权重中等（1.5-2.0）
        # - PER: 非地名，权重最低（0.8-1.0）避免抢夺地名识别的注意力
        self.use_class_weights = True
        self.class_weight_method = 'manual'  # 改为手动，使用下面按实验逻辑设计的权重
        self.ner_class_weights = [
            0.1,    # 0: O          - 非实体，极低权重
            3.5,    # 1: B-GEO      - 地名（核心），高权重
            2.5,    # 2: I-GEO      - 地名内部
            4.0,    # 3: B-ADMIN    - 行政区划（地名子类，最高权重）
            3.0,    # 4: I-ADMIN    - 行政区划内部
            3.0,    # 5: B-LANDMARK - 地标（地名子类，高权重）
            2.0,    # 6: I-LANDMARK - 地标内部
            2.0,    # 7: B-ORG      - 组织机构（含设施地名，中等权重）
            1.5,    # 8: I-ORG      - 组织内部
            1.0,    # 9: B-PER      - 人物（非地名，最低实体权重）
            0.8,    # 10: I-PER     - 人物内部
        ]
        self.class_weight_min = 0.1
        self.class_weight_max = 10.0
        
        # ========== 位置敏感门控配置 ==========
        self.use_position_gate = True  # 是否使用位置敏感门控
        self.use_level_gate = True     # 是否使用层级感知门控
        
        # ========== 近似坐标配置 ==========
        self.use_approx_coords = True  # 是否使用近似坐标
        
        # ========== 可视化配置 ==========
        self.plot_font_family = 'SimHei'
        self.plot_font_size = 12
    
    def switch_to_test_mode(self):
        """
        切换到TEST模式的输出目录
        将所有输出路径重定向到 output_ner_test/ 目录，避免影响正式实验结果
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, 'output_ner_test')
        self.model_save_dir = os.path.join(self.output_dir, 'models')
        self.result_dir = os.path.join(self.output_dir, 'results')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        
        # 创建目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        print(f"[TEST MODE] Output directory switched to: {self.output_dir}")

# 创建全局配置实例
cfg_ner = NERConfig()
class_weights = {
    0: 1.0,   # O
    1: 1.0,   # B-GEO
    2: 1.0,   # I-GEO
    3: 5.0,   # B-ADMIN
    4: 5.0,   # I-ADMIN
    5: 8.0,   # B-LANDMARK
    6: 8.0,   # I-LANDMARK
    7: 1.0,   # B-ORG
    8: 1.0,   # I-ORG
    9: 1.0,   # B-PER
    10: 1.0,  # I-PER
}
