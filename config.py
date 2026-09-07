#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件 - 集中管理所有超参数和配置信息
"""
import os
import torch
import random
import numpy as np

def set_random_seed(seed=42):
    """设置随机种子确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # ========== 随机种子 ==========
        self.seed = 42
        set_random_seed(self.seed)
        
        # ========== 路径配置 ==========
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        
        # GeoGLUE路径 - 新的数据结构包含6个子任务目录
        self.geoglue_path = os.path.join(self.data_dir, 'GeoGLUE')
        # 各子任务路径
        self.geoglue_tasks = {
            'GeoCPA': os.path.join(self.geoglue_path, 'GeoCPA'),
            'GeoEAG': os.path.join(self.geoglue_path, 'GeoEAG'),
            'GeoETA': os.path.join(self.geoglue_path, 'GeoETA'),
            'GeoTES-recall': os.path.join(self.geoglue_path, 'GeoTES-recall'),
            'GeoTES-rerank': os.path.join(self.geoglue_path, 'GeoTES-rerank'),
            'GeoWWC': os.path.join(self.geoglue_path, 'GeoWWC')
        }
        
        # 词典路径
        self.geo_lexicon_path = os.path.join(self.data_dir, 'processed', 'geo_lexicon.txt')
        
        # ========== 设备配置 ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ========== 模型配置 ==========
        # 本地模型路径（优先使用本地模型）
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.bert_model_name = os.path.join(self.model_dir, 'bert-base-chinese') if os.path.exists(os.path.join(self.model_dir, 'bert-base-chinese')) else 'bert-base-chinese'
        self.roberta_model_name = os.path.join(self.model_dir, 'chinese-roberta-wwm-ext') if os.path.exists(os.path.join(self.model_dir, 'chinese-roberta-wwm-ext')) else 'chinese-roberta-wwm-ext'
        
        # ========== 训练超参数 ==========
        self.batch_size = 8
        self.eval_batch_size = 16
        self.num_epochs = 30
        self.learning_rate = 1e-5  # 降低学习率，避免梯度爆炸
        self.weight_decay = 1e-4
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1
        self.warmup_steps = 0
        self.optimizer_type = 'AdamW'
        self.use_fp32 = False  # 启用混合精度训练，减少显存占用
        
        # ========== 早停配置 ==========
        self.patience = 5
        self.early_stopping_patience = 5
        self.early_stopping_metric = 'micro_f1'
        self.early_stopping_min_delta = 1e-4
        
        # ========== 模型架构设置 ==========
        self.model_variant = 'RoBERTaBiGRUAttentionCRF'
        self.use_crf = True
        self.use_lexicon = True
        self.middle_layer_type = 'bigru'  # None, 'bigru', 'attention', 'bigru_attention'
        
        # ========== 词汇增强设置 ==========
        self.lexicon_size = 542845
        self.lexicon_fusion_type = 'gate'
        self.use_lexicon_fusion_gate = True
        self.lexicon_dropout = 0.1
        
        # ========== 动态词典适配器设置 ==========
        self.use_dynamic_lexicon = True  # 动态词典总开关
        self.num_geo_levels = 7  # 行政层级数量（含0占位）
        self.geo_hierarchy_file = os.path.join(self.data_dir, 'processed', 'geo_lexicon_hierarchy.txt')
        self.geo_alias_file = os.path.join(self.data_dir, 'processed', 'geo_alias_map.json')
        self.geo_type_file = os.path.join(self.data_dir, 'processed', 'geo_type_map.json')
        self.lexicon_type = 'dynamic'  # 'static' or 'dynamic'
        
        # ========== 语义重塑机制设置 ==========
        self.use_semantic_reshaping = True  # 语义重塑层开关
        self.lambda_align = 0.1  # 对比对齐损失权重
        self.reshaping_attention_heads = 8  # 注意力头数
        
        # ========== 中间层设置 ==========
        self.middle_layer_hidden_size = 256
        self.middle_layer_dropout = 0.1
        
        # ========== 多任务损失权重 ==========
        self.ner_loss_weight = 1.0
        self.rel_loss_weight = 0.5
        self.event_loss_weight = 0.5
        
        # ========== 类别加权损失 ==========
        # 启用类别加权损失，用于处理类别不平衡
        self.use_class_weights = True
        # 类别权重：可以根据训练集的类别分布自定义设置
        # 默认使用逆频率权重
        self.ner_class_weights = None  # 如果为None，将自动计算
        self.rel_class_weights = None
        self.event_class_weights = None
        
        # ========== 序列长度 ==========
        self.max_seq_length = 128
        
        # ========== 数据加载 ==========
        self.num_workers = 0
        
        # ========== 标签映射 ==========
        self.ner_labels = ['O', 'B-GEO', 'I-GEO', 'B-ADMIN', 'I-ADMIN', 
                          'B-LANDMARK', 'I-LANDMARK', 'B-ORG', 'I-ORG', 
                          'B-PER', 'I-PER']
        self.ner_label2id = {label: idx for idx, label in enumerate(self.ner_labels)}
        self.ner_id2label = {idx: label for idx, label in enumerate(self.ner_labels)}
        
        # ========== 关系抽取标签 ==========
        self.rel_labels = [
            '无关系', '出生地', '出生日期', '国籍', '民族', '祖籍', '朝代',
            '身高', '体重', '妻子', '丈夫', '父亲', '母亲', '主演', '导演',
            '歌手', '作曲', '作词', '作者', '出版社', '连载网站', '出品公司',
            '成立日期', '总部地点', '海拔', '面积', '首都', '官方语言',
            '人口数量', '气候', '毕业院校', '专业'
        ]
        self.rel_label2id = {label: idx for idx, label in enumerate(self.rel_labels)}
        self.rel_id2label = {idx: label for idx, label in enumerate(self.rel_labels)}
        
        # ========== 事件抽取标签 ==========
        self.event_labels = [
            '无事件', '社交事件', '政治事件', '历史事件', '健康事件',
            '文化事件', '经济事件', '自然灾害'
        ]
        self.event_label2id = {label: idx for idx, label in enumerate(self.event_labels)}
        self.event_id2label = {idx: label for idx, label in enumerate(self.event_labels)}
        
        # ========== 12个模型变体配置 ==========
        self.ner_model_variants = [
            # ========== 单任务NER 9变体 ==========
            {
                "name": "BertNER",
                "model_name": self.bert_model_name,
                "use_crf": False,
                "use_lexicon": False,
                "middle_layer_type": None,
                "is_multitask": False
            },
            {
                "name": "BertBiLSTMCRF",
                "model_name": self.bert_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bilstm",
                "is_multitask": False
            },
            {
                "name": "BertBiGRUCRF",
                "model_name": self.bert_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bigru",
                "is_multitask": False
            },
            {
                "name": "BertBiLSTMAttention",
                "model_name": self.bert_model_name,
                "use_crf": False,
                "use_lexicon": False,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": False
            },
            {
                "name": "BaselineRoBERTa",
                "model_name": self.roberta_model_name,
                "use_crf": False,
                "use_lexicon": False,
                "middle_layer_type": None,
                "is_multitask": False
            },
            {
                "name": "RoBERTaBiLSTMCRF",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bilstm",
                "is_multitask": False
            },
            {
                "name": "RoBERTaBiLSTMAttention",
                "model_name": self.roberta_model_name,
                "use_crf": False,
                "use_lexicon": False,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": False
            },
            {
                "name": "RoBERTaBiLSTMAttentionCRF",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": False
            },
            {
                "name": "RoBERTaBiGRUAttentionCRF",    # 骨干模型
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bigru_attention",
                "is_multitask": False
            },
            # ========== 词汇增强单任务 ==========
            {
                "name": "Ours-Lex",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": True,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": False
            },
            # ========== 多任务（无词汇增强） ==========
            {
                "name": "MultiTask",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": True
            },
            # ========== 完整模型 ==========
            {
                "name": "Ours-Full",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": True,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": True
            },
            # ========== 动态词典单任务 ==========
            {
                "name": "Ours-DynamicLex",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": True,
                "lexicon_type": "dynamic",
                "middle_layer_type": "bilstm_attention",
                "is_multitask": False
            },
            # ========== 多任务+语义重塑 ==========
            {
                "name": "Ours-Reshaping",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": False,
                "use_reshaping": True,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": True
            },
            # ========== 最终完整模型（动态词典+多任务+语义重塑） ==========
            {
                "name": "Ours-Full",
                "model_name": self.roberta_model_name,
                "use_crf": True,
                "use_lexicon": True,
                "lexicon_type": "dynamic",
                "use_reshaping": True,
                "middle_layer_type": "bilstm_attention",
                "is_multitask": True
            }
        ]
        
        # ========== 案例分析文本配置 ==========
        self.case_texts = [
            "刚才朝阳区大望路发生严重追尾，堵了好几公里",
            "这周末琶洲有车展，有一起去的吗",
            "黄山风景区今天出现壮观云海",
            "帝都又下大雨了，三环已经成河了",
            "南京夫子庙附近人山人海，疑似有踩踏事件"
        ]
        
        # ========== 预测阈值配置 ==========
        self.rel_threshold = 0.5
        self.event_threshold = 0.5
        
        # ========== 创建目录 ==========
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def set_random_seed(self, seed):
        """设置随机种子的实例方法"""
        set_random_seed(seed)
    
    def get_model(self, variant, **kwargs):
        """根据模型变体名称获取模型实例
        
        Args:
            variant: 模型变体名称
            **kwargs: 额外参数（use_crf, use_lexicon等）
        
        Returns:
            模型实例
        """
        from models import create_ner_model, create_multitask_model
        
        # 检查是否为多任务模型
        if variant in ["MultiTask", "Ours-Full"]:
            return create_multitask_model(
                model_variant=variant,
                use_crf=kwargs.get('use_crf', True),
                use_lexicon=kwargs.get('use_lexicon', (variant == "Ours-Full")),
                lexicon_size=kwargs.get('lexicon_size', self.lexicon_size)
            )
        else:
            # 单任务NER模型
            return create_ner_model(
                model_variant=variant,
                use_crf=kwargs.get('use_crf', 'CRF' in variant),
                use_lexicon=kwargs.get('use_lexicon', False),
                lexicon_size=kwargs.get('lexicon_size', self.lexicon_size)
            )
    
    def get_model_config(self, variant):
        """获取指定模型变体的配置"""
        for config in self.ner_model_variants:
            if config['name'] == variant:
                return config
        return None
