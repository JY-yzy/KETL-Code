# models.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

try:
    from TorchCRF import CRF as CRFBase
except ImportError:
    CRFBase = None

from config import Config

cfg = Config()

# ========================== 词汇增强模块 ==========================
# Ours-StaticLex 默认消融配置（与基线一致：全特征开启、全局标量门控）
DEFAULT_ABLATION_CONFIG = {
    'use_level': True,        # 是否启用行政层级嵌入
    'use_type': True,         # 是否启用实体类型嵌入
    'use_coord': True,        # 是否启用坐标投影
    'use_mask': True,         # 是否启用位置 mask（未匹配位置保持原始表示）
    'use_global_gate': True,  # 是否启用全局标量融合门控
    'gate_type': 'global',    # 'global' = 标量门控；'token' = token 级门控
}


class LexiconAdapter(nn.Module):
    """
    地理知识感知的词汇增强适配器（Ours-StaticLex 主架构）

    设计要点：
    1. 多维知识：词汇嵌入 + 行政层级 + 实体类型 + 坐标投影 四维知识
    2. 门控特征融合：每个非词嵌入特征均带基于上下文的门控
    3. 位置 mask：未匹配词典位置保持原始 BERT 表示，防止噪声注入
    4. 全局标量融合门控：可学习的全局融合比例
    5. token 级门控（可选）：以 token 为粒度的融合门控，替代全局标量

    消融配置 ablation_config：
        use_level / use_type / use_coord / use_mask / use_global_gate / gate_type
    通过上述 flag 关闭各组件，构建 w/o Lex Emb / w/o Coord / w/o Mask / w/o Gate / Token Gate 变体。

    始终实例化所有子模块（参数名严格对齐 DynamicLexiconAdapter），保证基线检查点可 strict 加载到任意变体。
    """

    def __init__(self, hidden_size, lexicon_size, word_emb_dim=64,
                 num_levels=7, num_types=4,
                 fusion_type='gate', use_fusion_gate=True, dropout=0.1,
                 ablation_config: dict = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.word_emb_dim = word_emb_dim
        self.fusion_type = fusion_type
        self.num_levels = num_levels
        self.num_types = num_types

        # 合并消融配置（缺省走默认全开）
        cfg_ablation = dict(DEFAULT_ABLATION_CONFIG)
        if ablation_config:
            cfg_ablation.update(ablation_config)
        self.ablation = cfg_ablation
        self.use_level = cfg_ablation['use_level']
        self.use_type = cfg_ablation['use_type']
        self.use_coord = cfg_ablation['use_coord']
        self.use_mask = cfg_ablation['use_mask']
        self.use_global_gate = cfg_ablation['use_global_gate']
        self.gate_type = cfg_ablation['gate_type']
        # use_fusion_gate 字段保留向后兼容：等价于 use_global_gate
        self.use_fusion_gate = use_fusion_gate and self.use_global_gate

        # 词汇嵌入
        self.word_embed = nn.Embedding(lexicon_size + 1, word_emb_dim, padding_idx=0)

        # 行政层级嵌入（0=未知，使用可学习的未知层级嵌入）
        self.level_emb = nn.Embedding(num_levels, word_emb_dim)

        # 实体类型嵌入（0=未知, 1=GEO/ADMIN/LANDMARK, 2=ORG, 3=PER）
        self.type_emb = nn.Embedding(num_types, word_emb_dim)

        # 坐标嵌入层：将经纬度(2维)投影到 word_emb_dim 维
        self.coord_proj = nn.Sequential(
            nn.Linear(2, word_emb_dim),
            nn.GELU(),
            nn.LayerNorm(word_emb_dim)
        )

        # 可学习的零向量（用于无坐标实体）
        self.zero_coord_emb = nn.Parameter(torch.randn(1, 1, word_emb_dim) * 0.01)

        # 特征融合门控：基于 char_hidden（BERT 表示）学习特征贡献权重
        self.level_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.type_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 投影层：处理融合后的表示
        adapter_dropout = dropout
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + word_emb_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(adapter_dropout)
        )

        # 全局标量融合门控（初始 sigmoid≈0.5，处于梯度最大区域）
        # 始终创建以保证 state_dict key 完整（gate_type='token' 时本参数为 unexpected key，加载时 strict=False 忽略）
        self.fusion_gate = nn.Parameter(torch.ones(1) * 0.0)

        # token 级门控（仅 gate_type='token' 时使用）
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_size + word_emb_dim, hidden_size),
            nn.Sigmoid()
        ) if self.gate_type == 'token' else None

        print(f"[LexiconAdapter] 初始化完成: ablation={cfg_ablation}, dropout={adapter_dropout}")
        print(f"[LexiconAdapter] gate_type={self.gate_type}, token_gate={'on' if self.token_gate is not None else 'off'}")

    def forward(self, char_hidden, lexicon_indices,
                level_ids=None, type_ids=None, coord_features=None):
        """
        Args:
            char_hidden: (batch, seq_len, hidden_size) - BERT 隐藏状态
            lexicon_indices: (batch, seq_len) - 每个位置的词典 ID
            level_ids: (batch, seq_len) - 行政层级 ID（可选）
            type_ids: (batch, seq_len) - 类型 ID（可选）
            coord_features: (batch, seq_len, 2) - 坐标特征（纬度/经度归一化值）

        Returns:
            enhanced_hidden: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = char_hidden.shape

        # 索引范围检查（仅首次）
        if not hasattr(self, '_index_check_done'):
            idx_max = lexicon_indices.max().item()
            idx_min = lexicon_indices.min().item()
            print(f"[LexiconAdapter] Index range check: min={idx_min}, max={idx_max}, allowed=[0, {self.word_embed.num_embeddings-1}]")
            if idx_max >= self.word_embed.num_embeddings:
                print(f"[WARNING] lexicon_indices out of bounds! max={idx_max}")
            self._index_check_done = True

        # 防止索引越界导致 CUDA device-side assert
        lexicon_indices = torch.clamp(lexicon_indices, 0, self.word_embed.num_embeddings - 1)
        word_emb = self.word_embed(lexicon_indices)  # (B, L, word_emb_dim)

        # 计算位置 mask
        mask = (lexicon_indices != 0).float().unsqueeze(-1)  # (B, L, 1)

        # 全 0 直接返回原始表示
        if mask.sum().item() == 0:
            return char_hidden

        # ========== 门控特征融合 ==========

        # 层级特征融合
        if self.use_level and level_ids is not None:
            level_ids = torch.clamp(level_ids, 0, self.level_emb.num_embeddings - 1)
            level_emb = self.level_emb(level_ids)
            level_gate_val = self.level_gate(char_hidden)
            word_emb = word_emb + level_gate_val * level_emb

        # 类型特征融合
        if self.use_type and type_ids is not None:
            type_ids = torch.clamp(type_ids, 0, self.type_emb.num_embeddings - 1)
            type_emb = self.type_emb(type_ids)
            type_gate_val = self.type_gate(char_hidden)
            word_emb = word_emb + type_gate_val * type_emb

        # 坐标特征融合
        if self.use_coord and coord_features is not None:
            coord_mask = (coord_features != 0).any(dim=-1, keepdim=True).float()
            coord_emb = self.coord_proj(coord_features)
            zero_emb = self.zero_coord_emb.expand(batch_size, seq_len, -1)
            coord_emb = coord_mask * coord_emb + (1 - coord_mask) * zero_emb
            coord_gate_val = self.coord_gate(char_hidden)
            word_emb = word_emb + coord_gate_val * coord_emb

        # mask 应用：w/o Mask 变体不应用 mask（在所有位置注入知识）
        if self.use_mask:
            masked_word_emb = word_emb * mask
        else:
            masked_word_emb = word_emb

        # 融合到字符表示
        concat = torch.cat([char_hidden, masked_word_emb], dim=-1)
        projected = self.projection(concat)

        # 位置级 mask 残差
        if self.use_mask:
            enhanced = mask * projected + (1 - mask) * char_hidden
        else:
            enhanced = projected

        # gate 输出
        if self.gate_type == 'token' and self.token_gate is not None:
            # Token Gate 变体：token 级门控
            alpha_t = self.token_gate(concat)
            enhanced = alpha_t * enhanced + (1 - alpha_t) * char_hidden
        elif self.use_global_gate:
            # 全局标量门控
            alpha = torch.sigmoid(self.fusion_gate)
            enhanced = alpha * enhanced + (1 - alpha) * char_hidden
        # else: w/o Gate 变体，跳过此步（直接残差）

        return enhanced


# 历史命名保留，行为完全等价于 LexiconAdapter
# 旧的 DynamicLexiconAdapter 类已被 LexiconAdapter 统一替代，参数名兼容
DynamicLexiconAdapter = LexiconAdapter


# ========================== 语义重塑层 ==========================
class SemanticReshapingLayer(nn.Module):
    """
    联合语义重塑机制
    
    功能：通过任务感知的注意力池化和门控融合，重塑共享表示空间
    
    Args:
        hidden_size: 隐藏层维度
        num_tasks: 任务数量（默认3：NER、关系抽取、事件抽取）
        num_heads: 注意力头数
    """
    
    def __init__(self, hidden_size, num_tasks=3, num_heads=8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        
        # 任务感知查询向量
        self.task_queries = nn.Parameter(torch.randn(num_tasks, hidden_size))
        
        # 注意力层
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.1
        )
        
        # 门控融合层
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        print(f"[SemanticReshapingLayer] 初始化完成: num_tasks={num_tasks}, num_heads={num_heads}")
    
    def forward(self, H):
        """
        Args:
            H: (batch, seq_len, hidden) - 序列表示
        
        Returns:
            H_reshaped: (batch, seq_len, hidden) - 重塑后的序列表示
            task_contexts: (num_tasks, batch, hidden) - 各任务的全局上下文向量
        """
        batch_size, seq_len, hidden_size = H.shape
        
        # 计算任务感知的全局上下文向量
        task_contexts = []
        for task_idx in range(self.num_tasks):
            # 获取任务查询向量
            query = self.task_queries[task_idx].unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, 1, hidden)
            
            # 注意力池化
            context, _ = self.multihead_attn(query, H, H)  # (batch, 1, hidden)
            context = context.squeeze(1)  # (batch, hidden)
            task_contexts.append(context)
        
        # 门控融合
        H_reshaped = H.clone()
        for task_idx in range(self.num_tasks):
            context = task_contexts[task_idx].unsqueeze(1)  # (batch, 1, hidden)
            gate = self.gate_layers[task_idx](context)  # (batch, 1, 1)
            H_reshaped = H_reshaped + gate * H
        
        return H_reshaped, torch.stack(task_contexts, dim=0)


# ========================== 对比对齐损失 ==========================
def alignment_loss(ner_hidden, rel_hidden, event_hidden, entity_mask=None):
    """
    计算对比对齐损失，拉近同一实体在三个任务视图下的表示
    
    Args:
        ner_hidden: (batch, seq_len, hidden) - NER头之前的隐层表示
        rel_hidden: (batch, seq_len, hidden) - 关系头之前的隐层表示
        event_hidden: (batch, seq_len, hidden) - 事件头之前的隐层表示
        entity_mask: (batch, seq_len) - 实体位置掩码（可选）
    
    Returns:
        align_loss: 对齐损失值
    """
    if entity_mask is not None:
        # 只考虑实体位置
        mask = entity_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        ner_hidden = ner_hidden * mask
        rel_hidden = rel_hidden * mask
        event_hidden = event_hidden * mask
    
    # 计算三个视图表示之间的两两距离（均方误差）
    loss_ner_rel = torch.mean((ner_hidden - rel_hidden) ** 2)
    loss_ner_event = torch.mean((ner_hidden - event_hidden) ** 2)
    loss_rel_event = torch.mean((rel_hidden - event_hidden) ** 2)
    
    # 总对齐损失
    align_loss = (loss_ner_rel + loss_ner_event + loss_rel_event) / 3.0
    
    return align_loss


# ========================== 中间层构建器 ==========================
def build_middle_layer(input_size, layer_type, hidden_size=None, dropout=0.1, num_heads=8, num_layers=1):
    """动态构建中间层（BILSTM/BiGRU/Attention）"""
    if hidden_size is None:
        hidden_size = input_size
    
    if layer_type.lower() == 'bilstm':
        effective_dropout = dropout if num_layers > 1 else 0.0
        return nn.LSTM(
            input_size, hidden_size // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=effective_dropout
        )
    elif layer_type.lower() == 'bigru':
        effective_dropout = dropout if num_layers > 1 else 0.0
        return nn.GRU(
            input_size, hidden_size // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=effective_dropout
        )
    elif layer_type.lower() == 'attention':
        return nn.MultiheadAttention(
            hidden_size, num_heads,
            batch_first=True, dropout=dropout
        )
    elif layer_type.lower() == 'transformer':
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=1)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

# ========================== CRF层包装 ==========================
class CRF(nn.Module):
    """CRF层包装，支持GPU"""
    
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        if CRFBase is None:
            raise ImportError("请安装 torchcrf: pip install torchcrf")
        # TorchCRF API: __init__(num_labels, pad_idx=None, use_gpu=True)
        self.crf = CRFBase(num_tags)
    
    def forward(self, emissions, tags=None, mask=None, reduction='mean'):
        if tags is not None:
            # 确保 mask 是 BoolTensor，labels 是 LongTensor
            mask = mask.bool() if mask is not None else None
            tags = tags.long() if tags is not None else None
            
            # TorchCRF 不支持 -100 padding，需要创建临时标签
            # 将 -100 替换为 0（但这些位置会被 mask 忽略）
            safe_tags = tags.clone()
            safe_tags[safe_tags == -100] = 0
            
            # TorchCRF 返回对数似然，我们需要返回负对数似然作为损失
            log_likelihood = self.crf(emissions, safe_tags, mask=mask)
            
            if reduction == 'mean':
                return -log_likelihood.mean()
            elif reduction == 'sum':
                return -log_likelihood.sum()
            else:
                return -log_likelihood
        else:
            # TorchCRF 使用 viterbi_decode 方法进行解码
            mask = mask.bool() if mask is not None else None
            return self.crf.viterbi_decode(emissions, mask=mask)

# ========================== 核心NER模型 ==========================
class NERModel(nn.Module):
    """
    灵活的NER模型，支持多种架构组合
    
    支持的模型变体格式：
    - BERTBiLSTMCRF / RoBERTaBiLSTMCRF
    - BERTBiLSTMSoftmax / RoBERTaBiLSTMSoftmax
    - BERTBiLSTMAttentionCRF / RoBERTaBiLSTMAttentionCRF
    - BERTBiGRUCRF / RoBERTaBiGRUCRF
    - BERTAttentionCRF / RoBERTaAttentionCRF
    - BERTTransformerCRF / RoBERTaTransformerCRF
    
    参数：
        model_name: 预训练模型路径（支持 bert-base-chinese, chinese-roberta-wwm-ext）
        num_labels: NER标签数量
        model_variant: 模型变体名称
        use_crf: 是否使用CRF（True=CRF, False=Softmax）
        use_lexicon: 是否使用词典增强
        lexicon_size: 词典大小
        lexicon_config: 词典配置字典，包含:
            - word_emb_dim: 词嵌入维度
            - fusion_type: 融合类型
            - use_fusion_gate: 是否使用门控
    """
    
    def __init__(self, model_name, num_labels, model_variant='RoBERTaBiLSTMCRF',
                 use_crf=True, use_lexicon=False, lexicon_size=None,
                 lexicon_config=None, lexicon_type='static', use_reshaping=False,
                 ablation_config: dict = None):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.model_variant = model_variant
        self.use_crf = use_crf
        self.use_lexicon = use_lexicon
        self.lexicon_type = lexicon_type
        self.use_reshaping = use_reshaping
        self.lexicon_config = lexicon_config or {}
        self.ablation_config = ablation_config

        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = self.encoder.config.hidden_size

        self.lexicon_adapter = None
        if use_lexicon and lexicon_size is not None:
            # 优先使用 lexicon_config 中的配置，其次从全局 cfg 获取，最后用默认值
            word_emb_dim = self.lexicon_config.get(
                'word_emb_dim',
                getattr(cfg, 'lexicon_word_emb_dim', 128)
            )
            dropout = self.lexicon_config.get('dropout', 0.1)
            num_levels = self.lexicon_config.get('num_levels', cfg.num_geo_levels)
            fusion_type = self.lexicon_config.get('fusion_type', 'gate')
            use_fusion_gate = self.lexicon_config.get('use_fusion_gate', True)

            # 统一走 LexiconAdapter（Ours-StaticLex 主架构）
            # DynamicLexiconAdapter 已是 LexiconAdapter 别名，参数名严格对齐
            self.lexicon_adapter = LexiconAdapter(
                self.hidden_size,
                lexicon_size,
                word_emb_dim=word_emb_dim,
                num_levels=num_levels,
                num_types=4,
                fusion_type=fusion_type,
                use_fusion_gate=use_fusion_gate,
                dropout=dropout,
                ablation_config=ablation_config,
            )
            print(f"[NERModel] LexiconAdapter enabled: word_emb_dim={word_emb_dim}, "
                  f"num_levels={num_levels}, fusion_type={fusion_type}, "
                  f"ablation_config={ablation_config}")
        
        self.layers = nn.ModuleList()
        self.layer_types = []
        self._parse_and_build_layers()
        
        self.semantic_reshaping = None
        if use_reshaping:
            self.semantic_reshaping = SemanticReshapingLayer(
                self.hidden_size,
                num_tasks=3,
                num_heads=cfg.reshaping_attention_heads
            )
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        self.crf_layer = None
        if use_crf:
            self.crf_layer = CRF(num_labels)
    
    def _parse_and_build_layers(self):
        """解析模型变体名称，自动构建中间层"""
        variant = self.model_variant
        
        # 处理特殊名称映射
        special_mappings = {
            'Ours-Lex': 'RoBERTaBiLSTMAttentionCRF',
            'Ours-Full': 'RoBERTaBiLSTMAttentionCRF',
            'MultiTask': 'RoBERTaBiLSTMAttentionCRF',
            'BaselineRoBERTa': 'RoBERTa',
            'BaselineBERT': 'BERT',
            'Baseline': 'RoBERTa'
        }

        if variant in special_mappings:
            print(f"[NERModel] 特殊名称 {self.model_variant} 映射到 {special_mappings[variant]}")
            variant = special_mappings[variant]
        elif variant.startswith('Ours-StaticLex'):
            # Ours-StaticLex 全系列变体统一映射为 RoBERTa+BiLSTM+Attention+CRF
            # 消融差异通过 ablation_config 控制，不影响中间层结构
            print(f"[NERModel] Ours-StaticLex 变体 {self.model_variant} 映射到 RoBERTaBiLSTMAttentionCRF")
            variant = 'RoBERTaBiLSTMAttentionCRF'
        
        prefixes = ['BERT', 'RoBERTa', 'ERNIE', 'MacBERT', 'XLNet', 'Bert', 'Baseline']
        for prefix in prefixes:
            if variant.startswith(prefix):
                variant = variant[len(prefix):]
                break
        
        # 先处理 +DynamicLexicon 和 +Lexicon 后缀（这些是配置参数，不需要解析为层）
        # 注意：+DynamicLexicon 包含 +Lexicon，必须先检查较长的字符串
        if '+DynamicLexicon' in variant:
            variant = variant.replace('+DynamicLexicon', '')
        elif '+Lexicon' in variant:
            variant = variant.replace('+Lexicon', '')
        
        # 然后处理 CRF/Softmax/NER 后缀
        if variant.endswith('CRF'):
            variant = variant[:-3]
        if variant.endswith('Softmax'):
            variant = variant[:-7]
        if variant.endswith('NER'):
            variant = variant[:-3]
        
        layer_order = ['Transformer', 'BiLSTM', 'BiGRU', 'Attention']
        remaining = variant
        
        for layer_name in layer_order:
            if layer_name in remaining:
                layer_type = layer_name.lower()
                if layer_name == 'Transformer':
                    layer_type = 'transformer'
                
                layer = build_middle_layer(
                    self.hidden_size,
                    layer_type,
                    hidden_size=self.hidden_size
                )
                self.layers.append(layer)
                self.layer_types.append(layer_name)
                remaining = remaining.replace(layer_name, '', 1)
        
        if remaining.strip():
            print(f"警告: 未解析的部分: {remaining}")
    
    def _forward_encoder(self, input_ids, attention_mask):
        """编码器前向传播"""
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def _forward_middle_layers(self, hidden, attention_mask):
        """中间层前向传播"""
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type in ['BiLSTM', 'BiGRU']:
                hidden, _ = layer(hidden)
            elif layer_type in ['Attention', 'Transformer']:
                if layer_type == 'Attention':
                    hidden, _ = layer(
                        hidden, hidden, hidden,
                        key_padding_mask=~attention_mask.bool()
                    )
                else:
                    hidden = layer(hidden, src_key_padding_mask=~attention_mask.bool())
            hidden = self.dropout(hidden)
        return hidden
    
    def forward(self, input_ids, attention_mask, lexicon_indices=None,
                lexicon_levels=None, lexicon_types=None, lexicon_coords=None,
                ner_tags=None, return_hidden=False):
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            lexicon_indices: (batch, seq_len) - 可选，词典ID
            lexicon_levels: (batch, seq_len) - 可选，行政层级ID
            lexicon_types: (batch, seq_len) - 可选，类型ID
            lexicon_coords: (batch, seq_len, 2) - 可选，坐标特征（纬度/经度）
            ner_tags: (batch, seq_len) - 可选，训练时提供
            return_hidden: 是否返回隐层表示
        
        Returns:
            训练模式: (loss, logits, hidden, None)
            推理模式: (predictions, logits, hidden, None)
        """
        hidden = self._forward_encoder(input_ids, attention_mask)
        
        if self.lexicon_adapter is not None and lexicon_indices is not None:
            # 统一接口：static / dynamic 同一调用路径
            # LexiconAdapter.forward 接收 level_ids/type_ids/coord_features，根据 ablation_config 决定是否使用
            hidden = self.lexicon_adapter(
                hidden, lexicon_indices,
                level_ids=lexicon_levels,
                type_ids=lexicon_types,
                coord_features=lexicon_coords,
            )
        
        hidden = self._forward_middle_layers(hidden, attention_mask)
        
        reshaped_hidden = None
        if self.semantic_reshaping is not None:
            hidden, _ = self.semantic_reshaping(hidden)
            reshaped_hidden = hidden
        
        logits = self.classifier(hidden)
        
        if ner_tags is not None:
            loss = self._compute_loss(logits, ner_tags, attention_mask)
            return loss, logits, hidden if return_hidden else None, reshaped_hidden if (return_hidden and reshaped_hidden is not None) else None
        else:
            predictions = self._decode(logits, attention_mask)
            return predictions, logits, hidden if return_hidden else None, None
    
    def _compute_loss(self, logits, tags, mask):
        """计算损失，支持类别加权（CRF + 辅助CE损失）"""
        if self.use_crf and self.crf_layer is not None:
            crf_mask = mask.bool().clone()
            crf_mask[:, 0] = True

            # 限制 logits 范围，防止数值溢出导致 CRF 损失异常
            logits_clamped = torch.clamp(logits, min=-50, max=50)

            loss = self.crf_layer(logits_clamped, tags, mask=crf_mask)

            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                # 如果 CRF 损失无效，回退到 CrossEntropyLoss
                return self._compute_ce_loss(logits, tags, mask)

            # 辅助CE损失：让类别权重通过CE损失间接影响CRF训练
            # TorchCRF不支持class weights，通过加权CE辅助损失来引入类别权重
            # 将权重从0.3提高到1.0，使类别权重能够实际生效
            # CRF损失≈77 vs CE损失≈2.4，0.3×CE几乎被淹没，需要更高权重
            if cfg.use_class_weights:
                ce_loss = self._compute_ce_loss(logits, tags, mask)
                if not (torch.isnan(ce_loss) or torch.isinf(ce_loss)):
                    loss = loss + 1.0 * ce_loss

            return loss
        else:
            return self._compute_ce_loss(logits, tags, mask)
    
    def _compute_ce_loss(self, logits, tags, mask):
        """计算 CrossEntropy 损失（作为 CRF 的备选）"""
        logits_clamped = torch.clamp(logits, min=-50, max=50)
        
        # 构建损失函数，支持类别权重
        class_weights = None
        if cfg.use_class_weights:
            if hasattr(cfg, 'ner_class_weights') and cfg.ner_class_weights is not None:
                class_weights = torch.tensor(cfg.ner_class_weights, dtype=torch.float32, device=logits.device)
        
        if class_weights is not None:
            min_w, max_w = getattr(cfg, 'class_weight_min', 0.1), getattr(cfg, 'class_weight_max', 10.0)
            class_weights = torch.clamp(class_weights, min=min_w, max=max_w)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        active_loss = mask.view(-1) == 1
        active_logits = logits_clamped.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss,
            tags.view(-1),
            torch.tensor(loss_fn.ignore_index).type_as(tags)
        )
        
        loss = loss_fn(active_logits, active_labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=logits.device)
        
        return loss
    
    def _decode(self, logits, mask):
        """解码预测"""
        if self.use_crf and self.crf_layer is not None:
            predictions = self.crf_layer(logits, mask=mask.bool())
            return predictions
        else:
            return torch.argmax(logits, dim=-1)

# ========================== 多任务模型 ==========================
class MultiTaskModel(NERModel):
    """
    多任务模型（NER + 关系抽取 + 事件抽取）
    
    继承NERModel，共享BERT编码器和中间层，添加任务特定的分类头：
    - NER分类头（继承自NERModel）
    - 关系双仿射分类头
    - 事件序列分类头
    
    Args:
        model_name: 预训练模型路径
        num_ner_labels: NER标签数量
        num_rel_labels: 关系标签数量
        num_event_types: 事件类型数量
        model_variant: 模型变体名称
        use_crf: 是否使用CRF
        use_lexicon: 是否使用词典增强
        lexicon_size: 词典大小
        lexicon_config: 词典配置
    """
    
    def __init__(self, model_name, num_ner_labels, num_rel_labels, num_event_types,
                 model_variant='RoBERTaBiLSTMAttention', use_crf=True,
                 use_lexicon=False, lexicon_size=None, lexicon_config=None,
                 lexicon_type='static', use_reshaping=False,
                 ablation_config: dict = None):
        super().__init__(model_name, num_ner_labels, model_variant,
                         use_crf=use_crf, use_lexicon=use_lexicon,
                         lexicon_size=lexicon_size, lexicon_config=lexicon_config,
                         lexicon_type=lexicon_type, use_reshaping=use_reshaping,
                         ablation_config=ablation_config)
        
        self.num_rel_labels = num_rel_labels
        self.num_event_types = num_event_types
        self.use_reshaping = use_reshaping
        
        # 关系抽取头：双仿射层
        self.rel_classifier = nn.Bilinear(
            self.hidden_size, self.hidden_size, num_rel_labels
        )
        
        self.event_classifier = nn.Linear(self.hidden_size, num_event_types)
    
    def forward(self, input_ids, attention_mask, lexicon_indices=None,
                lexicon_levels=None, lexicon_types=None,
                ner_tags=None, rel_labels=None, event_labels=None, return_hidden=False):
        """
        多任务前向传播
        
        Args:
            input_ids: (batch, seq_len) - 输入token ID
            attention_mask: (batch, seq_len) - 注意力掩码
            lexicon_indices: (batch, seq_len) - 词典匹配索引（可选）
            lexicon_levels: (batch, seq_len) - 行政层级ID（可选）
            lexicon_types: (batch, seq_len) - 类型ID（可选）
            ner_tags: (batch, seq_len) - NER标签（训练时提供）
            rel_labels: (batch, seq_len, seq_len) - 关系标签（训练时提供）
            event_labels: (batch, seq_len, num_event_types) - 事件标签（训练时提供）
            return_hidden: 是否返回隐藏层表示
        
        Returns:
            训练模式: (total_loss, ner_logits, rel_logits, event_logits, hidden, align_loss)
            推理模式: (ner_preds, rel_preds, event_preds, ner_logits, rel_logits, event_logits, hidden)
        """
        if ner_tags is not None:
            ner_loss, ner_logits, hidden, _ = super().forward(
                input_ids, attention_mask,
                lexicon_indices=lexicon_indices,
                lexicon_levels=lexicon_levels,
                lexicon_types=lexicon_types,
                ner_tags=ner_tags,
                return_hidden=True
            )
            
            if torch.isnan(ner_loss) or torch.isinf(ner_loss):
                ner_loss = torch.tensor(0.0, device=hidden.device)
            
            rel_logits = self._compute_rel_logits(hidden)
            rel_loss = self._compute_rel_loss(rel_logits, rel_labels, attention_mask)
            
            event_logits = self.event_classifier(hidden)
            event_loss = self._compute_event_loss(event_logits, event_labels, attention_mask)
            
            total_loss = (
                cfg.ner_loss_weight * ner_loss +
                cfg.rel_loss_weight * rel_loss +
                cfg.event_loss_weight * event_loss
            )
            
            align_loss = torch.tensor(0.0, device=hidden.device)
            if self.use_reshaping and cfg.lambda_align > 0:
                entity_mask = (ner_tags != 0) & (ner_tags != -100)
                align_loss = alignment_loss(hidden, hidden, hidden, entity_mask)
                total_loss = total_loss + cfg.lambda_align * align_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if not (torch.isnan(ner_loss) or torch.isinf(ner_loss)):
                    total_loss = cfg.ner_loss_weight * ner_loss
                else:
                    total_loss = torch.tensor(0.0, device=hidden.device)
            
            return total_loss, ner_logits, rel_logits, event_logits, hidden, align_loss
        else:
            ner_preds, ner_logits, hidden, _ = super().forward(
                input_ids, attention_mask,
                lexicon_indices=lexicon_indices,
                lexicon_levels=lexicon_levels,
                lexicon_types=lexicon_types,
                ner_tags=None,
                return_hidden=True
            )
            
            rel_logits = self._compute_rel_logits(hidden)
            rel_preds = self._decode_rel(rel_logits)
            
            event_logits = self.event_classifier(hidden)
            event_preds = self._decode_event(event_logits)
            
            return ner_preds, rel_preds, event_preds, ner_logits, rel_logits, event_logits, hidden
    
    def _compute_rel_logits(self, hidden):
        """计算关系 logits（双仿射注意力机制）"""
        batch_size, seq_len, hidden_size = hidden.shape
        
        W = self.rel_classifier.weight
        bias = self.rel_classifier.bias
        
        # 使用 einsum 直接计算双仿射
        # hidden: (batch, seq_len, hidden) -> bik
        # W: (num_rel, hidden, hidden) -> rkl  
        # 结果: (batch, seq_len, seq_len, num_rel) -> bijr
        rel_logits = torch.einsum('bik,rkl,bjl->bijr', hidden, W, hidden)
        
        # 限制 logits 范围，防止数值溢出
        rel_logits = torch.clamp(rel_logits, min=-50, max=50)
        
        rel_logits = rel_logits + bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
        return rel_logits
    
    def _decode_rel(self, rel_logits):
        """解码关系预测（取最大概率）"""
        return torch.argmax(rel_logits, dim=-1)
    
    def _decode_event(self, event_logits):
        """解码事件预测（取最大概率）"""
        return torch.argmax(event_logits, dim=-1)
    
    def _compute_rel_loss(self, rel_logits, rel_labels, attention_mask):
        """计算关系损失，支持类别加权"""
        if rel_labels is None:
            return torch.tensor(0.0, device=rel_logits.device)
        
        # 检查是否所有标签都是忽略值（-100）
        if (rel_labels == -100).all():
            return torch.tensor(0.0, device=rel_logits.device)
        
        # 限制 logits 范围，防止数值溢出
        rel_logits_clamped = torch.clamp(rel_logits, min=-50, max=50)
        
        # 构建损失函数，支持类别权重
        if cfg.use_class_weights and cfg.rel_class_weights is not None:
            class_weights = torch.tensor(cfg.rel_class_weights, dtype=torch.float32, device=rel_logits.device)
            # 限制权重范围，避免极端值
            class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        rel_loss = loss_fn(
            rel_logits_clamped.permute(0, 3, 1, 2),
            rel_labels
        )
        
        # 检查损失是否有效
        if torch.isnan(rel_loss) or torch.isinf(rel_loss):
            return torch.tensor(0.0, device=rel_logits.device)
        
        return rel_loss
    
    def _compute_event_loss(self, event_logits, event_labels, attention_mask):
        """计算事件损失，支持类别加权"""
        if event_labels is None:
            return torch.tensor(0.0, device=event_logits.device)
        
        # 限制 logits 范围，防止数值溢出导致 NaN
        event_logits_clamped = torch.clamp(event_logits, min=-50, max=50)
        
        # 构建损失函数，支持类别权重
        if cfg.use_class_weights and cfg.event_class_weights is not None:
            pos_weight = torch.tensor(cfg.event_class_weights, dtype=torch.float32, device=event_logits.device)
            # 限制权重范围，避免极端值
            pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        loss = loss_fn(event_logits_clamped, event_labels)
        
        # 使用 attention_mask 过滤 padding 位置
        mask = attention_mask.unsqueeze(-1).float()
        masked_loss = loss * mask
        
        # 防止除零错误
        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=event_logits.device)
        
        event_loss = masked_loss.sum() / mask_sum
        
        # 检查损失是否有效
        if torch.isnan(event_loss) or torch.isinf(event_loss):
            return torch.tensor(0.0, device=event_logits.device)
        
        return event_loss

# ========================== 模型工厂函数 ==========================
def create_ner_model(model_variant='RoBERTaBiLSTMCRF', use_crf=True,
                    use_lexicon=False, lexicon_size=None, lexicon_config=None,
                    lexicon_type='static', use_reshaping=False,
                    ablation_config: dict = None):
    """工厂函数：创建NER模型

    Args:
        model_variant: 模型变体名称
        use_crf: 是否使用CRF
        use_lexicon: 是否使用词典增强
        lexicon_size: 词典大小
        lexicon_config: 词典配置字典
        lexicon_type: 词典类型 ('static' or 'dynamic')，仅作日志标记，结构统一
        use_reshaping: 是否使用语义重塑
        ablation_config: 消融配置字典（Ours-StaticLex 变体使用）

    Returns:
        NERModel实例
    """
    if model_variant.startswith('Bert'):
        model_name = cfg.bert_model_name
    elif model_variant.startswith('RoBERTa') or model_variant.startswith('Ours') or model_variant.startswith('MultiTask'):
        model_name = cfg.roberta_model_name
    else:
        model_name = cfg.base_model

    num_labels = cfg.num_ner_labels

    return NERModel(
        model_name=model_name,
        num_labels=num_labels,
        model_variant=model_variant,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=lexicon_size,
        lexicon_config=lexicon_config,
        lexicon_type=lexicon_type,
        use_reshaping=use_reshaping,
        ablation_config=ablation_config,
    )

def create_multitask_model(model_variant='Ours-Full', use_crf=True, use_lexicon=False, lexicon_size=None, lexicon_config=None,
                           lexicon_type='static', use_reshaping=False,
                           ablation_config: dict = None):
    """工厂函数：创建多任务模型

    Args:
        model_variant: 模型变体名称（用于推断编码器类型）
        use_crf: 是否使用CRF
        use_lexicon: 是否使用词典增强
        lexicon_size: 词典大小
        lexicon_config: 词典配置字典
        lexicon_type: 词典类型 ('static' or 'dynamic')，仅作日志标记，结构统一
        use_reshaping: 是否使用语义重塑
        ablation_config: 消融配置字典（Ours-StaticLex 变体使用）

    Returns:
        MultiTaskModel实例
    """
    if model_variant.startswith('Bert'):
        model_name = cfg.bert_model_name
    elif model_variant.startswith('RoBERTa') or model_variant.startswith('Ours') or model_variant.startswith('MultiTask'):
        model_name = cfg.roberta_model_name
    else:
        model_name = cfg.base_model

    return MultiTaskModel(
        model_name=model_name,
        num_ner_labels=cfg.num_ner_labels,
        num_rel_labels=cfg.num_rel_labels,
        num_event_types=cfg.num_event_types,
        use_crf=use_crf,
        use_lexicon=use_lexicon,
        lexicon_size=lexicon_size,
        lexicon_config=lexicon_config,
        lexicon_type=lexicon_type,
        use_reshaping=use_reshaping,
        ablation_config=ablation_config,
    )

def create_model_from_config(model_config):
    """从配置字典创建模型

    Args:
        model_config: 模型配置字典，包含 name, model_name, use_crf, use_lexicon,
                      middle_layer_type, is_multitask, lexicon_type, use_reshaping,
                      ablation_config (可选)

    Returns:
        模型实例
    """
    model_name = model_config['model_name']
    use_crf = model_config.get('use_crf', True)
    use_lexicon = model_config.get('use_lexicon', False)
    is_multitask = model_config.get('is_multitask', False)
    lexicon_type = model_config.get('lexicon_type', 'static')
    use_reshaping = model_config.get('use_reshaping', False)
    ablation_config = model_config.get('ablation_config', None)

    if is_multitask:
        return create_multitask_model(
            model_variant=model_config['name'],
            use_crf=use_crf,
            use_lexicon=use_lexicon,
            lexicon_type=lexicon_type,
            use_reshaping=use_reshaping,
            ablation_config=ablation_config,
        )
    else:
        return create_ner_model(
            model_variant=model_config['name'],
            use_crf=use_crf,
            use_lexicon=use_lexicon,
            lexicon_type=lexicon_type,
            use_reshaping=use_reshaping,
            ablation_config=ablation_config,
        )
