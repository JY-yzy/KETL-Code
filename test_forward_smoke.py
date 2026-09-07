"""
StaticLex 前向传播烟雾测试：
对 6 个 ablation 变体构造模型，做一次 forward，确认输出形状正确、无异常
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config_ner import cfg_ner
from models import NERModel
from data_utils import create_lexicon_matcher


def main():
    # 先同步 lexicon_size
    print("[Smoke] Syncing lexicon_size...")
    create_lexicon_matcher(cfg_ner, load_ownthink=cfg_ner.load_ownthink)
    print(f"[Smoke] cfg.lexicon_size = {cfg_ner.lexicon_size}")

    device = torch.device('cpu')  # CPU 即可，只测形状
    cfg_ner.device = device

    ablations = [
        ('full', None),
        ('lexonly', {'use_level': False, 'use_type': False, 'use_coord': False, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nocoord', {'use_level': True, 'use_type': True, 'use_coord': False, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nomask', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': False, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nogate', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': True, 'use_global_gate': False, 'gate_type': 'global'}),
        ('tokengate', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'token'}),
    ]

    B, L = 2, 16
    input_ids = torch.randint(1, 1000, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = torch.zeros(B, L, dtype=torch.long)
    # 模拟词典特征
    lexicon_indices = torch.randint(0, cfg_ner.lexicon_size, (B, L))
    lexicon_levels = torch.randint(0, 7, (B, L))
    lexicon_types = torch.randint(0, 4, (B, L))
    lexicon_coords = torch.rand(B, L, 2)

    all_pass = True
    for name, abl_cfg in ablations:
        print(f"\n[Smoke] === {name} ===")
        try:
            model = NERModel(
                model_name=cfg_ner.roberta_model_name,
                num_labels=len(cfg_ner.ner_label2id),
                model_variant='Ours-StaticLex_' + name,
                use_crf=True, use_lexicon=True,
                lexicon_size=cfg_ner.lexicon_size,
                lexicon_type='static',
                use_reshaping=False,
                ablation_config=abl_cfg,
                lexicon_config={
                    'word_emb_dim': getattr(cfg_ner, 'lexicon_word_emb_dim', 64),
                    'fusion_type': cfg_ner.lexicon_fusion_type,
                    'use_fusion_gate': cfg_ner.use_lexicon_fusion_gate,
                    'dropout': cfg_ner.lexicon_dropout,
                    'num_levels': cfg_ner.num_geo_levels,
                }
            ).to(device)

            # 训练模式 forward
            model.train()
            loss, logits, hidden, _ = model(
                input_ids=input_ids, attention_mask=attention_mask,
                lexicon_indices=lexicon_indices,
                lexicon_levels=lexicon_levels,
                lexicon_types=lexicon_types,
                lexicon_coords=lexicon_coords,
                ner_tags=labels,
            )
            logits_shape = logits.shape if logits is not None else 'None (CRF mode)'
            hidden_shape = hidden.shape if hidden is not None else 'None'
            print(f"  train: loss={loss.item():.4f}, logits={logits_shape}, hidden={hidden_shape}")

            # 推理模式 forward
            model.eval()
            with torch.no_grad():
                predictions, _, _, _ = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    lexicon_indices=lexicon_indices,
                    lexicon_levels=lexicon_levels,
                    lexicon_types=lexicon_types,
                    lexicon_coords=lexicon_coords,
                )
            n_pred = len(predictions) if isinstance(predictions, list) else 1
            print(f"  eval: predictions={n_pred} seqs")
            print(f"  [PASS] {name}")
            del model
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback; traceback.print_exc()
            all_pass = False

    print(f"\n[Smoke] Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == '__main__':
    main()
