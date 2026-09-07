"""
StaticLex 检查点复用校验：
1) 加载 DynamicLexicon 检查点的 state_dict
2) 对 6 个 ablation 变体，构造 NERModel，strict=False 加载，报告 missing/unexpected
3) 验证：所有变体的 missing/unexpected keys 仅限于适配器新增/移除子模块（不影响编码器加载）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config_ner import cfg_ner
from models import NERModel


def main():
    ckpt_path = os.path.join(cfg_ner.model_save_dir, cfg_ner.staticlex_dynamiclex_ckpt_name)
    print(f"[Test] Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Checkpoint not found")
        return False

    state = torch.load(ckpt_path, map_location='cpu')
    print(f"[Test] Checkpoint keys: {len(state)}")

    # 提取 checkpoint 中 lexicon_adapter 相关键
    lex_keys_in_ckpt = [k for k in state.keys() if 'lexicon_adapter' in k]
    print(f"[Test] lexicon_adapter keys in checkpoint: {len(lex_keys_in_ckpt)}")
    for k in lex_keys_in_ckpt[:5]:
        print(f"  - {k}: shape={state[k].shape if hasattr(state[k], 'shape') else type(state[k])}")
    print("  ...")
    for k in lex_keys_in_ckpt[-3:]:
        print(f"  - {k}: shape={state[k].shape if hasattr(state[k], 'shape') else type(state[k])}")

    # 同步 lexicon_size：通过 create_lexicon_matcher 让 cfg.lexicon_size 与实际词典一致
    print(f"\n[Test] Syncing cfg.lexicon_size via create_lexicon_matcher (load_ownthink={cfg_ner.load_ownthink})...")
    from data_utils import create_lexicon_matcher
    create_lexicon_matcher(cfg_ner, load_ownthink=cfg_ner.load_ownthink)
    print(f"[Test] After sync: cfg.lexicon_size = {cfg_ner.lexicon_size}")

    # 6 个 ablation 配置
    ablations = [
        ('full', None),  # 默认全开
        ('lexonly', {'use_level': False, 'use_type': False, 'use_coord': False, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nocoord', {'use_level': True, 'use_type': True, 'use_coord': False, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nomask', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': False, 'use_global_gate': True, 'gate_type': 'global'}),
        ('nogate', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': True, 'use_global_gate': False, 'gate_type': 'global'}),
        ('tokengate', {'use_level': True, 'use_type': True, 'use_coord': True, 'use_mask': True, 'use_global_gate': True, 'gate_type': 'token'}),
    ]

    results = []
    for name, abl_cfg in ablations:
        print(f"\n[Test] === Ablation: {name} (cfg={abl_cfg}) ===")
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
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            # 过滤 lexicon_adapter 相关键
            miss_adapter = [k for k in missing if 'lexicon_adapter' in k]
            miss_other = [k for k in missing if 'lexicon_adapter' not in k]
            unexp_adapter = [k for k in unexpected if 'lexicon_adapter' in k]
            unexp_other = [k for k in unexpected if 'lexicon_adapter' not in k]

            status = 'OK' if not miss_other and not unexp_other else 'WARN'
            print(f"  Missing: total={len(missing)}, adapter={len(miss_adapter)}, other={len(miss_other)}")
            print(f"  Unexpected: total={len(unexpected)}, adapter={len(unexp_adapter)}, other={len(unexp_other)}")
            if miss_other:
                print(f"  [WARN] Non-adapter missing keys:")
                for k in miss_other[:5]:
                    print(f"    - {k}")
            if unexp_other:
                print(f"  [WARN] Non-adapter unexpected keys (count={len(unexp_other)}):")
                for k in unexp_other[:15]:
                    print(f"    - {k}")
            results.append({'name': name, 'status': status, 'miss_adapter': miss_adapter, 'unexp_adapter': unexp_adapter, 'miss_other': miss_other, 'unexp_other': unexp_other})
            del model
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback; traceback.print_exc()
            results.append({'name': name, 'status': 'FAIL', 'error': str(e)})

    # 总结
    print(f"\n{'='*70}")
    print(f"[Test] Checkpoint Reuse Validation Summary")
    print(f"{'='*70}")
    print(f"{'Variant':<15} {'Status':<8} {'MissAdapt':<10} {'UnexpAdapt':<10} {'MissOther':<10} {'UnexpOther':<10}")
    for r in results:
        print(f"{r['name']:<15} {r['status']:<8} "
              f"{len(r.get('miss_adapter', [])):<10} {len(r.get('unexp_adapter', [])):<10} "
              f"{len(r.get('miss_other', [])):<10} {len(r.get('unexp_other', [])):<10}")
    print(f"\nVerdict: 所有变体的编码器/CRF/中间层权重均能加载（miss_other/unexp_other 为空即正确）")
    print(f"        适配器层的 missing/unexpected 是预期行为：")
    print(f"        - full/nocoord/nomask/nogate: 应该无 missing/unexpected（结构一致）")
    print(f"        - lexonly: 无 missing/unexpected（仅 forward 行为不同，参数都还在）")
    print(f"        - tokengate: token_gate.* 应在 missing（新增模块），fusion_gate 在 unexpected（不再使用）")

    all_ok = all(r['status'] == 'OK' for r in results)
    print(f"\nOverall: {'PASS' if all_ok else 'WARN'}")
    return all_ok


if __name__ == '__main__':
    main()
