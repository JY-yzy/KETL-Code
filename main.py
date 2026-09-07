#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识增强的中文地理信息提取 - 主入口文件

核心训练任务：
1. 数据预处理
2. 单任务NER实验（9个模型变体）
3. 词汇增强消融实验
4. 多任务学习实验（NER+RE+EE）
5. 完整模型（Ours-Full）实验
6. 消融实验完整组合
7. 频率分层评估
8. GeoGLUE零样本评测
9. 四阶段课程学习
10. 端到端综合评测（模式12）
11. 真实文本案例分析（模式13）
12. 可视化
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量防止重复的OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入配置
from config import Config
cfg = Config()

# ========== 运行模式设置 ==========
RUN_MODE = 'experiment'  # 'test' 或 'experiment'
TEST_SAMPLE_SIZE = 100  # 测试模式下的样本量

# ========== 超参数设置 ==========
MODEL_NAME = cfg.roberta_model_name
USE_CRF = True
USE_LEXICON = True
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5


def print_environment_info():
    """打印环境信息（只在主入口调用一次）"""
    print("="*60)
    print("环境信息")
    print("="*60)
    print(f"Python版本: {sys.version.split()[0]}")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch未安装")
    print(f"工作目录: {os.getcwd()}")
    print("="*60)


def run_preprocess():
    """运行数据预处理"""
    from data_preprocess import main as preprocess_main
    preprocess_main()


def run_train(args):
    """运行训练任务"""
    import train
    train.main(args)


def run_visualize(args):
    """运行可视化"""
    from visualization import main as vis_main
    vis_main(args)


def run_end2end_eval(checkpoint_path=None):
    """运行端到端综合评测（模式12）"""
    from evaluation_end2end import run_end2end_eval as eval_func
    eval_func(checkpoint_path=checkpoint_path)


def run_case_study(checkpoint_path=None):
    """运行真实文本案例分析（模式13）"""
    from evaluation_end2end import run_case_study as case_study_func
    case_study_func(checkpoint_path=checkpoint_path)


def run_all():
    """
    直接运行所有核心训练任务（无需命令行参数）
    包含：单任务NER实验、多任务学习实验、消融实验、频率分层评估、GeoGLUE零样本评测
    """
    print("="*60)
    print("直接运行完整流程（无需命令行参数）")
    print("="*60)
    
    # 显示当前运行模式
    print(f"运行模式: {'[TEST] 测试模式' if RUN_MODE == 'test' else '[EXP] 实验模式'}")
    if RUN_MODE == 'test':
        print(f"测试样本量: {TEST_SAMPLE_SIZE}")
    
    # ========== 阶段1：数据预处理 ==========
    print("\n" + "="*60)
    print("[阶段1/8] 数据预处理")
    run_preprocess()
    
    # ========== 阶段2：单任务NER实验（9个模型变体） ==========
    print("\n" + "="*60)
    print("[阶段2/8] 单任务NER实验（9个模型变体）")
    train_args = {
        'mode': 'run_ner_experiments',
        'model_name': MODEL_NAME,
        'use_crf': USE_CRF,
        'use_lexicon': USE_LEXICON,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'config': None,
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段3：多任务学习实验（NER+RE+EE） ==========
    print("\n" + "="*60)
    print("[阶段3/8] 多任务学习实验（NER+RE+EE）")
    train_args = {
        'mode': 'run_multitask_experiments',
        'model_name': MODEL_NAME,
        'use_crf': USE_CRF,
        'use_lexicon': USE_LEXICON,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'loss_weights': '1.0,1.0,1.0',
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段4：消融实验完整组合 ==========
    print("\n" + "="*60)
    print("[阶段4/8] 消融实验完整组合")
    train_args = {
        'mode': 'run_ablation_study',
        'model_name': cfg.roberta_model_name,
        'use_crf': True,
        'use_lexicon': True,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段5：频率分层评估 ==========
    print("\n" + "="*60)
    print("[阶段5/8] 频率分层评估")
    train_args = {
        'mode': 'run_frequency_analysis',
        'model_name': cfg.roberta_model_name,
        'use_crf': True,
        'use_lexicon': True,
        'batch_size': BATCH_SIZE,
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段6：词汇增强消融实验（单独测试） ==========
    print("\n" + "="*60)
    print("[阶段6/8] 词汇增强消融实验")
    train_args = {
        'mode': 'run_lexicon_ablation',
        'model_name': cfg.roberta_model_name,
        'use_crf': True,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段7：完整模型（Ours-Full）实验 ==========
    print("\n" + "="*60)
    print("[阶段7/8] 完整模型（Ours-Full）实验")
    train_args = {
        'mode': 'train_multitask',
        'model_name': cfg.roberta_model_name,
        'use_crf': True,
        'use_lexicon': True,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS * 2,  # 完整模型训练更多轮
        'learning_rate': LEARNING_RATE,
        'run_mode': RUN_MODE,
        'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
    }
    run_train(train_args)
    
    # ========== 阶段8：GeoGLUE零样本评测 ==========
    print("\n" + "="*60)
    print("[阶段8/8] GeoGLUE零样本评测")
    from geoglue_evaluation import run_geoglue_zero_shot_evaluation
    geoglue_results = run_geoglue_zero_shot_evaluation(
        model_name=cfg.roberta_model_name,
        split='dev'
    )
    
    # ========== 可视化 ==========
    print("\n" + "="*60)
    print("[可视化] 生成图表")
    
    # 绘制NER结果图
    ner_path = 'results/results_ner.csv'
    if os.path.exists(ner_path):
        vis_args = {'plot_type': 'ner_results', 'plot_input': ner_path}
        run_visualize(vis_args)
    
    # 绘制消融实验图
    ablation_path = 'results/ablation_study_results.csv'
    if os.path.exists(ablation_path):
        vis_args = {'plot_type': 'ablation_study', 'plot_input': ablation_path}
        run_visualize(vis_args)
    
    # 绘制频率分层评估图
    freq_path = 'results/frequency_analysis.csv'
    if os.path.exists(freq_path):
        vis_args = {'plot_type': 'frequency_bar', 'plot_input': freq_path}
        run_visualize(vis_args)
    
    # 绘制训练曲线
    train_curve_path = 'results/multitask_epoch_metrics.csv'
    if os.path.exists(train_curve_path):
        vis_args = {'plot_type': 'training_curves', 'plot_input': train_curve_path}
        run_visualize(vis_args)
    
    print("\n" + "="*60)
    print("所有核心训练任务执行完成！")


def run_curriculum_learning():
    """
    四阶段课程学习
    
    阶段一：在百科NER数据（CLUENER+MSRA）上训练骨干模型（RoBERTaBiLSTMAttentionCRF），单任务，保存checkpoint
    阶段二：加载阶段一权重，在社交NER数据（Weibo NER, CMNER）上微调骨干模型，单任务，保存checkpoint
    阶段三：加载阶段二权重，切换到Ours-Full（启用词汇增强、多任务），在混合数据上联合训练，保存checkpoint
    阶段四：加载阶段三的Ours-Full，进行标准测试集评测、端到端评测（模式12）和案例分析（模式13）
    """
    print("\n" + "="*60)
    print("四阶段课程学习")
    print("="*60)
    
    stages = [
        {
            'name': '阶段1-百科预训练',
            'description': 'CLUENER + MSRA 百科NER数据预训练骨干模型',
            'model_variant': 'RoBERTaBiLSTMAttentionCRF',
            'datasets': ['cluener', 'msra'],
            'use_lexicon': False,
            'use_multitask': False,
            'epochs': 10
        },
        {
            'name': '阶段2-社交微调',
            'description': 'Weibo NER + CMNER 社交媒体数据微调骨干模型',
            'model_variant': 'RoBERTaBiLSTMAttentionCRF',
            'datasets': ['weibo_ner', 'cmner'],
            'use_lexicon': False,
            'use_multitask': False,
            'epochs': 10
        },
        {
            'name': '阶段3-多任务训练',
            'description': '切换到Ours-Full，启用动态词典、语义重塑和多任务学习',
            'model_variant': 'Ours-Full',
            'datasets': ['all'],
            'use_lexicon': True,
            'use_multitask': True,
            'epochs': 15,
            'lexicon_type': 'dynamic',
            'use_reshaping': True
        },
        {
            'name': '阶段4-全量评估',
            'description': '全测试集评估 + GeoGLUE零样本评测',
            'model_variant': 'Ours-Full',
            'datasets': ['all'],
            'use_lexicon': True,
            'use_multitask': True,
            'epochs': 5,
            'lexicon_type': 'dynamic',
            'use_reshaping': True
        }
    ]
    
    prev_checkpoint = None
    
    for stage_idx, stage in enumerate(stages, 1):
        print(f"\n{'='*60}")
        print(f"[课程学习 {stage['name']}]")
        print(f"描述: {stage['description']}")
        print(f"模型变体: {stage['model_variant']}")
        print(f"数据集: {stage['datasets']}")
        print(f"词汇增强: {'启用' if stage['use_lexicon'] else '禁用'}")
        print(f"多任务: {'启用' if stage['use_multitask'] else '禁用'}")
        print(f"训练轮数: {stage['epochs']}")
        
        if stage_idx < 4:  # 前三个阶段是训练
            train_args = {
                'mode': 'train_single' if not stage['use_multitask'] else 'train_multitask',
                'model_name': cfg.roberta_model_name,
                'model_variant': stage['model_variant'],
                'use_crf': True,
                'use_lexicon': stage['use_lexicon'],
                'lexicon_type': stage.get('lexicon_type', 'static'),
                'use_reshaping': stage.get('use_reshaping', False),
                'batch_size': BATCH_SIZE,
                'num_epochs': stage['epochs'] if RUN_MODE == 'experiment' else 2,
                'learning_rate': LEARNING_RATE,
                'load_checkpoint': prev_checkpoint,
                'run_mode': RUN_MODE,
                'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
            }
            
            run_train(train_args)
            
            # 更新checkpoint路径
            prev_checkpoint = os.path.join('models', 'best_model.pt')
        else:  # 第四阶段是评估
            # 运行标准测试集评测
            eval_args = {
                'mode': 'evaluate_model',
                'model_name': cfg.roberta_model_name,
                'model_variant': stage['model_variant'],
                'use_crf': True,
                'use_lexicon': True,
                'load_checkpoint': prev_checkpoint,
                'run_mode': RUN_MODE,
                'sample_size': TEST_SAMPLE_SIZE if RUN_MODE == 'test' else None
            }
            run_train(eval_args)
            
            # GeoGLUE零样本评测
            from geoglue_evaluation import run_geoglue_zero_shot_evaluation
            geoglue_results = run_geoglue_zero_shot_evaluation(
                model_path=prev_checkpoint,
                model_name=cfg.roberta_model_name,
                split='dev'
            )
    
    # ========== 额外评测：模式12和模式13（仅在课程学习后运行） ==========
    print("\n" + "="*60)
    print("[课程学习后评测] 端到端综合评测（模式12）")
    run_end2end_eval(checkpoint_path=prev_checkpoint)
    
    print("\n" + "="*60)
    print("[课程学习后评测] 真实文本案例分析（模式13）")
    run_case_study(checkpoint_path=prev_checkpoint)
    
    print("\n" + "="*60)
    print("四阶段课程学习完成！")


if __name__ == "__main__":
    print_environment_info()
    
    print("\n" + "="*60)
    print("使用说明：")
    print("="*60)
    print("推荐使用命令行交互入口：python cli.py")
    print("")
    print("或者直接运行特定函数：")
    print("1. run_all() - 运行所有核心训练任务（8个阶段）")
    print("2. run_curriculum_learning() - 运行四阶段课程学习")
    print("3. run_end2end_eval(checkpoint_path='models/best_model.pt') - 端到端评测")
    print("4. run_case_study(checkpoint_path='models/best_model.pt') - 案例分析")
    print("="*60 + "\n")
    
    # ======================================
    # 取消以下注释以直接运行特定任务
    # ======================================
    
    # 运行所有核心训练任务（8个阶段）
    # run_all()
    
    # 运行四阶段课程学习（包含模式12和13）
    # run_curriculum_learning()
    
    # 单独运行端到端综合评测（模式12）
    # run_end2end_eval()
    
    # 单独运行真实文本案例分析（模式13）
    # run_case_study()
