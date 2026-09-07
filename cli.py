#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识增强的中文地理信息提取 - 命令行交互入口
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from config import Config
cfg = Config()

def print_menu():
    """打印菜单"""
    print("\n" + "="*70)
    print("              知识增强的中文地理信息提取 - 功能选择")
    print("="*70)
    print("  [1]  数据预处理")
    print("  [2]  单任务NER实验（9个模型变体）")
    print("  [3]  多任务学习实验（NER+RE+EE）")
    print("  [4]  词汇增强消融实验")
    print("  [5]  完整模型（Ours-Full）实验")
    print("  [6]  消融实验完整组合")
    print("  [7]  频率分层评估")
    print("  [8]  GeoGLUE零样本评测")
    print("  [9]  可视化（生成图表）")
    print("  [10] 四阶段课程学习（完整流程）")
    print("  [11] 端到端综合评测（模式12）")
    print("  [12] 真实文本案例分析（模式13）")
    print("  [13] 运行全部（8个核心阶段）")
    print("  [0]  退出")
    print("="*70)

def run_selected_task(choice):
    """运行选择的任务"""
    from main import (
        run_preprocess, run_train, run_visualize, 
        run_end2end_eval, run_case_study, run_all,
        run_curriculum_learning
    )
    
    # 超参数设置
    BATCH_SIZE = 32
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    if choice == '1':
        print("\n[数据预处理]")
        run_preprocess()
        
    elif choice == '2':
        print("\n[单任务NER实验（9个模型变体）]")
        train_args = {
            'mode': 'run_ner_experiments',
            'model_name': cfg.roberta_model_name,
            'use_crf': True,
            'use_lexicon': True,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '3':
        print("\n[多任务学习实验（NER+RE+EE）]")
        train_args = {
            'mode': 'run_multitask_experiments',
            'model_name': cfg.roberta_model_name,
            'use_crf': True,
            'use_lexicon': True,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'loss_weights': '1.0,0.2,0.1',  # 降低关系和事件任务权重
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '4':
        print("\n[词汇增强消融实验]")
        train_args = {
            'mode': 'run_lexicon_ablation',
            'model_name': cfg.roberta_model_name,
            'use_crf': True,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '5':
        print("\n[完整模型（Ours-Full）实验]")
        train_args = {
            'mode': 'train_multitask',
            'model_name': cfg.roberta_model_name,
            'model_variant': 'Ours-Full',
            'use_crf': True,
            'use_lexicon': True,
            'lexicon_type': 'dynamic',
            'use_reshaping': True,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS * 2,
            'learning_rate': LEARNING_RATE,
            'loss_weights': '1.0,0.2,0.1',
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '6':
        print("\n[消融实验完整组合]")
        train_args = {
            'mode': 'run_ablation_study',
            'model_name': cfg.roberta_model_name,
            'use_crf': True,
            'use_lexicon': True,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '7':
        print("\n[频率分层评估]")
        train_args = {
            'mode': 'run_frequency_analysis',
            'model_name': cfg.roberta_model_name,
            'use_crf': True,
            'use_lexicon': True,
            'batch_size': BATCH_SIZE,
            'run_mode': 'experiment',
            'sample_size': None
        }
        run_train(train_args)
        
    elif choice == '8':
        print("\n[GeoGLUE零样本评测]")
        from geoglue_evaluation import run_geoglue_zero_shot_evaluation
        run_geoglue_zero_shot_evaluation(
            model_name=cfg.roberta_model_name,
            split='dev'
        )
        
    elif choice == '9':
        print("\n[可视化]")
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
        
        # 绘制GeoGLUE雷达图
        geoglue_path = 'results/geoglue_results.csv'
        if os.path.exists(geoglue_path):
            vis_args = {'plot_type': 'geoglue_radar', 'plot_input': geoglue_path}
            run_visualize(vis_args)
            
    elif choice == '10':
        print("\n[四阶段课程学习]")
        run_curriculum_learning()
        
    elif choice == '11':
        print("\n[端到端综合评测（模式12）]")
        checkpoint_path = input("请输入模型checkpoint路径（按回车使用默认路径 models/best_model.pt）：").strip()
        if not checkpoint_path:
            checkpoint_path = 'models/best_model.pt'
        run_end2end_eval(checkpoint_path=checkpoint_path)
        
    elif choice == '12':
        print("\n[真实文本案例分析（模式13）]")
        checkpoint_path = input("请输入模型checkpoint路径（按回车使用默认路径 models/best_model.pt）：").strip()
        if not checkpoint_path:
            checkpoint_path = 'models/best_model.pt'
        run_case_study(checkpoint_path=checkpoint_path)
        
    elif choice == '13':
        print("\n[运行全部核心训练任务]")
        run_all()
        
    elif choice == '0':
        print("\n退出程序...")
        sys.exit(0)
        
    else:
        print("\n无效选择，请输入0-13之间的数字")

def main():
    """主函数"""
    print("="*70)
    print("知识增强的中文地理信息提取系统")
    print("="*70)
    print(f"工作目录: {os.getcwd()}")
    try:
        import torch
        print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    except ImportError:
        print("PyTorch未安装")
    print("="*70)
    
    while True:
        print_menu()
        choice = input("\n请输入选择的功能编号: ").strip()
        run_selected_task(choice)
        
        if choice != '0':
            input("\n按回车键继续...")

if __name__ == "__main__":
    main()