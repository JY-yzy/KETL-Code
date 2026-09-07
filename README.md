# 地名实体属性提取

基于深度学习的中文地理信息抽取项目，支持命名实体识别（NER）、关系抽取（RE）和事件抽取（EE）的多任务联合训练。

## 📁 项目结构

```
├── config.py          # 配置文件（超参数、路径、标签映射）
├── data_preprocess.py # 数据预处理（数据集加载、格式转换、词典构建）
├── data_utils.py      # 数据加载工具（MultiTaskDataset、LexiconMatcher）
├── models.py          # 模型定义（NERModel、MultiTaskModel、LexiconAdapter）
├── train.py           # 训练与评估（train_epoch、evaluate_ner/re/ee）
├── main.py            # 主入口（命令行接口、多阶段执行）
├── visualization.py   # 可视化（训练曲线、消融实验图、雷达图）
├── evaluation.py      # 综合评估工具
├── check_data.py      # 数据检查工具
├── logger.py          # 日志管理
├── requirements.txt   # 依赖列表
├── data/              # 数据集目录
│   ├── processed/     # 预处理后数据
│   ├── CLUENER/       # CLUENER数据集
│   ├── MSRA/          # MSRA数据集
│   ├── Weibo NER/     # Weibo NER数据集
│   ├── CMNER/         # CMNER数据集
│   ├── DuIE2.0/       # DuIE 2.0数据集
│   ├── DuEE1.0/       # DuEE 1.0数据集
│   └── GeoGLUE/       # GeoGLUE数据集
├── models/            # 预训练模型目录
└── outputs/           # 输出目录
    └── figures/       # 可视化图表
```

## 🛠️ 环境要求

```bash
# 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 数据预处理

```bash
python main.py --stage preprocess
```

### 2. 训练模型

```bash
# 单任务NER训练
python main.py --stage train --mode train_single --use_crf --use_lexicon

# 多任务训练
python main.py --stage train --mode train_multitask --use_crf --use_lexicon

# NER对比实验（9个模型变体）
python main.py --stage train --mode run_ner_experiments

# 消融实验
python main.py --stage train --mode run_ablation_study

# GeoGLUE零样本测试
python main.py --stage train --mode run_geoglue_zero_shot
```

### 3. 评估模型

```bash
python main.py --stage eval --model_path models/best_model.pt --test_dataset all
```

### 4. 可视化

```bash
# 绘制训练曲线
python main.py --stage visualize --plot_type training_curves --plot_input results/training_log.csv

# 绘制消融实验图
python main.py --stage visualize --plot_type ablation_study --plot_input results/ablation_study_results.csv

# 绘制GeoGLUE雷达图
python main.py --stage visualize --plot_type geoglue_radar --plot_input results/geoglue_results.csv
```

### 5. 完整流程

```bash
# 一键执行完整流程（预处理→训练→评估→可视化）
python main.py --stage full_pipeline --use_crf --use_lexicon
```

## 📋 四阶段课程学习

项目支持分阶段课程学习训练流程：

| 阶段 | 数据 | 任务 | 模型配置 |
|------|------|------|----------|
| **阶段一** | CLUENER + MSRA | 基础NER训练 | RoBERTa-BiLSTM-Attention-CRF |
| **阶段二** | Weibo NER + CMNER | 领域适应微调 | 加载阶段一权重 |
| **阶段三** | DuIE + DuEE | 多任务联合训练 | 启用词汇增强 |
| **阶段四** | GeoGLUE | 零样本测试 | 加载阶段三权重 |

```bash
# 阶段一：基础NER训练
python main.py --stage train --mode train_single --model_name hfl/chinese-roberta-wwm-ext --use_crf

# 阶段二：领域适应（加载阶段一权重）
python main.py --stage train --mode train_single --load_checkpoint models/best_model_stage1.pt

# 阶段三：多任务训练（加载阶段二权重）
python main.py --stage train --mode train_multitask --load_checkpoint models/best_model_stage2.pt --use_lexicon

# 阶段四：GeoGLUE测试（加载阶段三权重）
python main.py --stage train --mode run_geoglue_zero_shot --geoglue_model_path models/best_model_stage3.pt
```

## 📊 评估指标

### NER评估
- **实体级别 Micro-F1**：整体实体识别性能
- **实体级别 Macro-F1**：类别平衡实体识别性能
- 使用 `seqeval` 库计算

### 关系抽取评估
- **三元组全匹配 Micro-F1**：(subject, predicate, object) 全部匹配才算正确
- **精确率、召回率**

### 事件抽取评估
- **触发词分类 Micro-F1**
- **事件类型分类 Micro-F1/Macro-F1**

### GeoGLUE评估
- 6个子任务分别报告指标：
  - address_matching
  - address_parsing
  - geocoding
  - landmark_recognition
  - region_classification
  - spatial_reasoning

## ⚙️ 命令行参数

### 通用参数
```bash
--stage              执行阶段: preprocess/train/eval/visualize/full_pipeline
--mode               训练模式: train_single/train_multitask/run_ner_experiments/run_ablation_study/run_geoglue_zero_shot
--config             配置文件路径
--model_name         预训练模型名称
```

### 模型参数
```bash
--use_crf            使用CRF层
--use_lexicon        使用词汇增强
--model_variant      模型变体: bert/bert-crf/roberta/roberta-crf
```

### 训练参数
```bash
--batch_size         批次大小
--num_epochs         训练轮数
--learning_rate      学习率
--gradient_accumulation_steps  梯度累积步数
--load_checkpoint    加载checkpoint继续训练
```

### 评估参数
```bash
--model_path         已保存模型路径
--test_dataset       测试数据集: all/cluener/msra/weibo_ner/cmner
```

### 可视化参数
```bash
--plot_type          绘图类型: training_curves/ablation_study/frequency_analysis/geoglue_radar/ner_results
--plot_input         绘图输入文件路径
```

## 📈 预期实验结果

| 实验 | 预期提升 |
|------|----------|
| 词汇增强 vs 基础模型 | F1 +2~3% |
| 低频地名 vs 高频地名 | F1 +5~6% |
| 完整模型 vs RoBERTa基线（GeoGLUE） | 平均分 +3% |

## 📝 配置说明

所有超参数在 `config.py` 中定义：

- **路径配置**：数据、模型、结果、日志目录
- **模型配置**：预训练模型名称、模型变体
- **训练超参数**：批次大小、学习率、优化器类型、早停配置
- **模型架构**：是否使用CRF、词汇增强、中间层类型
- **多任务配置**：损失权重

## 📄 输出文件

```
results/
├── results_ner.csv           # NER对比实验结果
├── ablation_study_results.csv # 消融实验结果
├── geoglue_results.csv       # GeoGLUE测试结果
├── multitask_epoch_metrics.csv # 多任务训练指标
└── tensorboard/              # TensorBoard日志

outputs/figures/
├── training_curves.png/pdf    # 训练曲线
├── ablation_study.png/pdf     # 消融实验图
├── frequency_analysis.png/pdf # 频率分析图
├── geoglue_radar.png/pdf     # GeoGLUE雷达图
└── ner_results.png/pdf       # NER对比结果图
```

## 📚 参考数据来源

- **CLUENER**：https://github.com/CLUEbenchmark/CLUENER2020
- **MSRA NER**：微软亚洲研究院命名实体识别数据集
- **Weibo NER**：微博命名实体识别数据集
- **CMNER**：中文社交媒体命名实体识别数据集
- **DuIE 2.0**：百度信息抽取数据集
- **DuEE 1.0**：百度事件抽取数据集
- **GeoGLUE**：地理信息理解基准测试

## 📜 许可证

MIT License