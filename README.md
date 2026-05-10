## rulelift 是一个用于信用风险管理中策略规则的自动挖掘、有效性分析及监控的Python工具包。
- 实时评估上线规则的效度（无需分流测试、无需表现标签）；
- 自动化挖掘高价值的规则（挖掘并评估多种规则、符合业务解释性）；
  
**基于对上线规则的评估模块，我们可以及时发现规则效率低下或不稳定的问题，从而及时调整规则阈值或删减，实现规则A类(Ascending)调优，提升通过率并优化逾期情况。也可以结合规则挖掘模块，新增有效规则，降低逾期率，实现规则D类(Descending)调优，提升规则系统的整体效果及稳定性。**

### 项目统计

[![PyPI Downloads下载量](https://img.shields.io/pypi/dm/rulelift?label=PyPI项目下载量)](https://pypistats.org/packages/rulelift)
[![PyPI version](https://img.shields.io/pypi/v/rulelift.svg)](https://pypi.org/project/rulelift/)

## 一、项目背景

在风控领域，规则系统因其配置便捷性和较强的解释性而被广泛应用，但也存在明显的缺陷：

1. **规则线上效果监控难**：规则效果可能随时间漂移，需要定期监控和调整，但被上线规则拒掉的客户没有后续表现数据，无法直接评估规则拦截效果。此外，规则之间的相互影响难以评估，容易导致冗余或冲突，陷入局部最优；
2. **规则维护复杂**：手动挖掘规则、调整规则耗时耗力；

## 二、rulelift 解决方案

**rulelift** 提供了全面的解决方案，帮助风控团队克服上述挑战：

### 1. 规则智能评估模块
- **无需分流测试**：基于规则命中用户的评级分布即可评估规则效果
- **实时监控**：支持基于生产数据的实时规则效果分析
- **多维度评估**：综合考虑命中率、逾期率、召回率、精确率、lift值、F1分数等指标
- **规则相关性分析**：识别冗余规则，评估规则之间的相互影响
- **策略增益计算**：评估不同规则组合的效果提升

### 2. 规则自动挖掘模块
- **变量分布及全面分析**：特征分析缺失率、单值率、PSI、IV、KS、AUC、损失率、损失率提升度等指标，以及分布情况；
- **单特征规则挖掘**：自动从单个特征中挖掘有效的风控规则
- **多特征交叉规则挖掘**：发现特征之间的复杂交叉关系
- **决策树规则提取**：从多种树模型（随机森林、GBDT、卡方决策树、孤立森林等）中提取可解释的规则
- **可视化支持**：多维度指标直观展示规则效果


### 快速开始

```bash
# 使用pip安装（推荐）
pip install rulelift



注： 考虑风控场景通常是内网使用，文末附上离线安装rulelift的教程
```
### 核心依赖
项目依赖项精简，兼容性良好，仅需要常见的依赖包
| 依赖包 | 版本要求 | 用途 |
|--------|----------|------|
| pandas | >=1.0.0,<2.4.0 | 数据处理和分析 |
| numpy | >=1.18.0,<2.5.0 | 数值计算 |
| scikit-learn | >=0.24.0,<1.9.0 | 机器学习算法 |
| matplotlib | >=3.3.0,<3.11.0 | 基础可视化 |
| seaborn | >=0.11.0,<0.14.0 | 统计可视化 |
| openpyxl | >=3.0.0 | Excel文件读写 |
---
title: RuleLift - 风控规则挖掘与评估工具包 | Credit Risk Rule Mining Toolkit
description: 专业的信用风险管理 Python 工具包，支持规则自动挖掘、智能评估和监控。Automated rule mining and evaluation toolkit for credit risk management.
keywords: rule mining, rule extraction, credit risk management, decision rule extraction, tree rules, fraud detection rules, 风控规则挖掘, 规则评估, 信用风险
---

# RuleLift: 风控规则挖掘与评估工具包

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-4400cc.svg)](https://github.com/psf/black)

[English](#english-version) | [中文](#中文版本)

---

<a name="中文版本"></a>

## 项目概述

**RuleLift** 是一个专业的 **Python 信用风险管理工具包**，专注于 **风控规则挖掘**、**规则评估** 和 **规则监控**。

### 为什么选择 RuleLift？

在风控领域，规则系统因其配置便捷性和较强的解释性而被广泛应用，但也存在明显的痛点：

| 传统痛点 | RuleLift 解决方案 |
|---------|------------------|
| 规则线上效果监控难：被拦截客户无后续表现数据 | 基于用户评级分布实时评估规则效果，无需 A/B 测试 |
| 规则挖掘复杂：手动挖掘和调整规则耗时耗力 | 自动从数据中挖掘高价值业务规则 |
| 特征分析繁琐：需切换多个工具 | 一站式完成 IV/KS/AUC/PSI 等全部分析 |
| 大数据处理困难：内存溢出崩溃 | 内存优化设计，支持万级特征、百万级样本 |

### 核心能力

```
RuleLift
├── 规则智能评估   - 无需分流测试，实时评估规则效果
├── 规则自动挖掘   - 支持单特征、多特征交叉、树模型等多种挖掘方式
├── 变量深度分析   - IV/KS/AUC/PSI 等指标全面分析
├── 内存优化设计   - 批处理、向量化、缓存机制，支持大规模数据
└── 一体化Pipeline - 自动化全流程规则挖掘
```

### 项目统计

- **支持数据规模**: 百万级样本 × 万级特征
- **核心算法**: 单特征挖掘、多特征交叉、决策树/随机森林/GBDT/卡方随机森林/孤立森林
- **评估指标**: IV/KS/AUC/PSI/Lift/F1/Recall/Precision
- **内存优化**: Numpy向量化 + 批处理 + 缓存机制

---

## 目录

- [快速开始](#快速开始)
- [简化调用](#简化调用)
- [核心功能](#核心功能)
- [Pipeline 一体化分析](#pipeline-一体化分析)
- [API 完整参考](#api-完整参考)
- [内存优化与性能](#内存优化与性能)
- [最佳实践](#最佳实践)
- [架构文档](#架构文档)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 快速开始

### 安装

```bash
pip install rulelift
```

**环境要求**：Python >= 3.8 | pandas >= 1.0.0 | numpy >= 1.18.0 | scikit-learn >= 0.24.0 | matplotlib >= 3.3.0

### 5分钟上手

```python
from rulelift import RuleMiningPipeline

# 准备数据
import pandas as pd
df = pd.read_csv('your_data.csv')

# 一键完成全流程分析
pipeline = RuleMiningPipeline(
    df=df,
    target_col='ISBAD',
    exclude_cols=['ID', 'CREATE_TIME'],
    select_max_features=100,        # 限制特征数
    enable_variable_analysis=True,   # 变量分析
    enable_single_rules=True,        # 单特征规则
    enable_cross_rules=True,         # 交叉特征规则
    enable_tree_rules=True,          # 树模型规则
    verbose=True
)

results = pipeline.fit()

# 查看结果
print(results.get_summary())  # 或直接访问 results.summary

# 获取所有规则
all_rules = results.get_all_rules()
all_rules.to_excel('rules_output.xlsx')
```

更多完整示例请参考 [`examples/`](examples/) 目录。

---

## 简化调用

核心类提供了简化别名方法，可以用更短的名称调用常用功能，零性能开销。

### 使用对比

```python
from rulelift import VariableAnalyzer, SingleFeatureRuleMiner, DecisionTreeRuleExtractor

# === 传统调用 ===
result = analyzer.analyze_all_variables(oot_split_date='2026-02-01', date_col='repay_datetime')
detail = analyzer.analyze_variables_detail(variables=['age', 'income'], visualize=True)
selected = analyzer.select_features(iv_threshold=0.02)
rules = miner.get_top_rules(feature=['age', 'income'], top_n=10)
perf = extractor.get_model_performance()

# === 简化调用（等价）===
result = analyzer.vars(oot_split_date='2026-02-01', date_col='repay_datetime')
detail = analyzer.vars_detail(variables=['age', 'income'], visualize=True)
selected = analyzer.select(iv_threshold=0.02)
rules = miner.rules(feature=['age', 'income'], top_n=10)
perf = extractor.perf()
```

### 完整别名列表

| 类 | 简化名 | 原方法 | 说明 |
|----|--------|--------|------|
| **VariableAnalyzer** | `.vars()` | `.analyze_all_variables()` | 分析所有变量 |
| | `.vars_detail()` | `.analyze_variables_detail()` | 详细变量分析 |
| | `.vars_one()` | `.analyze_single_variable()` | 分析单个变量 |
| | `.select()` | `.select_features()` | 特征筛选 |
| | `.plot_bins()` | `.plot_variable_bins()` | 绘制分箱图 |
| | `.quality()` | `.check_data_quality()` | 数据质量检查 |
| | `.psi()` | `.calculate_psi()` | 计算PSI |
| **SingleFeatureRuleMiner** | `.rules()` | `.get_top_rules()` | 获取单特征规则 |
| **MultiFeatureRuleMiner** | `.rules()` | `.get_top_rules()` | 获取交叉规则 |
| | `.rules_hist()` | `.get_top_rules_histogram()` | 直方图阈值搜索 |
| | `.cross_matrix()` | `.generate_cross_matrix()` | 生成交叉矩阵 |
| | `.cross_excel()` | `.generate_cross_matrices_excel()` | 交叉矩阵导出Excel |
| | `.heatmap()` | `.plot_cross_heatmap()` | 交叉热力图 |
| **DecisionTreeRuleExtractor** | `.rules_list()` | `.get_rules_as_dataframe()` | 获取规则DataFrame |
| | `.top_rules()` | `.get_top_rules()` | 获取Top N规则 |
| | `.importance()` | `.get_feature_importance()` | 特征重要性 |
| | `.perf()` | `.get_model_performance()` | 模型性能 |
| | `.generalize()` | `.analyze_rule_generalization()` | 规则泛化分析 |
| **TreeRuleExtractor** | `.importance()` | `.get_feature_importance()` | 特征重要性 |
| **RuleMiningResults** | `.all()` | `.get_all_rules()` | 获取所有规则 |
| | `.top()` | `.get_top_rules()` | 获取Top N规则 |

---

## 核心功能

### 1. 觘则智能评估

无需 A/B 测试，基于规则命中用户的评级分布即可评估规则效果。

**支持指标**：
- **预估指标**：坏账率、Lift值、召回率、精确率
- **实际指标**：F1分数、实际坏账率、实际提升度
- **稳定性指标**：命中率标准差、变异系数

### 2. 规则自动挖掘

支持多种挖掘算法，覆盖不同业务场景：

| 算法 | 适用场景 | 特点 |
|------|---------|------|
| `SingleFeatureRuleMiner` | 快速发现强特征 | 单特征最优阈值挖掘，内存优化 |
| `MultiFeatureRuleMiner` | 提升规则覆盖率 | 多特征交叉组合，numpy向量化 |
| `TreeRuleExtractor('dt')` | 快速生成规则 | 决策树，简单直观 |
| `TreeRuleExtractor('rf')` | 需要稳定规则 | 随机森林，多树集成 |
| `TreeRuleExtractor('gbdt')` | 追求高精度 | 梯度提升树 |
| `TreeRuleExtractor('chi2')` | 卡方分箱+随机森林 | 卡方自动分箱后构建随机森林 |
| `TreeRuleExtractor('isf')` | 异常检测场景 | 孤立森林，通过异常分数发现风险规则 |

### 3. 变量深度分析

全方位评估变量价值：

| 指标 | 说明 | 应用 | 判断标准 |
|------|------|------|----------|
| IV (Information Value) | 变量预测能力 | 特征筛选 | >0.1强, 0.02-0.1中, <0.02弱 |
| KS (Kolmogorov-Smirnov) | 变量区分能力 | 评估分箱效果 | >0.3强, 0.2-0.3中, <0.2弱 |
| AUC | 预测准确性 | 模型评估 | >0.7较好 |
| PSI (Population Stability) | 变量稳定性 | 监控特征漂移 | <0.1稳定, >0.25不稳定 |

### 4. 策略优化

计算规则组合的边际增益，找到最优策略组合。

---

## Pipeline 一体化分析

`RuleMiningPipeline` 整合所有功能，一键完成全流程分析。

### 完整参数说明

```python
from rulelift import RuleMiningPipeline

pipeline = RuleMiningPipeline(
    df=data,
    target_col='ISBAD',                # 目标变量

    # === 数据配置 ===
    exclude_cols=['ID', 'TIME'],       # 排除的列
    amount_col='AMOUNT',                # 金额列（可选）
    ovd_bal_col='OVD_BAL',             # 逾期余额列（可选）
    date_col='CREATE_TIME',            # 日期列（用于OOT分割）
    oot_split_date='2024-01-01',       # OOT分割日期

    # === 特征选择参数 ===
    select_iv_threshold=0.02,           # 最低有效IV阈值
    select_max_features=100,           # 最大特征数限制
    select_psi_threshold=None,         # PSI阈值（过滤不稳定特征，None=不过滤）

    # === 变量分析参数 ===
    variable_binning_method='chi2',    # 分箱方法: 'chi2' | 'quantile'
    variable_n_bins=10,                # 默认分箱数量
    variable_min_samples_pct=0.05,     # 最小分箱样本比例
    variable_chi2_threshold=3.841,     # 卡方阈值
    variable_n_jobs=-1,                # 并行任务数 (-1表示全部CPU)

    # === 单特征规则参数 ===
    single_iv_threshold=0.1,           # 使用IV>0.1的特征
    single_top_n=10,                   # 每特征返回规则数
    single_min_lift=1.1,               # 最小lift值
    single_min_samples=10,             # 最小样本数
    single_algorithm='histogram',      # 算法: 'histogram' | 'chi2'
    single_n_jobs=-1,                  # 并行任务数

    # === 交叉特征规则参数 ===
    cross_iv_threshold=0.05,           # 使用0.05<=IV<0.1的特征
    cross_top_features=3,              # 使用前N个特征
    cross_top_n=5,                     # 每对特征返回规则数
    cross_min_samples=10,              # 最小样本数
    cross_min_lift=1.1,                # 最小lift值
    cross_n_bins=8,                    # 分箱数量
    cross_max_pairs=6,                 # 最多处理特征对数

    # === 树模型参数 ===
    tree_algorithm='rf',               # 'dt', 'rf', 'gbdt', 'chi2', 'isf'
    tree_max_depth=3,
    tree_min_samples_leaf=5,           # 叶子最小样本数
    tree_n_estimators=10,
    tree_max_features='sqrt',          # 最大特征数
    tree_top_n=20,                     # 返回规则数

    # === 内存管理参数 ===
    memory_mode='auto',                # 'auto', 'full', 'low'
    min_free_memory_mb=500,            # 最小可用内存（MB）
    enable_auto_cleanup=True,          # 自动清理内存
    auto_skip_on_low_memory=False,     # True=直接跳过, False=降级到低内存模式

    # === 功能开关 ===
    feature_trends='auto',             # 特征趋势约束: Dict / 'auto' / None
    enable_variable_analysis=True,
    enable_single_rules=True,
    enable_cross_rules=True,
    enable_tree_rules=True,
    enable_validation=False,           # 启用规则验证
    random_state=42,                   # 随机种子
    verbose=True
)

results = pipeline.fit()
```

### Pipeline 执行流程

```
Step 0: 数据验证
  └─> 验证数据完整性和目标列存在性

Step 1: 变量分析
  └─> 计算所有变量的 IV/KS/AUC/PSI

Step 2: 特征分组
  └─> 按IV阈值分为: 高IV | 中IV | 低IV

Step 3: 单特征规则挖掘
  └─> 对高IV特征进行单特征阈值挖掘

Step 4: 交叉特征规则挖掘
  └─> 对中IV特征进行交叉组合挖掘

Step 5: 树模型规则挖掘
  └─> 使用决策树/随机森林提取规则
```

---

## API 完整参考

---

### 一、工具函数 (utils/)

#### 1.1 load_example_data

加载内置示例数据文件。

```python
from rulelift.utils import load_example_data

df_hit = load_example_data('hit_rule_info')  # 规则命中数据 (998行)
df_feas = load_example_data('feas_target')    # 可行性目标数据 (499行)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_name` | str | `'hit_rule_info'` | 数据名称：`'hit_rule_info'` 或 `'feas_target'` |
| `file_path` | str | None | 自定义数据文件路径 |

**返回**: `pd.DataFrame`

---

#### 1.2 preprocess_data

预处理数据，将百分比字符串转为浮点数。

```python
from rulelift.utils import preprocess_data

df = preprocess_data(df, user_level_badrate_col='BADRATE')
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 原始数据 |
| `user_level_badrate_col` | str | None | 用户评级坏账率字段名（含百分号字符串） |

**返回**: `pd.DataFrame`

---

#### 1.3 UnifiedBinningCalculator

统一分箱计算器，支持多种分箱方法。

```python
from rulelift.utils import UnifiedBinningCalculator
import numpy as np

calc = UnifiedBinningCalculator(n_bins=10, default_method='chi2')

# 计算分箱边界（传入 numpy 数组）
bins = calc.compute_bins(df['feature'].values, df['target'].values, n_bins=10)

# 计算分箱统计量（返回 tuple: (stats_df, iv, ks)）
stats_df, iv, ks = calc.compute_bin_stats(df['feature'].values, df['target'].values, bins)

# 应用分箱到数据
binned = calc.apply_bins(df['feature'].values, bins)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_method` | str | `'quantile'` | 默认分箱方法：`'quantile'`/`'chi2'`/`'custom'`/`'equal_width'` |
| `n_bins` | int | 10 | 默认分箱数量 |
| `chi2_threshold` | float | 3.841 | 卡方阈值 |
| `min_samples_pct` | float | 0.02 | 最小样本比例 |
| `decimal_places` | int | 3 | 小数位数 |
| `missing_values` | list | None | 缺失值列表 |
| `special_values` | list | None | 特殊值列表 |
| `max_iterations` | int | 500 | 卡方分箱最大迭代次数 |
| `categorical_nunique_threshold` | int | 10 | 类别变量唯一值阈值 |
| `empty_separate` | bool | True | 空值单独分箱 |
| `robust_mode` | bool | True | 鲁棒模式 |

**主要方法**:

| 方法 | 说明 | 返回 |
|------|------|------|
| `compute_bins(feature_values, target_values, n_bins)` | 计算分箱边界 | `np.ndarray` |
| `compute_bin_stats(feature_values, target_values, bin_edges)` | 计算分箱统计量 | `(DataFrame, iv, ks)` |
| `apply_bins(feature_values, bin_edges)` | 应用分箱 | `np.ndarray` |

---

#### 1.4 CategoricalVariableProcessor

类别变量处理器，自动检测和处理类别型特征。

```python
from rulelift.utils.categorical import CategoricalVariableProcessor

proc = CategoricalVariableProcessor()
info = proc.detect_and_prepare(df, 'app_type', 'label')
# info: {'feature': 'app_type', 'method': '...', 'detection': {...}, 'bin_mapping': {...}}
```

| 方法 | 说明 | 返回 |
|------|------|------|
| `detect_and_prepare(df, feature, target_col)` | 检测类别变量并准备分箱 | `Dict` |

---

#### 1.5 ParallelExecutor

并行执行器，支持 joblib 多种后端。

```python
from rulelift.utils import ParallelExecutor

executor = ParallelExecutor(n_jobs=-1, backend='loky')
results = executor.map(func, items_list)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_jobs` | int | -1 | 并行数（-1=全部核心） |
| `backend` | str | `'loky'` | 后端：`'loky'`/`'multiprocessing'`/`'threading'` |
| `timeout` | float | 300 | 超时时间（秒） |
| `parallel_threshold` | int | 20 | 最小并行任务数 |

---

#### 1.6 类别检测函数

```python
from rulelift.utils import (
    is_categorical, smart_detect_categorical,
    should_bin_categorical, detect_categorical_type,
    batch_detect_categorical
)

# 基础判断
is_categorical(df['app_type'])           # True/False
smart_detect_categorical(df['app_type']) # 智能判断（含可转换检测）

# 是否需要分箱
needs, reason = should_bin_categorical(df['app_type'])

# 完整检测
info = detect_categorical_type(df['app_type'])
# {'is_categorical': True, 'needs_binning': True, 'nunique': 11, 'unique_ratio': 0.0015}

# 批量检测
results = batch_detect_categorical(df, columns=['col1', 'col2'])
```

---

### 二、指标计算 (metrics/)

#### 2.1 compute_feature_trends

自动推断特征趋势方向（基于相关系数）。

```python
from rulelift.metrics import compute_feature_trends

trends = compute_feature_trends(df, ['age', 'income'], target_col='label')
# {'age': 1, 'income': -1}
# 1 = 正相关（建议保留 >= 规则），-1 = 负相关（建议保留 <= 规则）
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 数据集 |
| `features` | List[str] | 特征列表 |
| `target_col` | str | 目标列名 |

**返回**: `Dict[str, int]` — {特征名: 1 或 -1}

---

#### 2.2 add_cumulative_metrics

为规则结果增加累计指标。

```python
from rulelift.metrics import add_cumulative_metrics

rules_df = add_cumulative_metrics(rules_df, sort_by='threshold', ascending=True)
# 新增列：cum_total_pct, cum_bad_rate, cum_bad_rate_remaining
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 需含 `selected_samples`、`selected_bad` 列 |
| `sort_by` | str | `'threshold'` | 排序依据 |
| `ascending` | bool | True | 升序（从低到高逐级收紧） |

**返回**: `pd.DataFrame` — 增加了 `cum_total_pct`、`cum_bad_rate`、`cum_bad_rate_remaining` 列

---

#### 2.3 calculate_psi

计算 Population Stability Index。

```python
from rulelift.metrics import calculate_psi

psi = calculate_psi(train_data, oot_data, buckets=10)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `expected` | Series | - | 预期分布（训练集） |
| `actual` | Series | - | 实际分布（OOT集） |
| `buckets` | int | 10 | 分箱数量 |

**返回**: `float` — PSI值（<0.1 稳定，0.1-0.25 中等，>0.25 不稳定）

---

#### 2.4 calculate_rule_correlation

计算规则间相关性矩阵。

```python
from rulelift.metrics import calculate_rule_correlation

corr_matrix = calculate_rule_correlation(user_rule_df)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `user_rule_df` | DataFrame | 用户-规则矩阵（0/1） |

**返回**: `pd.DataFrame` — 相关系数矩阵

---

#### 2.5 calculate_estimated_metrics / calculate_actual_metrics

基于用户评级分布计算规则预估指标和实际指标。

```python
from rulelift.metrics import calculate_estimated_metrics, calculate_actual_metrics

# 预估指标（基于 USER_LEVEL_BADRATE）
est = calculate_estimated_metrics(rule_score, user_rule_df, 'USER_ID', 'BADRATE')

# 实际指标（基于 ISBAD）
act = calculate_actual_metrics(rule_score, user_rule_df, 'USER_ID', 'ISBAD')
```

**返回**: `Dict[str, Dict]` — {规则名: {指标名: 值}}

---

#### 2.6 calculate_strategy_pair_gain

计算两两策略间的边际增益。

```python
from rulelift.metrics import calculate_strategy_pair_gain

gain = calculate_strategy_pair_gain(user_rule_df, user_target, ['R1'], ['R2'])
# {'gain_users': 50, 'gain_bads': 10, 'gain_badrate': 0.20, 'gain_lift': 1.5, ...}
```

---

#### 2.7 稳定性指标

```python
from rulelift.metrics import calculate_rule_psi, calculate_rule_stability, calculate_long_term_stability

# 规则在不同时期的PSI
psi_df = calculate_rule_psi(rule_score, 'RULE', 'HIT_DATE', 'USER_ID')

# 规则月度稳定性
stability = calculate_rule_stability(rule_score, 'RULE', 'HIT_DATE', 'USER_ID')
# {'R1': {'hit_rate_std': 0.02, 'hit_rate_cv': 0.1, 'months_analyzed': 6}}

# 规则长期稳定性（滚动窗口）
long_term = calculate_long_term_stability(rule_score, 'RULE', 'HIT_DATE', 'USER_ID', window_size=30)
```

---

### 三、变量分析 (analysis/VariableAnalyzer)

#### 3.1 VariableAnalyzer 构造器

```python
from rulelift.analysis import VariableAnalyzer

analyzer = VariableAnalyzer(
    df,
    target_col='label',
    exclude_cols=['user_id', 'date_col'],
    n_bins=10,
    binning_method='chi2',          # 'chi2' | 'quantile'
    min_samples_pct=0.02,           # 最小分箱样本比例
    n_jobs=-1,                       # 并行数（-1=全部核心）
    enable_adaptive_parallel=True,   # 自适应并行（内存感知）
    min_batch_size=10,               # 最小批次大小
    max_memory_usage_ratio=0.7,      # 最大内存使用比例
    log_level='INFO'                # 日志级别
)
```

**数据配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 输入数据集 |
| `target_col` | str | `'ISBAD'` | 目标列名 |
| `exclude_cols` | list | None | 排除的列 |
| `amount_col` | str | None | 金额列（可选） |
| `ovd_bal_col` | str | None | 逾期余额列（可选） |

**分箱配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_bins` | int | 10 | 默认分箱数量 |
| `binning_method` | str | `'chi2'` | 分箱方法：`'chi2'`/`'quantile'` |
| `chi2_threshold` | float | 3.841 | 卡方分箱合并阈值 |
| `min_samples_pct` | float | 0.02 | 最小分箱样本比例 |
| `iv_calculation_method` | str | `'standard'` | IV计算方法 |
| `epsilon` | float | 1e-10 | 数值稳定小量 |

**类别变量配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `categorical_cols` | list | None | 手动指定类别列 |
| `auto_detect_categorical` | bool | True | 自动检测类别变量 |
| `max_categorical_bins` | int | 10 | 类别变量最大分箱数 |
| `categorical_nunique_threshold` | int | 10 | 唯一值数量阈值 |
| `categorical_unique_ratio_threshold` | float | 0.5 | 唯一值比例阈值 |

**缺失值配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `handle_missing` | bool | True | 是否处理缺失值 |
| `missing_value` | float | -9999 | 缺失值标识 |
| `missing_strategy` | str | `'single'` | 缺失值处理策略 |
| `missing_fill_value` | float | None | 缺失值填充值 |

**并行与性能配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_jobs` | int | -1 | 并行进程数（-1=全部核心） |
| `enable_adaptive_parallel` | bool | True | 自适应并行（内存感知） |
| `memory_threshold_mb` | float | 500 | 内存阈值（MB） |
| `min_batch_size` | int | 10 | 最小批次大小 |
| `max_memory_usage_ratio` | float | 0.7 | 内存使用上限 |
| `gc_interval` | int | 5 | GC间隔 |
| `log_level` | str | `'INFO'` | 日志级别 |

---

#### 3.2 analyze_all_variables

> 简化别名：`.vars()`

批量分析所有变量，计算 IV/KS/AUC/PSI 等指标。

```python
# 带OOT分割
result = analyzer.analyze_all_variables(
    oot_split_date='2026-02-01',
    date_col='repay_datetime',
    batch_size=50,
    show_progress=True
)

# 不带OOT分割
result = analyzer.analyze_all_variables()
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `oot_split_date` | str | None | OOT分割日期（如 `'2024-01-01'`） |
| `date_col` | str | None | 日期列名 |
| `batch_size` | int | 50 | 批处理大小 |
| `show_progress` | bool | True | 是否显示进度条 |

**返回**: `pd.DataFrame` — 每行一个特征，包含 `variable`, `iv`, `ks`, `auc`, `gini`, `psi` 等列

---

#### 3.3 analyze_single_variable

> 简化别名：`.vars_one()`

分析单个变量的分箱统计。

```python
stats = analyzer.analyze_single_variable('age', n_bins=10)
```

**返回**: `pd.DataFrame` — 分箱统计结果

---

#### 3.4 analyze_variables_detail

> 简化别名：`.vars_detail()`

详细分析变量的分箱明细，支持自定义分箱和可视化。

```python
detail = analyzer.analyze_variables_detail(
    variables=['age', 'income'],
    n_bins=10,
    visualize=True,
    custom_bins_params={
        'age': [18, 25, 35, 45, 55, 65],
        'city': [['北京', '上海'], ['深圳', '广州'], ['其他']]
    },
    oot_split_date='2026-02-01',
    date_col='repay_datetime',
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `variables` | list | None | 变量列表（None=全部） |
| `n_bins` | int | 10 | 分箱数量 |
| `visualize` | bool | True | 是否可视化 |
| `custom_bins_params` | dict | None | 自定义分箱参数 |
| `oot_split_date` | str | None | OOT分割日期 |
| `date_col` | str | None | 日期列名 |
| `binning_method` | str | `'chi2'` | 分箱方法 |

---

#### 3.5 select_features

> 简化别名：`.select()`

基于多维指标筛选特征。

```python
result = analyzer.select_features(
    iv_threshold=0.02,
    psi_threshold=0.25,
    ks_threshold=0.02,
)
# result: {
#     'selected_features': ['feature1', 'feature2', ...],
#     'selected_df': DataFrame,
#     'rejected_features': {'feature3': ['IV<0.02', 'KS<0.02'], ...},
#     'correlation_removed': {'feature4': '与 feature1 相关性过高'},
#     'summary': {'total_features': 100, 'selected_count': 20, ...}
# }
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `analysis_result` | DataFrame | None | 自定义分析结果（None=使用缓存） |
| `iv_threshold` | float | 0.02 | IV最小阈值 |
| `missing_rate_threshold` | float | 0.8 | 最大缺失率阈值 |
| `single_value_rate_threshold` | float | 0.95 | 最大单值率阈值 |
| `psi_threshold` | float | 0.25 | PSI最大阈值（过滤不稳定特征） |
| `ks_threshold` | float | 0.02 | KS最小阈值 |
| `correlation_threshold` | float | 0.85 | 相关性最大阈值 |
| `apply_correlation_filter` | bool | True | 是否应用相关性过滤 |
| `mode` | str | `'and'` | 过滤模式：`'and'`（全部满足）/ `'or'`（任一满足） |

**返回**: `Dict` — 包含 `selected_features`, `selected_df`, `rejected_features`, `correlation_removed`, `summary`

---

#### 3.6 calculate_psi

计算单个特征的 PSI 值。

```python
psi = analyzer.calculate_psi(
    feature='age',
    oot_split_date='2026-02-01',
    date_col='repay_datetime'
)
```

**返回**: `float` — PSI值

---

#### 3.7 plot_variable_bins

> 简化别名：`.plot_bins()`

绘制变量分箱可视化图。

```python
fig = analyzer.plot_variable_bins('age', n_bins=10, save_path='age_bins.png')
```

---

#### 3.8 check_data_quality

数据质量检查，识别空列、高缺失列、常量列。

```python
report = analyzer.check_data_quality(
    check_missing=True,
    check_constant=True,
    missing_threshold=0.95,
)
```

---

### 四、规则分析 (analysis/)

#### 4.1 evaluate_rule_description

通过规则描述直接评估规则效果（无需预计算命中矩阵）。

```python
from rulelift.analysis import evaluate_rule_description

results = evaluate_rule_description(
    [
        {'age': [60, None]},            # age >= 60
        {'income': [None, 5000]},      # income <= 5000
        {'city': ['北京', '上海']},      # city in ['北京', '上海']
        {'age': [30, 50], 'city': '北京'}, # 多条件 AND
    ],
    df=df,
    target_col='label'
)
# 返回 DataFrame: rule_description, badrate, lift, recall, precision, f1,
#                 cum_total_pct, cum_bad_rate, cum_bad_rate_remaining
```

**支持的规则格式**:

| 格式 | 示例 | 含义 |
|------|------|------|
| 数值 >= | `{'age': [60, None]}` | age >= 60 |
| 数值 <= | `{'age': [None, 80]}` | age <= 80 |
| 数值范围 | `{'age': [60, 80]}` | 60 <= age <= 80 |
| 类别匹配 | `{'city': '北京'}` | city == '北京' |
| 类别列表 | `{'city': ['北京', '上海']}` | city in [...] |
| 多条件 AND | `{'age': [60, None], 'city': '北京'}` | 同时满足 |

---

#### 4.2 analyze_rules

基于规则命中数据评估规则效果。

```python
from rulelift.analysis import analyze_rules

result = analyze_rules(
    rule_score_df,
    rule_col='RULE',
    user_id_col='USER_ID',
    user_target_col='ISBAD',
    user_level_badrate_col='BADRATE',
    hit_date_col='HIT_DATE',
    include_stability=True
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule_col` | str | `'RULE'` | 规则名字段 |
| `user_id_col` | str | `'USER_ID'` | 用户ID字段 |
| `user_level_badrate_col` | str | None | 预估坏账率字段 |
| `user_target_col` | str | None | 实际目标字段 |
| `hit_date_col` | str | None | 命中日期字段 |
| `include_stability` | bool | True | 是否计算稳定性指标 |

---

#### 4.3 analyze_rule_correlation

分析规则间相关性。

```python
from rulelift.analysis import analyze_rule_correlation

corr_matrix, max_corr = analyze_rule_correlation(
    rule_score_df, 'RULE', 'USER_ID'
)
```

**返回**: `(DataFrame, Dict)` — (相关系数矩阵, 每条规则最大相关性)

---

#### 4.4 get_user_rule_matrix

构建用户-规则命中矩阵。

```python
from rulelift.analysis import get_user_rule_matrix

matrix = get_user_rule_matrix(rule_score_df, 'RULE', 'USER_ID')
```

---

#### 4.5 calculate_strategy_gain

计算策略组合的边际增益。

```python
from rulelift.analysis import calculate_strategy_gain

gain_matrix, details = calculate_strategy_gain(
    rule_score_df, 'RULE', 'USER_ID', 'ISBAD',
    strategy_definitions={
        'Strategy1': ['R1', 'R2'],
        'Strategy2': ['R3', 'R4'],
    },
    metric='gain_lift'
)
```

| 参数 | 说明 |
|------|------|
| `metric` | `'gain_lift'`/`'gain_badrate'`/`'gain_users'`/`'gain_bads'`/`'gain_coverage'`/`'gain_recall'` |

---

### 五、规则挖掘 (mining/)

> **已废弃**: `XGBoostRuleMiner` 已标记为废弃（deprecated），请使用 `TreeRuleExtractor(algorithm='gbdt')` 替代。TreeRuleExtractor 的 `'xgb'` 算法标识也已废弃，会自动转为 `'gbdt'`。

#### 5.1 SingleFeatureRuleMiner

单特征规则挖掘器，通过阈值搜索找到最优规则。

```python
from rulelift.mining import SingleFeatureRuleMiner

miner = SingleFeatureRuleMiner(
    df,
    target_col='label',
    exclude_cols=['user_id'],
    min_lift=1.1,
    algorithm='histogram',     # 'histogram' | 'chi2'
    n_jobs=-1,
    feature_trends='auto',     # Dict / 'auto' / None
)

# 挖掘指定特征
rules = miner.get_top_rules(
    feature=['age', 'income'],
    top_n=10,
    min_samples=10,
    use_parallel=True,
    show_progress=True,
    group_by_feature=True     # 每特征取top_n
)

# 挖掘全部特征
rules = miner.get_top_rules(
    feature=None,
    top_n=5,
    metric='lift',            # 'lift' | 'badrate'
    group_by_feature=True
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 数据集 |
| `target_col` | str | `'ISBAD'` | 目标列 |
| `exclude_cols` | list | None | 排除列 |
| `amount_col` | str | None | 金额列（可选） |
| `ovd_bal_col` | str | None | 逾期余额列（可选） |
| `algorithm` | str | `'histogram'` | 算法：`'histogram'`/`'chi2'` |
| `min_lift` | float | 1.1 | 最小Lift值 |
| `histogram_bins` | int | 100 | 直方图分箱数 |
| `chi2_threshold` | float | 3.841 | 卡方阈值 |
| `n_jobs` | int | -1 | 并行数 |
| `feature_trends` | dict/str | None | 特征趋势约束 |

**类别变量配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `categorical_nunique_threshold` | int | 10 | 类别唯一值阈值 |
| `categorical_unique_ratio_threshold` | float | 0.5 | 唯一值比例阈值 |
| `max_categorical_bins` | int | 10 | 类别最大分箱数 |
| `custom_categorical_mappings` | dict | None | 自定义类别映射 |

**缺失值配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `missing_threshold` | float | 0.95 | 缺失率阈值 |
| `missing_strategy` | str | `'fill'` | 缺失值处理策略 |
| `missing_fill_value` | float | -999 | 缺失值填充值 |

**验证配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_size` | float | 0.2 | 测试集比例 |
| `validation_mode` | str | `'split'` | 验证模式：`'split'`/`'oot'` |
| `date_col` | str | None | 日期列（OOT模式） |
| `oot_split_date` | str | None | OOT分割日期 |
| `enable_validation` | bool | False | 是否启用验证 |

**并行与性能配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_jobs` | int | -1 | 并行进程数（-1=全部核心） |
| `parallel_backend` | str | `'loky'` | 并行后端：`'loky'`/`'multiprocessing'`/`'threading'` |
| `enable_adaptive_parallel` | bool | True | 自适应并行（内存感知） |
| `memory_threshold_mb` | float | 500 | 内存阈值（MB） |
| `gc_interval` | int | 10 | GC间隔 |
| `feature_trends` | dict/str | None | 特征趋势约束：Dict / `'auto'` / None |

**返回**: `pd.DataFrame` — 包含 `feature`, `threshold`, `operator`, `lift`, `badrate`, `selected_samples` 等列

---

#### 5.2 MultiFeatureRuleMiner

交叉特征规则挖掘器。

```python
from rulelift.mining import MultiFeatureRuleMiner

miner = MultiFeatureRuleMiner(
    df,
    target_col='label',
    enable_validation=False,
    feature_trends='auto'
)

# 网格分箱法
rules = miner.get_top_rules(
    feature1='age', feature2='income',
    top_n=10, min_samples=10, min_lift=1.1, n_bins=8
)

# 直方图阈值搜索法
rules = miner.get_top_rules_histogram(
    feature1='age', feature2='income',
    top_n=10, min_samples=10, min_lift=1.1, n_thresholds=20
)

# 交叉矩阵
cross_matrix = miner.generate_cross_matrix('age', 'income')

# 热力图
miner.plot_cross_heatmap('age', 'income', metric='lift', save_path='heatmap.png')
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 数据集 |
| `target_col` | str | `'ISBAD'` | 目标列 |
| `categorical_nunique_threshold` | int | 10 | 类别唯一值阈值 |
| `feature_trends` | dict/str | None | 特征趋势约束 |

---

#### 5.3 DecisionTreeRuleExtractor

基于决策树的规则提取。

```python
from rulelift.mining import DecisionTreeRuleExtractor

extractor = DecisionTreeRuleExtractor(
    df,
    target_col='label',
    exclude_cols=['user_id', 'repay_datetime'],
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

train_acc, test_acc = extractor.train()
rules = extractor.extract_rules()
evaluation = extractor.evaluate_rules(rules)
importance = extractor.get_feature_importance()
performance = extractor.get_model_performance()
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 数据集 |
| `target_col` | str | `'ISBAD'` | 目标列 |
| `exclude_cols` | list | None | 排除列 |
| `max_depth` | int | 5 | 最大深度 |
| `min_samples_leaf` | int | 5 | 叶子最小样本数 |
| `min_samples_split` | int | 10 | 分裂最小样本数 |
| `test_size` | float | 0.2 | 测试集比例 |
| `random_state` | int | 42 | 随机种子 |
| `validation_mode` | str | `'split'` | 验证模式：`'split'`/`'oot'` |
| `date_col` | str | None | 日期列（OOT模式） |
| `oot_split_date` | str | None | OOT分割日期 |
| `enable_advanced_validation` | bool | False | 启用高级验证 |

---

#### 5.4 TreeRuleExtractor

统一树模型规则提取器，支持 dt/rf/gbdt/chi2/isf 五种算法。

```python
from rulelift.mining import TreeRuleExtractor

extractor = TreeRuleExtractor(
    df,
    target_col='label',
    exclude_cols=['user_id'],
    algorithm='rf',              # 'dt' | 'rf' | 'gbdt' | 'chi2' | 'isf'
    max_depth=3,
    min_samples_leaf=5,
    n_estimators=10,             # dt时为1
    random_state=42,
    feature_trends='auto'
)

extractor.train()
rules = extractor.extract_rules()
result = extractor.evaluate_rules()   # 注意：不需要传参（isf除外）
```

**算法说明**:

| 算法 | 适用场景 | 说明 |
|------|---------|------|
| `dt` | 快速生成规则 | 单棵决策树，简单直观 |
| `rf` | 需要稳定规则 | 随机森林，多树集成 |
| `gbdt` | 追求高精度 | 梯度提升树，需设置 `learning_rate` 和 `subsample` |
| `chi2` | 自动分箱+随机森林 | 先用卡方算法自动分箱，再构建随机森林，需设置 `min_bin_ratio` |
| `isf` | 异常检测场景 | 孤立森林，通过异常分数发现风险规则。**注意**: 不支持 `evaluate_rules()` |

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 数据集 |
| `target_col` | str | `'ISBAD'` | 目标列 |
| `exclude_cols` | list | None | 排除列 |
| `algorithm` | str | `'rf'` | 算法：`'dt'`/`'rf'`/`'gbdt'`/`'chi2'`/`'isf'` |
| `max_depth` | int | 3 | 最大深度 |
| `min_samples_split` | int | 10 | 分裂最小样本数 |
| `min_samples_leaf` | int/float | 5 | 叶子最小样本数（支持浮点比例） |
| `n_estimators` | int | 10 | 树数量（dt时忽略） |
| `max_features` | str | `'sqrt'` | 最大特征数 |
| `learning_rate` | float | 0.1 | 学习率（gbdt） |
| `subsample` | float | 1.0 | 子采样比例（gbdt） |
| `min_bin_ratio` | float | 0.05 | 最小分箱比例（chi2算法） |
| `isf_weights` | dict | None | 孤立森林规则权重配置 |
| `test_size` | float | 0.3 | 测试集比例 |
| `random_state` | int | 42 | 随机种子 |
| `amount_col` | str | None | 金额列（可选） |
| `ovd_bal_col` | str | None | 逾期余额列（可选） |
| `feature_trends` | dict/str | None | 特征趋势约束 |
| `validation_mode` | str | `'split'` | 验证模式：`'split'`/`'oot'` |
| `date_col` | str | None | 日期列（OOT模式） |
| `oot_split_date` | str | None | OOT分割日期 |
| `enable_advanced_validation` | bool | False | 启用高级验证 |

**`isf_weights` 可配置项**（孤立森林规则评分权重）:

| 键 | 默认值 | 说明 |
|----|--------|------|
| `purity` | 0.5 | 坏客户纯度权重 |
| `anomaly` | 0.3 | 异常分数权重 |
| `sample` | 0.15 | 样本数量权重 |
| `hit` | 0.05 | 异常坏客户命中比例权重 |

**注意**: `evaluate_rules()` 无需传入 rules 参数，内部自动使用已提取的规则。`isf` 算法不支持规则评估。

---

#### 5.5 RuleValidator

独立规则验证器，支持 split/OOT 两种验证模式。

```python
from rulelift.mining import RuleValidator

validator = RuleValidator(
    df, target_col='label',
    validation_mode='split',      # 'split' | 'oot'
    test_size=0.3,
    date_col='repay_datetime',
    oot_split_date='2026-02-01'
)

# 分割数据（必须先调用）
validator.split_train_test()

# 评估单条规则
result = validator.evaluate_rule("feature1 > 100")

# 批量评估规则
results = validator.evaluate_rules(["feature1 > 100", "feature2 <= 50"])
comparison = validator.compare_train_test_performance(results)
validator.print_validation_report(comparison)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | - | 数据集 |
| `target_col` | str | `'ISBAD'` | 目标列 |
| `test_size` | float | 0.2 | 测试集比例 |
| `validation_mode` | str | `'split'` | 验证模式：`'split'`/`'oot'` |
| `random_state` | int | 42 | 随机种子 |
| `date_col` | str | None | 日期列（OOT模式） |
| `oot_split_date` | str | None | OOT分割日期 |

> **RuleValidatorMixin**: `DecisionTreeRuleExtractor` 和 `TreeRuleExtractor` 自动继承 `RuleValidatorMixin`，无需单独创建 `RuleValidator` 即可使用验证功能。

---

### 六、可视化 (visualization/)

#### 6.1 RuleVisualizer

```python
from rulelift.visualization import RuleVisualizer

viz = RuleVisualizer(dpi=300)

# 规则比较图
fig = viz.plot_rule_comparison(rules_df, metrics=['lift', 'badrate'], save_path='comp.png')

# 规则分布直方图
fig = viz.plot_rule_distribution(rules_df, metric='lift', save_path='dist.png')

# Lift-Precision 散点图
fig = viz.plot_lift_precision_scatter(rules_df, save_path='scatter.png')

# 热力图
fig = viz.plot_heatmap(correlation_matrix, save_path='heatmap.png')

# 决策树图
fig = viz.plot_decision_tree(model, feature_cols, save_path='tree.png')

# 导出规则
viz.export_rules(rules_df, 'rules', export_format='csv')  # 'csv'/'json'/'excel'

# 生成综合报告
viz.generate_rule_report(rules_df, report_path='./report')
```

#### 6.2 便捷函数

```python
from rulelift.visualization import (
    plot_rule_comparison, plot_rule_distribution,
    plot_lift_precision_scatter, plot_heatmap,
    generate_rule_report
)

fig = plot_rule_comparison(rules_df)
fig = plot_rule_distribution(rules_df, metric='lift')
fig = plot_lift_precision_scatter(rules_df)
fig = plot_heatmap(corr_matrix)
generate_rule_report(rules_df, report_path='./report')
```

**rules_df 所需列**: `rule_description`, `lift`, `badrate`, `sample_count`, `precision`（按需）

---

### 七、Pipeline

#### 7.1 RuleMiningPipeline

一键完成全流程规则挖掘。

```python
from rulelift.pipeline import RuleMiningPipeline

pipeline = RuleMiningPipeline(
    df,
    target_col='label',
    exclude_cols=['user_id', 'repay_datetime'],

    # OOT分割
    date_col='repay_datetime',
    oot_split_date='2026-02-01',

    # 内存管理
    memory_mode='auto',          # 'auto' | 'full' | 'low'
    min_free_memory_mb=500,

    # 特征选择
    select_iv_threshold=0.02,
    select_psi_threshold=0.25,
    select_max_features=None,    # None=不限制

    # 变量分析
    variable_binning_method='chi2',
    variable_n_bins=10,
    variable_n_jobs=-1,

    # 单特征规则
    single_iv_threshold=0.1,    # 使用 IV>=0.1 的特征
    single_top_n=10,
    single_min_lift=1.1,

    # 交叉特征规则
    cross_iv_threshold=0.05,
    cross_top_features=3,
    cross_max_pairs=6,

    # 树模型规则
    tree_algorithm='rf',
    tree_max_depth=3,
    tree_n_estimators=10,

    # 特征趋势约束
    feature_trends='auto',

    # 功能开关
    enable_variable_analysis=True,
    enable_single_rules=True,
    enable_cross_rules=True,
    enable_tree_rules=True,

    verbose=True
)

results = pipeline.fit()
```

**执行流程**: 数据验证 → 变量分析 → 特征分组 → 单特征挖掘 → 交叉特征挖掘 → 树模型挖掘 → 结果汇总

---

#### 7.2 RuleMiningResults

Pipeline 返回的结果对象。

```python
# 获取所有规则（合并排序）
all_rules = results.get_all_rules(sort_by='lift', min_lift=1.2)

# 按类型获取
single = results.get_single_rules(n=10, sort_by='lift')
cross = results.get_cross_rules()
tree = results.get_tree_rules()

# Top N 规则
top = results.get_top_rules(n=10, metric='lift', rule_type='single')

# 汇总
summary = results.get_summary()

# 导出 Excel
results.to_excel('results.xlsx')

# 可视化摘要（特征分组饼图 + 规则类型条形图）
fig = results.plot_summary()
```

| 方法 | 说明 | 返回 |
|------|------|------|
| `get_all_rules(sort_by, ascending, min_lift, min_samples)` | 合并所有规则 | DataFrame |
| `get_single_rules(n, sort_by)` | 获取单特征规则 | DataFrame |
| `get_cross_rules(n, sort_by)` | 获取交叉规则 | DataFrame |
| `get_tree_rules(n, sort_by)` | 获取树模型规则 | DataFrame |
| `get_top_rules(n, metric, rule_type)` | Top N 规则 | DataFrame |
| `get_summary()` | 汇总统计 | DataFrame |
| `to_excel(path)` | 导出 Excel（多Sheet） | None |
| `plot_summary()` | 绘制摘要图（特征分组饼图 + 规则类型条形图） | Figure |

---

## 内存优化与性能

### 内存优化策略

| 优化技术 | 说明 | 效果 |
|---------|------|------|
| **批处理** | 动态调整批次大小，每批后gc.collect() | 减少50%内存峰值 |
| **Numpy向量化** | 使用np.digitize代替pd.cut | 减少80%临时内存 |
| **缓存机制** | 分箱结果缓存，避免重复计算 | 提升30%速度 |
| **内存监控** | 实时监控，自动降级 | 避免OOM崩溃 |

### 大数据集配置建议

```python
# 场景1: 百万级样本 × 千级特征
pipeline = RuleMiningPipeline(
    df,
    target_col='label',
    memory_mode='auto',
    select_max_features=500,
    variable_n_jobs=1,
    enable_auto_cleanup=True
)

# 场景2: 服务器大内存 (>16GB)
pipeline = RuleMiningPipeline(
    df,
    target_col='label',
    memory_mode='full',
    variable_n_jobs=-1,
    select_max_features=None
)
```

### 实际测试结果

| 数据规模 | 特征数 | 耗时 | 内存峰值 |
|---------|-------|------|---------|
| 73K × 12,327 | 12,325 (含OOT PSI) | ~13min | ~14GB |
| 73K × 12,327 | Pipeline fit (无OOT) | ~26min | ~28GB |
| 73K × 12,327 | Pipeline fit (含OOT) | ~25min | ~28GB |
| 26K × 14,468 | 50 (子集测试) | ~18s | ~4GB |
| 26K × 14,468 | Pipeline fit (50特征, 含OOT) | ~1.5s | ~4GB |

---

## 最佳实践

### 1. 完整分析工作流

```python
from rulelift import VariableAnalyzer, RuleMiningPipeline

# Step 1: Pipeline一键分析
pipeline = RuleMiningPipeline(df, target_col='label', select_max_features=100)
results = pipeline.fit()

# Step 2: 查看变量分析
top_iv = results.variable_analysis.nlargest(10, 'iv')

# Step 3: 查看规则
print(results.single_rules.sort_values('lift', ascending=False).head(10))
```

### 2. 自定义分箱

```python
custom_bins = {
    'age': [18, 25, 35, 45, 55, 65],
    'city': [['北京', '上海'], ['深圳', '广州'], ['其他']]
}

analyzer = VariableAnalyzer(df, target_col='label')
detail = analyzer.analyze_variables_detail(
    variables=['age', 'city'],
    custom_bins_params=custom_bins,
    visualize=True
)
```

### 3. OOT稳定性分析

```python
result = analyzer.analyze_all_variables(
    oot_split_date='2026-02-01',
    date_col='repay_datetime'
)
stable = result[result['psi'] < 0.1]
print(f"稳定特征数: {len(stable)}")
```

### 4. 规则描述评估

```python
from rulelift.analysis import evaluate_rule_description

rules = [
    {'overdue_days': [90, None]},         # 逾期天数 >= 90
    {'history_num': [None, 5]},          # 历史次数 <= 5
    {'app_type': ['TYPE_A', 'TYPE_B']}, # 特定产品类型
    {'pd123': [0.5, None], 'overdue_days': [30, None]},  # 多条件
]
result = evaluate_rule_description(rules, df, target_col='label')
print(result[['rule_description', 'badrate', 'lift', 'cum_total_pct']])
```

---

## 架构文档

### 项目结构

```
rulelift/
├── pipeline.py                 # RuleMiningPipeline 一体化流程
├── analysis/                   # 分析模块
│   ├── variable_analysis.py    # 变量分析 (VariableAnalyzer)
│   ├── rule_analysis.py        # 规则评估 (evaluate_rule_description 等)
│   └── strategy_analysis.py    # 策略分析 (calculate_strategy_gain)
├── mining/                     # 规则挖掘模块
│   ├── single_feature.py       # 单特征挖掘 (SingleFeatureRuleMiner)
│   ├── multi_feature.py        # 交叉特征挖掘 (MultiFeatureRuleMiner)
│   ├── tree_rule_extractor.py  # 统一树模型 (TreeRuleExtractor: dt/rf/gbdt/chi2/isf)
│   ├── decision_tree.py        # 决策树 (DecisionTreeRuleExtractor)
│   └── rule_validator.py       # 规则验证 (RuleValidator)
├── metrics/                    # 指标计算模块
│   ├── basic.py                # 基础指标 (trends, cumulative, correlation)
│   ├── advanced.py             # 高级指标 (strategy pair gain)
│   └── stability.py            # 稳定性指标 (PSI, stability)
├── visualization/              # 可视化模块
│   └── rule.py                 # RuleVisualizer + 便捷函数
├── utils/                      # 工具模块
│   ├── binning_calculator.py   # UnifiedBinningCalculator
│   ├── categorical.py           # 类别变量处理
│   ├── data_loader.py          # 加载示例数据
│   ├── data_processing.py      # 数据预处理
│   ├── validation.py           # 列验证
│   └── parallel.py             # 并行执行器
└── base/                       # 基础模块
    ├── analyzer_base.py        # BaseAnalyzer, DataQualityChecker
    └── pipeline_result.py      # RuleMiningResults
```

---

## 常见问题

### Q1: 如何选择分箱方法？

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| `chi2` | 基于统计显著性，自动合并 | 数据分布不均匀，需要业务解释 |
| `quantile` | 等频分箱，样本均匀分布 | 数据分布相对均匀 |

### Q2: IV/KS/PSI 如何解读？

| 指标 | 强 | 中 | 弱 |
|------|-----|-----|-----|
| IV | > 0.3 | 0.1~0.3 | < 0.1 |
| KS | > 0.3 | 0.2~0.3 | < 0.2 |
| PSI | < 0.1 (稳定) | 0.1~0.25 | > 0.25 |

### Q3: 如何处理大规模数据？

```python
pipeline = RuleMiningPipeline(
    df, target_col='label',
    memory_mode='auto',
    select_max_features=500,
    enable_auto_cleanup=True
)
```

### Q4: DecisionTreeRuleExtractor 报错 dtype 不兼容？

v1.5.1 已自动排除 datetime/timedelta 列，无需手动处理。如果使用旧版本，可手动排除：
```python
exclude = ['date_col'] + [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
extractor = DecisionTreeRuleExtractor(df, target_col='label', exclude_cols=exclude)
```

### Q5: TreeRuleExtractor.evaluate_rules() 报错参数错误？

`TreeRuleExtractor.evaluate_rules()` 无需传入 rules 参数：
```python
extractor.train()
rules = extractor.extract_rules()
result = extractor.evaluate_rules()  # 正确：不传参
```

---

## 更新日志

### v1.6.0 (最新)
- 新增简化调用别名：核心类提供更短的方法名（如 `.vars()`、`.rules()`、`.perf()`）

### v1.5.1
- 修复 DecisionTreeRuleExtractor/TreeRuleExtractor 不自动排除 datetime 列导致 sklearn 崩溃
- 修复 DecisionTreeRuleExtractor/TreeRuleExtractor 遇到 dict/list/混合类型列时 LabelEncoder 报错
- 修复 DecisionTreeRuleExtractor 高级验证模式下 train/test 分割使用未编码数据

### v1.5.0
- 统一 feature_trends 特征趋势约束
- 新增 `compute_feature_trends()` 自动推断特征趋势方向
- 新增 `evaluate_rule_description()` 规则描述直接评估
- 新增 `add_cumulative_metrics()` 累计指标计算
- 新增 MultiFeatureRuleMiner `get_top_rules_histogram()`
- 所有挖掘器输出均包含累计指标列
- Pipeline feature_trends 参数透传

### v1.4.0
- 新增 RuleMiningPipeline 一体化分析流程
- 内存优化：批处理 + numpy向量化
- 支持大规模数据（万级特征）
- 新增二元特征处理

### v1.1.0
- 新增 TreeRuleExtractor
- 新增 MultiFeatureRuleMiner

### v1.0.0
- 首次发布

---

## 许可证

MIT License

---

## 联系方式

- GitHub: https://github.com/aialgorithm/rulelift
- Issues: https://github.com/aialgorithm/rulelift/issues
- Email: 15880982687@qq.com

---

<a name="english-version"></a>

# English Version

## Project Overview

**RuleLift** is a professional **Python credit risk management toolkit**, focused on **rule mining**, **rule evaluation**, and **rule monitoring**.

### Why RuleLift?

| Traditional Pain Point | RuleLift Solution |
|-----------------------|-------------------|
| Hard to monitor online rules: intercepted customers lack performance data | Real-time rule evaluation based on user rating distribution, no A/B testing needed |
| Complex rule mining: manual mining is time-consuming | Automatically mine high-value business rules from data |
| Tedious feature analysis: switching between multiple tools | All-in-one IV/KS/AUC/PSI analysis |
| Large data processing: OOM crashes | Memory-optimized design, supports 10K+ features, million-level samples |

### Core Capabilities

```
RuleLift
├── Rule Intelligence   - Evaluate rule performance without A/B testing
├── Auto Rule Mining    - Single feature, cross feature, tree model mining
├── Deep Variable Analysis - Comprehensive IV/KS/AUC/PSI metrics
├── Memory Optimization - Batching, vectorization, caching for large-scale data
└── One-stop Pipeline   - Automated full-process rule mining
```

---

## Quick Start

### Installation

```bash
pip install rulelift
```

**Requirements**: Python >= 3.8 | pandas >= 1.0.0 | numpy >= 1.18.0 | scikit-learn >= 0.24.0 | matplotlib >= 3.3.0

### 5-Minute Getting Started

```python
from rulelift import RuleMiningPipeline

import pandas as pd
df = pd.read_csv('your_data.csv')

# One-click full analysis
pipeline = RuleMiningPipeline(
    df=df,
    target_col='ISBAD',
    exclude_cols=['ID', 'CREATE_TIME'],
    select_max_features=100,
    enable_variable_analysis=True,
    enable_single_rules=True,
    enable_cross_rules=True,
    enable_tree_rules=True,
    verbose=True
)

results = pipeline.fit()

# View results
print(results.get_summary())

# Get all rules
all_rules = results.get_all_rules()
all_rules.to_excel('rules_output.xlsx')
```

---

## Simplified Aliases

Core classes provide simplified alias methods for zero-overhead convenience.

### Comparison

```python
from rulelift import VariableAnalyzer, SingleFeatureRuleMiner, DecisionTreeRuleExtractor

# === Traditional Calls ===
result = analyzer.analyze_all_variables(oot_split_date='2026-02-01', date_col='repay_datetime')
detail = analyzer.analyze_variables_detail(variables=['age', 'income'], visualize=True)
rules = miner.get_top_rules(feature=['age', 'income'], top_n=10)
perf = extractor.perf()

# === Simplified Calls (equivalent) ===
result = analyzer.vars(oot_split_date='2026-02-01', date_col='repay_datetime')
detail = analyzer.vars_detail(variables=['age', 'income'], visualize=True)
rules = miner.rules(feature=['age', 'income'], top_n=10)
perf = extractor.perf()
```

### Complete Alias List

| Class | Alias | Original Method | Description |
|-------|-------|----------------|-------------|
| **VariableAnalyzer** | `.vars()` | `.analyze_all_variables()` | Analyze all variables |
| | `.vars_detail()` | `.analyze_variables_detail()` | Detailed variable analysis |
| | `.vars_one()` | `.analyze_variables_detail()` | Analyze single variable |
| | `.select()` | `.select_features()` | Feature selection |
| | `.plot_bins()` | `.plot_variable_bins()` | Plot binning chart |
| | `.quality()` | `.check_data_quality()` | Data quality check |
| | `.psi()` | `.calculate_psi()` | Calculate PSI |
| **SingleFeatureRuleMiner** | `.rules()` | `.get_top_rules()` | Get single feature rules |
| **MultiFeatureRuleMiner** | `.rules()` | `.get_top_rules()` | Get cross feature rules |
| | `.rules_hist()` | `.get_top_rules_histogram()` | Histogram threshold search |
| | `.cross_matrix()` | `.generate_cross_matrix()` | Generate cross matrix |
| | `.cross_excel()` | `.generate_cross_matrices_excel()` | Export cross rules to Excel |
| | `.heatmap()` | `.plot_cross_heatmap()` | Cross feature heatmap |
| **DecisionTreeRuleExtractor** | `.rules_list()` | `.get_rules_as_dataframe()` | Get rules as DataFrame |
| | `.top_rules()` | `.get_top_rules()` | Get Top N rules |
| | `.importance()` | `.get_feature_importance()` | Feature importance |
| | `.perf()` | `N/A` | Model performance |
| | `.generalize()` | `.analyze_rule_generalization()` | Rule generalization |
| **TreeRuleExtractor** | `.importance()` | `.get_feature_importance()` | Feature importance |
| **RuleMiningResults** | `.all()` | `.get_all_rules()` | Get all rules |
| | `.top()` | `.get_top_rules()` | Get Top N rules |

---

## Core Features

### 1. Rule Intelligence Evaluation

Evaluate rule performance based on user rating distributions without A/B testing.

**Supported Metrics**:
- **Estimated metrics**: Bad rate, Lift, Recall, Precision
- **Actual metrics**: F1 Score, Actual bad rate, Actual lift
- **Stability metrics**: Hit rate std, Coefficient of variation

### 2. Auto Rule Mining

Multiple mining algorithms for different business scenarios:

| Algorithm | Use Case | Characteristics |
|-----------|----------|----------------|
| `SingleFeatureRuleMiner` | Fast strong feature discovery | Single feature optimal threshold mining, memory optimized |
| `MultiFeatureRuleMiner` | Improve rule coverage | Cross feature combinations, numpy vectorized |
| `TreeRuleExtractor('dt')` | Quick rule generation | Decision tree, simple and intuitive |
| `TreeRuleExtractor('rf')` | Need stable rules | Random forest, multi-tree ensemble |
| `TreeRuleExtractor('gbdt')` | Pursue high accuracy | Gradient boosting trees |
| `TreeRuleExtractor('chi2')` | Auto-binning + random forest | Chi-square auto-binning then random forest |
| `TreeRuleExtractor('isf')` | Anomaly detection | Isolation forest, discovers risk rules via anomaly scores |

### 3. Deep Variable Analysis

Comprehensive variable evaluation:

| Metric | Description | Application | Criteria |
|--------|------------|-------------|----------|
| IV (Information Value) | Predictive power | Feature selection | >0.3 strong, 0.02-0.1 medium, <0.02 weak |
| KS (Kolmogorov-Smirnov) | Discriminative power | Binning evaluation | >0.3 strong, 0.2-0.3 medium, <0.2 weak |
| AUC | Prediction accuracy | Model evaluation | >0.7 good |
| PSI (Population Stability) | Variable stability | Feature drift monitoring | <0.1 stable, >0.25 unstable |

### 4. Strategy Optimization

Calculate marginal gains for rule combinations to find optimal strategy combinations.

---

## Pipeline Reference

`RuleMiningPipeline` integrates all functionalities for one-click full analysis.

### Complete Parameters

```python
from rulelift.pipeline import RuleMiningPipeline

pipeline = RuleMiningPipeline(
    df=data,
    target_col='ISBAD',

    # === Data Configuration ===
    exclude_cols=['ID', 'TIME'],
    amount_col='AMOUNT',
    ovd_bal_col='OVD_BAL',
    date_col='CREATE_TIME',
    oot_split_date='2024-01-01',

    # === Feature Selection ===
    select_iv_threshold=0.02,
    select_max_features=100,
    select_psi_threshold=None,       # None = no PSI filtering

    # === Variable Analysis ===
    variable_binning_method='chi2',
    variable_n_bins=10,
    variable_min_samples_pct=0.05,
    variable_chi2_threshold=3.841,
    variable_n_jobs=-1,

    # === Single Feature Rules ===
    single_iv_threshold=0.1,
    single_top_n=10,
    single_min_lift=1.1,
    single_min_samples=10,
    single_algorithm='histogram',
    single_n_jobs=-1,

    # === Cross Feature Rules ===
    cross_iv_threshold=0.05,
    cross_top_features=3,
    cross_top_n=5,
    cross_min_samples=10,
    cross_min_lift=1.1,
    cross_n_bins=8,
    cross_max_pairs=6,

    # === Tree Model Rules ===
    tree_algorithm='rf',              # 'dt', 'rf', 'gbdt', 'chi2', 'isf'
    tree_max_depth=3,
    tree_min_samples_leaf=5,
    tree_n_estimators=10,
    tree_max_features='sqrt',
    tree_top_n=20,

    # === Global Controls ===
    feature_trends='auto',           # Dict / 'auto' / None
    enable_variable_analysis=True,
    enable_single_rules=True,
    enable_cross_rules=True,
    enable_tree_rules=True,
    enable_validation=False,
    random_state=42,
    verbose=True,

    # === Memory Management ===
    memory_mode='auto',              # 'auto', 'full', 'low'
    min_free_memory_mb=500,
    enable_auto_cleanup=True,
    auto_skip_on_low_memory=False,
)

results = pipeline.fit()
```

### Pipeline Execution Flow

```
Step 0: Data Validation
  └─> Validate data integrity and target column

Step 1: Variable Analysis
  └─> Calculate IV/KS/AUC/PSI for all variables

Step 2: Feature Grouping
  └─> Group by IV thresholds: High | Mid | Low

Step 3: Single Feature Rule Mining
  └─> Threshold mining for high-IV features

Step 4: Cross Feature Rule Mining
  └─> Cross combination mining for mid-IV features

Step 5: Tree Model Rule Mining
  └─> Decision tree / random forest / GBDT rule extraction

Step 6: Result Aggregation
```

---

## Full API Reference

### I. Utility Functions (utils/)

#### load_example_data

Load built-in example data.

```python
from rulelift.utils import load_example_data
df = load_example_data()  # 998 rows × 6 columns
```

#### preprocess_data

Preprocess data, convert percentage strings to floats.

```python
from rulelift.utils import preprocess_data
df = preprocess_data(df, user_level_badrate_col='BADRATE')
```

#### UnifiedBinningCalculator

Unified binning calculator supporting multiple binning methods.

```python
from rulelift.utils import UnifiedBinningCalculator
import numpy as np

calc = UnifiedBinningCalculator(n_bins=10, default_method='chi2')

# Compute bin edges (pass numpy arrays)
bins = calc.compute_bins(df['feature'].values, df['target'].values, n_bins=10)

# Compute bin statistics (returns tuple: (stats_df, iv, ks))
stats_df, iv, ks = calc.compute_bin_stats(df['feature'].values, df['target'].values, bins)

# Apply bins
binned = calc.apply_bins(df['feature'].values, bins)
```

| Constructor Parameter | Type | Default | Description |
|----------------------|------|---------|-------------|
| `default_method` | str | `'quantile'` | Binning method: `'quantile'`/`'chi2'`/`'equal_width'` |
| `n_bins` | int | 10 | Default bin count |
| `chi2_threshold` | float | 3.841 | Chi-square threshold |
| `min_samples_pct` | float | 0.02 | Minimum sample percentage |
| `decimal_places` | int | 3 | Decimal precision |
| `robust_mode` | bool | True | Robust mode (fallback on errors) |

#### CategoricalVariableProcessor

Automatic categorical variable detection and processing.

```python
from rulelift.utils.categorical import CategoricalVariableProcessor

proc = CategoricalVariableProcessor()
info = proc.detect_and_prepare(df, 'app_type', 'label')
# info: {'feature': 'app_type', 'method': '...', 'detection': {...}, 'bin_mapping': {...}}
```

---

### II. Metrics (metrics/)

#### compute_feature_trends

Auto-detect feature trend direction (based on correlation).

```python
from rulelift.metrics import compute_feature_trends

trends = compute_feature_trends(df, ['age', 'income'], target_col='label')
# {'age': 1, 'income': -1}
# 1 = positive correlation, -1 = negative correlation
```

#### add_cumulative_metrics

Add cumulative metrics to rule results.

```python
from rulelift.metrics import add_cumulative_metrics

# DataFrame must contain 'selected_samples' and 'selected_bad' columns
rules_df = add_cumulative_metrics(rules_df, sort_by='threshold', ascending=True)
# Adds: cum_total_pct, cum_bad_rate, cum_bad_rate_remaining
```

#### calculate_psi

Calculate Population Stability Index.

```python
from rulelift.metrics import calculate_psi

psi = calculate_psi(train_data, oot_data, buckets=10)
# <0.1 stable, 0.1-0.25 moderate, >0.25 unstable
```

#### Stability Metrics

```python
from rulelift.metrics import calculate_rule_psi, calculate_rule_stability, calculate_long_term_stability

# Rule PSI over time periods
psi_df = calculate_rule_psi(rule_score, 'RULE', 'HIT_DATE', 'USER_ID')

# Monthly rule stability
stability = calculate_rule_stability(rule_score, 'RULE', 'HIT_DATE', 'USER_ID')

# Long-term stability (rolling window)
long_term = calculate_long_term_stability(rule_score, 'RULE', 'HIT_DATE', 'USER_ID', window_months=6)
```

---

### III. Variable Analysis (analysis/VariableAnalyzer)

#### Constructor

```python
from rulelift.analysis import VariableAnalyzer

analyzer = VariableAnalyzer(
    df,
    target_col='label',
    exclude_cols=['user_id', 'date_col'],
    n_bins=10,
    binning_method='chi2',          # 'chi2' | 'quantile'
    min_samples_pct=0.02,
    n_jobs=-1,
    log_level='INFO'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input dataset |
| `target_col` | str | `'ISBAD'` | Target column |
| `exclude_cols` | list | None | Columns to exclude |
| `amount_col` | str | None | Amount column (optional) |
| `ovd_bal_col` | str | None | Overdue balance column (optional) |
| `n_bins` | int | 10 | Default bin count |
| `binning_method` | str | `'chi2'` | Binning method |
| `chi2_threshold` | float | 3.841 | Chi-square threshold |
| `min_samples_pct` | float | 0.02 | Minimum bin sample percentage |
| `iv_calculation_method` | str | `'standard'` | IV calculation method |
| `n_jobs` | int | -1 | Parallel processes (-1 = all cores) |
| `enable_adaptive_parallel` | bool | True | Adaptive parallel (memory-aware) |
| `memory_threshold_mb` | float | 500 | Memory threshold (MB) |
| `gc_interval` | int | 5 | GC interval |
| `log_level` | str | `'INFO'` | Log level |

#### analyze_all_variables

> Alias: `.vars()`

Analyze all variables, computing IV/KS/AUC/PSI.

```python
result = analyzer.analyze_all_variables(
    oot_split_date='2026-02-01',
    date_col='repay_datetime',
    include_categorical=True,
    show_progress=True,
    batch_size=20,
    sample_size=None
)
```

**Returns**: `pd.DataFrame` — one row per feature with `variable`, `iv`, `ks`, `auc`, `gini`, `psi` columns

#### analyze_variables_detail

> Alias: `.vars_detail()` / `.vars_one()`

Detailed binning analysis for specific variables.

```python
detail = analyzer.analyze_variables_detail(
    variables=['age', 'income'],
    n_bins=10,
    visualize=True,
    custom_bins_params={
        'age': [18, 25, 35, 45, 55, 65],
        'city': [['Beijing', 'Shanghai'], ['Shenzhen', 'Guangzhou'], ['Other']]
    },
    oot_split_date='2026-02-01',
    date_col='repay_datetime',
    binning_method='chi2'
)
```

**Returns**: `pd.DataFrame` — binning statistics

#### select_features

> Alias: `.select()`

Multi-dimensional feature selection.

```python
result = analyzer.select_features(
    iv_threshold=0.02,
    psi_threshold=0.25,
    ks_threshold=0.02,
    correlation_threshold=0.85
)
# Returns dict: {
#     'selected_features': [...],
#     'selected_df': DataFrame,
#     'rejected_features': {...},
#     'correlation_removed': {...},
#     'summary': {...}
# }
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analysis_result` | DataFrame | None | Custom analysis result (None = use cache) |
| `iv_threshold` | float | 0.02 | Minimum IV |
| `missing_rate_threshold` | float | 0.8 | Maximum missing rate |
| `single_value_rate_threshold` | float | 0.95 | Maximum single-value rate |
| `psi_threshold` | float | 0.25 | Maximum PSI |
| `ks_threshold` | float | 0.02 | Minimum KS |
| `correlation_threshold` | float | 0.85 | Maximum correlation |
| `mode` | str | `'and'` | Filter mode: `'and'`/`'or'` |

**Returns**: `Dict` — with keys `selected_features`, `selected_df`, `rejected_features`, `correlation_removed`, `summary`

---

### IV. Rule Analysis (analysis/)

#### evaluate_rule_description

Evaluate rules directly from rule descriptions (no pre-computed hit matrix needed).

```python
from rulelift.analysis import evaluate_rule_description

results = evaluate_rule_description(
    [
        {'age': [60, None]},            # age >= 60
        {'income': [None, 5000]},      # income <= 5000
        {'city': ['Beijing', 'Shanghai']},  # city in [...]
        {'age': [30, 50], 'city': 'Beijing'},  # Multi-condition AND
    ],
    df=df,
    target_col='label'
)
```

**Supported Rule Formats**:

| Format | Example | Meaning |
|--------|---------|---------|
| Numeric >= | `{'age': [60, None]}` | age >= 60 |
| Numeric <= | `{'age': [None, 80]}` | age <= 80 |
| Numeric range | `{'age': [60, 80]}` | 60 <= age <= 80 |
| Category match | `{'city': 'Beijing'}` | city == 'Beijing' |
| Category list | `{'city': ['Beijing', 'Shanghai']}` | city in [...] |
| Multi-condition AND | `{'age': [60, None], 'city': 'Beijing'}` | All conditions must match |

---

### V. Rule Mining (mining/)

> **Deprecated**: `XGBoostRuleMiner` is deprecated. Use `TreeRuleExtractor(algorithm='gbdt')` instead. The `'xgb'` algorithm identifier is also deprecated and auto-converted to `'gbdt'`.

#### 5.1 SingleFeatureRuleMiner

Single feature rule miner via threshold search.

```python
from rulelift.mining import SingleFeatureRuleMiner

miner = SingleFeatureRuleMiner(
    df, target_col='label',
    exclude_cols=['user_id'],
    min_lift=1.1,
    algorithm='histogram',     # 'histogram' | 'chi2'
    n_jobs=-1,
    feature_trends='auto'
)

rules = miner.get_top_rules(
    feature=['age', 'income'],
    top_n=10,
    min_samples=10,
    group_by_feature=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Dataset |
| `target_col` | str | `'ISBAD'` | Target column |
| `exclude_cols` | list | None | Columns to exclude |
| `algorithm` | str | `'histogram'` | Algorithm: `'histogram'`/`'chi2'` |
| `min_lift` | float | 1.1 | Minimum lift value |
| `histogram_bins` | int | 100 | Histogram bin count |
| `chi2_threshold` | float | 3.841 | Chi-square threshold |
| `n_jobs` | int | -1 | Parallel process count |
| `feature_trends` | dict/str | None | Feature trend constraints |
| `missing_threshold` | float | 0.95 | Missing rate threshold |
| `missing_strategy` | str | `'fill'` | Missing value strategy |
| `test_size` | float | 0.2 | Test set ratio |
| `validation_mode` | str | `'split'` | Validation mode: `'split'`/`'oot'` |

**Returns**: `pd.DataFrame` — with `feature`, `threshold`, `operator`, `lift`, `badrate`, `selected_samples` etc.

#### 5.2 MultiFeatureRuleMiner

Cross feature rule miner.

```python
from rulelift.mining import MultiFeatureRuleMiner

miner = MultiFeatureRuleMiner(df, target_col='label')

# Grid binning method
rules = miner.get_top_rules(
    feature1='age', feature2='income',
    top_n=10, min_samples=10, min_lift=1.1
)

# Histogram threshold search
rules = miner.get_top_rules_histogram(
    feature1='age', feature2='income',
    top_n=10, min_samples=10, min_lift=1.1
)

# Cross matrix
cross_matrix = miner.generate_cross_matrix('age', 'income')

# Heatmap
miner.plot_cross_heatmap('age', 'income', metric='lift', save_path='heatmap.png')
```

> **Note**: `MultiFeatureRuleMiner` has no `exclude_cols` parameter.

#### 5.3 DecisionTreeRuleExtractor

Decision tree based rule extraction.

```python
from rulelift.mining import DecisionTreeRuleExtractor

extractor = DecisionTreeRuleExtractor(
    df, target_col='label',
    exclude_cols=['user_id', 'repay_datetime'],
    max_depth=5, min_samples_leaf=5
)

train_acc, test_acc = extractor.train()
rules = extractor.extract_rules()
evaluation = extractor.evaluate_rules(rules)  # Accepts DataFrame or None
importance = extractor.get_feature_importance()
```

> **Auto-excludes** datetime/timedelta columns (no manual exclusion needed).

#### 5.4 TreeRuleExtractor

Unified tree model rule extractor supporting 5 algorithms: dt/rf/gbdt/chi2/isf.

```python
from rulelift.mining import TreeRuleExtractor

extractor = TreeRuleExtractor(
    df, target_col='label',
    algorithm='rf',              # 'dt' | 'rf' | 'gbdt' | 'chi2' | 'isf'
    max_depth=3,
    min_samples_leaf=5,
    n_estimators=10,
    feature_trends='auto'
)

extractor.train()
rules = extractor.extract_rules()
result = extractor.evaluate_rules()   # No arguments needed (except 'isf')
```

**Algorithm Details**:

| Algorithm | Use Case | Description |
|-----------|----------|-------------|
| `dt` | Quick rule generation | Single decision tree |
| `rf` | Need stable rules | Random forest ensemble |
| `gbdt` | Pursue high accuracy | Gradient boosting (set `learning_rate`, `subsample`) |
| `chi2` | Auto-binning + RF | Chi-square auto-binning then random forest (set `min_bin_ratio`) |
| `isf` | Anomaly detection | Isolation forest via anomaly scores. **Note**: `evaluate_rules()` not supported |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | str | `'rf'` | Algorithm: `'dt'`/`'rf'`/`'gbdt'`/`'chi2'`/`'isf'` |
| `max_depth` | int | 3 | Maximum depth |
| `min_samples_leaf` | int/float | 5 | Minimum leaf samples (supports float ratio) |
| `n_estimators` | int | 10 | Tree count |
| `max_features` | str | `'sqrt'` | Max features per split |
| `learning_rate` | float | 0.1 | Learning rate (gbdt) |
| `subsample` | float | 1.0 | Subsample ratio (gbdt) |
| `min_bin_ratio` | float | 0.05 | Min bin ratio (chi2) |
| `isf_weights` | dict | None | Isolation forest rule weight config |
| `test_size` | float | 0.3 | Test set ratio |
| `random_state` | int | 42 | Random seed |

**`isf_weights` Options** (isolation forest rule scoring):

| Key | Default | Description |
|-----|---------|-------------|
| `purity` | 0.5 | Bad customer purity weight |
| `anomaly` | 0.3 | Anomaly score weight |
| `sample` | 0.15 | Sample count weight |
| `hit` | 0.05 | Anomaly bad customer hit ratio weight |

**Important**: `evaluate_rules()` takes no arguments (uses internally extracted rules). `isf` algorithm does not support rule evaluation.

#### 5.5 RuleValidator

Standalone rule validator supporting split/OOT validation modes.

```python
from rulelift.mining import RuleValidator

validator = RuleValidator(df, target_col='label', validation_mode='split')

# Split data first (required)
validator.split_train_test()

# Evaluate a single rule
result = validator.evaluate_rule("feature1 > 100")

# Batch evaluate
results = validator.evaluate_rules(["feature1 > 100", "feature2 <= 50"])
comparison = validator.compare_train_test_performance(results)
validator.print_validation_report(comparison)
```

> `RuleValidatorMixin` is inherited by `DecisionTreeRuleExtractor` and `TreeRuleExtractor` automatically.

---

### VI. Visualization (visualization/)

#### RuleVisualizer

```python
from rulelift.visualization import RuleVisualizer

viz = RuleVisualizer(dpi=300)

fig = viz.plot_rule_comparison(rules_df, metrics=['lift', 'badrate'])
fig = viz.plot_rule_distribution(rules_df, metric='lift')
fig = viz.plot_lift_precision_scatter(rules_df)
fig = viz.plot_heatmap(correlation_matrix)
```

---

### VII. Pipeline Results (base/RuleMiningResults)

```python
# Get all rules (merged and sorted)
all_rules = results.get_all_rules(sort_by='lift', min_lift=1.2)

# By type
single = results.get_single_rules(n=10, sort_by='lift')
cross = results.get_cross_rules()
tree = results.get_tree_rules()

# Top N
top = results.get_top_rules(n=10, metric='lift', rule_type='single')

# Summary
summary = results.get_summary()

# Export Excel
results.to_excel('results.xlsx')

# Visualization (feature group pie chart + rule type bar chart)
fig = results.plot_summary()
```

| Method | Description | Returns |
|--------|-------------|---------|
| `get_all_rules(sort_by, ascending, min_lift, min_samples)` | Merge all rules | DataFrame |
| `get_single_rules(n, sort_by)` | Get single feature rules | DataFrame |
| `get_cross_rules(n, sort_by)` | Get cross feature rules | DataFrame |
| `get_tree_rules(n, sort_by)` | Get tree model rules | DataFrame |
| `get_top_rules(n, metric, rule_type)` | Top N rules | DataFrame |
| `get_summary()` | Summary statistics | DataFrame |
| `to_excel(path)` | Export Excel (multi-sheet) | None |
| `plot_summary()` | Plot summary (pie + bar chart) | Figure |

---

## Memory Optimization & Performance

### Optimization Strategies

| Technique | Description | Effect |
|-----------|-------------|--------|
| **Batching** | Dynamic batch sizes with gc.collect() | -50% memory peak |
| **Numpy Vectorization** | np.digitize instead of pd.cut | -80% temp memory |
| **Caching** | Bin results cached to avoid recomputation | +30% speed |
| **Memory Monitoring** | Real-time monitoring, auto-degradation | Prevent OOM |

### Large Dataset Configuration

```python
# Million-level samples × thousand-level features
pipeline = RuleMiningPipeline(
    df, target_col='label',
    memory_mode='auto',
    select_max_features=500,
    variable_n_jobs=1,
    enable_auto_cleanup=True
)

# Large memory server (>16GB)
pipeline = RuleMiningPipeline(
    df, target_col='label',
    memory_mode='full',
    variable_n_jobs=-1,
    select_max_features=None
)
```

### Performance Benchmarks

| Dataset Scale | Feature Count | Duration | Peak Memory |
|--------------|--------------|----------|-------------|
| 73K x 12,327 | 12,325 (with OOT PSI) | ~13min | ~14GB |
| 73K x 12,327 | Pipeline fit (no OOT) | ~26min | ~28GB |
| 73K x 12,327 | Pipeline fit (with OOT) | ~25min | ~28GB |
| 26K x 14,468 | 50 (subset test) | ~18s | ~4GB |
| 26K x 14,468 | Pipeline fit (50 features, with OOT) | ~1.5s | ~4GB |

---

## Best Practices

### 1. Complete Analysis Workflow

```python
from rulelift import VariableAnalyzer, RuleMiningPipeline

# Step 1: Pipeline one-click analysis
pipeline = RuleMiningPipeline(df, target_col='label', select_max_features=100)
results = pipeline.fit()

# Step 2: View variable analysis
top_iv = results.variable_analysis.nlargest(10, 'iv')

# Step 3: View rules
print(results.single_rules.sort_values('lift', ascending=False).head(10))
```

### 2. OOT Stability Analysis

```python
result = analyzer.analyze_all_variables(
    oot_split_date='2026-02-01',
    date_col='repay_datetime'
)
stable = result[result['psi'] < 0.1]
print(f"Stable features: {len(stable)}")
```

### 3. Rule Description Evaluation

```python
from rulelift.analysis import evaluate_rule_description

rules = [
    {'overdue_days': [90, None]},
    {'history_num': [None, 5]},
    {'app_type': ['TYPE_A', 'TYPE_B']},
]
result = evaluate_rule_description(rules, df, target_col='label')
print(result[['rule_description', 'badrate', 'lift', 'cum_total_pct']])
```

---

## Architecture

### Project Structure

```
rulelift/
├── pipeline.py                 # RuleMiningPipeline
├── analysis/                   # Analysis module
│   ├── variable_analysis.py    # VariableAnalyzer
│   ├── rule_analysis.py        # Rule evaluation
│   └── strategy_analysis.py    # Strategy analysis
├── mining/                     # Rule mining module
│   ├── single_feature.py       # SingleFeatureRuleMiner
│   ├── multi_feature.py        # MultiFeatureRuleMiner
│   ├── tree_rule_extractor.py  # TreeRuleExtractor (dt/rf/gbdt/chi2/isf)
│   ├── decision_tree.py        # DecisionTreeRuleExtractor
│   └── rule_validator.py       # RuleValidator + RuleValidatorMixin
├── metrics/                    # Metrics module
│   ├── basic.py                # Basic metrics (trends, cumulative, correlation)
│   ├── advanced.py             # Advanced metrics (strategy pair gain)
│   └── stability.py            # Stability metrics (PSI, stability)
├── visualization/              # Visualization module
│   └── rule.py                 # RuleVisualizer + convenience functions
├── utils/                      # Utility module
│   ├── binning_calculator.py   # UnifiedBinningCalculator
│   ├── categorical.py          # Categorical variable processing
│   ├── data_loader.py          # Example data loader
│   ├── data_processing.py      # Data preprocessing
│   ├── validation.py           # Column validation
│   └── parallel.py             # Parallel executor
└── base/                       # Base module
    ├── analyzer_base.py        # BaseAnalyzer, DataQualityChecker
    └── pipeline_result.py      # RuleMiningResults
```

---

## FAQ

### Q1: How to choose a binning method?

| Method | Characteristics | Use Case |
|--------|----------------|----------|
| `chi2` | Statistical significance, auto-merge | Non-uniform distribution, need business interpretation |
| `quantile` | Equal-frequency, uniform samples | Relatively uniform distribution |

### Q2: How to interpret IV/KS/PSI?

| Metric | Strong | Medium | Weak |
|--------|--------|--------|------|
| IV | > 0.3 | 0.1~0.3 | < 0.1 |
| KS | > 0.3 | 0.2~0.3 | < 0.2 |
| PSI | < 0.1 (stable) | 0.1~0.25 | > 0.25 |

### Q3: DecisionTreeRuleExtractor dtype error?

v1.5.1 auto-excludes datetime/timedelta columns. No manual handling needed.

### Q4: TreeRuleExtractor.evaluate_rules() parameter error?

`TreeRuleExtractor.evaluate_rules()` takes no arguments:
```python
extractor.train()
rules = extractor.extract_rules()
result = extractor.evaluate_rules()  # Correct: no arguments
```

### Q5: What about the `isf` (Isolation Forest) algorithm?

The `isf` algorithm discovers risk rules through anomaly detection. Note that `evaluate_rules()` is not supported for `isf`. Use `extract_rules()` to get rules, then evaluate them separately with `evaluate_rule_description()`.

---
---

## License

MIT License

---

## Contact

- GitHub: https://github.com/aialgorithm/rulelift
- Issues: https://github.com/aialgorithm/rulelift/issues
- Email: 15880982687@qq.com


## 作者

微信&github：aialgorithm <15880982687@qq.com>

## 版本信息

- 当前版本：1.2.2
- 发布日期：2025-12-25

## 项目地址

- GitHub: https://github.com/aialgorithm/rulelift
- PyPI: https://pypi.org/project/rulelift/

## 离线使用方式

### 方式一：离线安装rulelift及相关依赖

1. **在有网络的环境中下载依赖包**：

```bash
# 下载rulelift及其所有依赖
pip download rulelift -d ./packages/
```

2. **将下载的packages文件夹传输到离线环境**

3. **在离线环境中安装**：

```bash
# 进入packages文件夹
cd ./packages/

# 安装所有依赖包
pip install *.whl --no-index --find-links=.
```

### 方式二：通过源码直接调用

1. **下载源码**：

   * 从GitHub下载源码包：<https://github.com/aialgorithm/rulelift>

2. **将源码包传输到离线环境并解压**
需要手动安装pandas、numpy、scikit-learn、matplotlib、seaborn
3. **在Python代码中添加源码路径并导入**：

```python
import sys
import os

# 添加源码路径到系统路径
sys.path.append('/path/to/rulelift-master')

# 直接导入模块
from rulelift import load_example_data, analyze_rules, TreeRuleExtractor
```

## 后续维护
如有bug或维护建议，请通过GitHub [Issues](https://github.com/aialgorithm/rulelift/issues) 反馈，我们会尽快响应并解决。

也可以提交Pull Request（PR）来贡献代码。
- 整合多个规则的评估结果，形成策略级结论
- 增强实际场景数据处理能力
- 结果展示&操作可视化
- 考虑敏感信息，暂无法支持AI大模型

## 谢谢使用

-----

<img width="350" height="400" alt="e99495131259259f29088d333b51819" src="https://github.com/user-attachments/assets/b2611aa6-4ff8-40a3-88ba-867bb384a89d" />
