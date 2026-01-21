## rulelift 是一个用于信用风险管理中策略规则的自动挖掘、有效性分析及监控的Python工具包。
- 实时评估上线规则的效度（无需分流测试、无需表现标签）；
- 自动化挖掘高价值的规则（挖掘并评估多种规则、符合业务解释性）；
### 项目统计
[![PyPI Downloads下载量](https://img.shields.io/pypi/dm/rulelift?label=PyPI项目下载量)](https://pypistats.org/packages/rulelift)
[![PyPI version](https://img.shields.io/pypi/v/rulelift.svg)](https://pypi.org/project/rulelift/)

[![Star History Chart](https://api.star-history.com/svg?repos=aialgorithm/rulelift&type=Date)](https://star-history.com/#aialgorithm/rulelift&Date)

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
- **决策树规则提取**：从多种树模型（随机森林、XGB、孤立森林等）中提取可解释的规则
- **可视化支持**：多维度指标直观展示规则效果

基于对上线规则的评估结果，我们可以及时发现规则效率低下或不稳定的问题，从而及时调整规则阈值或删减。也可以结合规则挖掘，新增有效规则，提升规则系统的整体效果及稳定性。

### 快速开始

```bash
# 使用pip安装（推荐）
pip install rulelift

# 从源码安装
pip install git+https://github.com/aialgorithm/rulelift.git

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
### 基本使用示例

```python
# 加载示例数据
from rulelift import load_example_data, analyze_rules, TreeRuleExtractor

# 1. 规则评估示例（使用用户评级评估）
print("=== 规则评估示例（使用用户评级） ===")
# 加载规则命中数据
hit_rule_df = load_example_data('hit_rule_info.csv')

# 使用用户评级评估规则效度
result = analyze_rules(hit_rule_df, user_level_badrate_col='USER_LEVEL_BADRATE')
print(f"规则评估结果（前5条）:")
print(result[['rule', 'estimated_badrate_pred', 'estimated_lift_pred']].head())

# 2. 规则挖掘示例（使用决策树规则挖掘）
print("\n=== 规则挖掘示例（使用决策树） ===")
# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')
# 初始化决策树规则提取器
tree_miner = TreeRuleExtractor(
        feature_df, 
        target_col='ISBAD', 
        exclude_cols=['ID', 'CREATE_TIME','OVD_BAL','AMOUNT'],
        algorithm='dt',
        max_depth=3, 
        min_samples_split=20,
        min_samples_leaf=10,
        test_size=0.3,
        random_state=42
    )
# 训练模型
train_acc, test_acc = tree_miner.train()
print(f"模型训练完成，训练集准确率: {train_acc:.4f}，测试集准确率: {test_acc:.4f}")
# 提取规则
dt_rules = tree_miner.extract_rules()
print(f"从决策树中提取到 {len(dt_rules)} 条规则")
# 规则清单及全面评估
tree_miner.evaluate_rules()
```

## 三、规则智能评估模块介绍

对于规则系统，我们很难监控规则的是否还有成效（因为被规则拒掉的客户是不会有后续表现，不知道拦截得准不准）。一套规则拒绝率可能有70%，通常除了监控拒绝率的稳定性，我们不知道规则拦截效果怎么样。传统的解决方案是分流一部分客户不被规则拦截，然后再反过来看下本应该这个规则拦截客户是好是坏。但这种办法比较费钱，本该拒绝的用户放进来容易增加违约风险。

rulelift旨在解决上述挑战，为信用风险管理团队提供一个高效、全面的规则效度分析工具：

- **实时高效监控**：无需分流测试，基于规则记录数据的实时规则效度分析
- **全面评估**：综合考虑命中率、逾期率、召回率、精确率、lift值、F1分数等多个指标
- **系统性视角**：评估规则之间的相关性和整体效果，减少人工干预，提高分析效率和准确性

### 技术原理
基本思路是，基于分析规则拦截用户评级分布与整体客户评级分布的差异，估算出规则的效度。当然也支持常规的拿客户现有的逾期情况去评估规则有效性，这种一般在规则开发空跑时期使用，这里我们就不赘述。

在信用风控中，我们常用信用评分卡评级或者风险评级去衡量不同客户的风险。对于一个有效规则来说，它所拦截的客户的风险也应该比全量客户更高些，在评级分布上会存在明显差异。我们就可以根据评级对应的坏账率去推估规则拦截这些用户的好坏程度。如下示例：
#### 原全量客户评级分布（规则拦截前）
| 评级 | 客户占比 | 对应坏账率 |
|------|----------|------------|
| 1    | 10%      | 20.00%     |
| 2    | 20%      | 15.00%     |
| 3    | 30%      | 10.00%     |
| 4    | 25%      | 5.00%      |
| 5    | 15%      | 2.00%      |

#### 规则拦截客户评级分布
| 评级 | 客户占比 | 对应坏账率 |
|------|----------|------------|
| 1    |**35%**    | 20.00%     |
| 2    | 30%      | 15.00%     |
| 3    | 20%      | 10.00%     |
| 4    | 10%      | 5.00%      |
| 5    | **5%**       | 2.00%      |

从上述数据可以看出，有效规则拦截的客户中，高风险评级（1-2级）的占比明显高于全量客户，而低风险评级（3-5级）的占比明显低于全量客户。这种分布差异是评估规则有效性的重要依据，以此我们可以计算出规则的lift值、精确率、召回率、F1分数等指标。

我们以详细代码示例说明下用法，

### 规则监控数据集
首先，我们需要准备一个规则命中记录流水数据去分析评估规则效果，包含规则名称、用户编号、命中日期（可选）、用户评级（可选）、用户评级对应的坏账率（可选）、用户实际逾期（可选，如果有用户实际逾期表现，就可以不用借由用户评级去推估规则效果了）。

#### 示例1：规则命中记录数据（关联用户评级）

| RULE | USER_ID | HIT_DATE | USER_LEVEL | USER_LEVEL_BADRATE |
|------|---------|----------|------------|--------------------|
| 阿里欺诈分>=95 | ID20261120004467 | 2026/10/1 | 1 | 0.20 |
| 百度欺诈分>=90 | ID20261120004467 | 2026/10/1 | 1 | 0.20 |
| 百度欺诈分>=90 | ID20261119001974 | 2026/10/1 | 2 | 0.15 |
| 授信通过 | ID20261116003965 | 2026/10/1 | 3 | 0.10 |

**用户评级及对应坏账率定义**：用户评级是指对客户风险进行分类的指标，可以使用任意具有风险排序性的字段，如评分卡评级、分层或风险评级，甚至学历收入字段。对于每种评级，都有对应的当前客群历史坏账率。这个历史评级的坏账率会影响规则评估的精确率等指标，但如果你只在于评估规则的lift值这种相对的数值，那么这个历史坏账率大小就不是那么重要了。以下是一个示例评级对应关系：

| 评级 | 评级名称 | 对应坏账率 | 风险描述 |
|------|----------|------------|----------|
| 1    | 高风险   | 20.00%     | 高逾期风险客户 |
| 2    | 中高风险 | 15.00%     | 较高逾期风险客户 |
| 3    | 中风险   | 10.00%     | 中等逾期风险客户 |
| 4    | 中低风险 | 5.00%      | 较低逾期风险客户 |
| 5    | 低风险   | 2.00%      | 低逾期风险客户 |

#### 示例2：规则记录数据（关联实际逾期）

| RULE | USER_ID | HIT_DATE | USER_TARGET |
|------|---------|----------|-------------|
| 阿里欺诈分>=95 | ID20261120002631 | 2026/10/1 | 1 |
| 百度欺诈分>=90 | ID20261116003919 | 2026/10/1 | 0 |
| 授信通过 | ID20261115001234 | 2026/10/1 | 1 |

### 规则评估功能完整示例

#### 示例1：使用预估指标评估规则（基于用户评级）

当没有实际逾期数据时，可以使用用户评级和对应坏账率来评估规则效果：

```python
from rulelift import analyze_rules, load_example_data

# 加载规则命中数据
df = load_example_data('hit_rule_info.csv')

# 分析规则效度（仅使用用户评级数据）
result = analyze_rules(df, user_level_badrate_col='USER_LEVEL_BADRATE')

# 查看分析结果
print(result[['rule', 'estimated_badrate_pred', 'estimated_lift_pred']].head())
```

**运行结果**：
```
        rule  estimated_badrate_pred  estimated_lift_pred
0  阿里欺诈分>=95                0.191988              1.185612
1  百度欺诈分>=90                0.193519              1.195128
2       授信通过                0.175126              1.080866
```

#### 示例2：使用实际指标评估规则（基于实际逾期数据）

当有实际逾期数据时，可以直接使用实际指标评估规则效果：

```python
# 分析规则效度（仅使用实际逾期数据）
result = analyze_rules(df, user_target_col='USER_TARGET')

# 查看分析结果，按lift值排序
result_sorted = result.sort_values(by='actual_lift', ascending=False)
print(result_sorted[['rule', 'actual_badrate', 'actual_lift', 'f1']].head())
```

**运行结果**：
```
        rule  actual_badrate  actual_lift        f1
0  阿里欺诈分>=95        0.150000     1.250000  0.260870
1  百度欺诈分>=90        0.120000     1.000000  0.210526
2       授信通过        0.080000     0.666667  0.145455
```

#### 示例3：结合命中率稳定性监控

监控规则的命中率变化，评估规则的稳定性：

```python
# 分析规则效度（包含命中率计算）
result_with_hitrate = analyze_rules(df, 
                                  user_target_col='USER_TARGET',
                                  hit_date_col='HIT_DATE')

# 查看命中率相关指标
hitrate_cols = ['rule', 'base_hit_rate', 'current_hit_rate', 'hit_rate_cv']
print(result_with_hitrate[hitrate_cols].head())
```

**运行结果**：
```
        rule  base_hit_rate  current_hit_rate  hit_rate_cv
0  阿里欺诈分>=95       0.331609         0.323529     0.420712
1  百度欺诈分>=90       0.333333         0.347826     0.256944
2       授信通过       0.334953         0.328638     0.086924
```

#### 示例4：规则相关性分析

分析规则之间的相关性，发现冗余和冲突规则：

```python
from rulelift import analyze_rule_correlation

# 计算规则相关性矩阵
correlation_matrix, max_correlation = analyze_rule_correlation(df)

# 查看相关性矩阵
print("规则相关性矩阵：")
print(correlation_matrix)

# 查看每条规则的最大相关性
print("\n每条规则的最大相关性：")
for rule, corr in max_correlation.items():
    print(f"  {rule}: {corr['max_correlation_value']:.4f}")
```

**运行结果**：
```
规则相关性矩阵：
RULE           授信通过  百度欺诈分>=90  阿里欺诈分>=95
RULE
授信通过       1.000000  -0.398847  -0.887592
百度欺诈分>=90 -0.398847   1.000000  -0.053852
阿里欺诈分>=95 -0.887592  -0.053852   1.000000

每条规则的最大相关性：
  授信通过: -0.3988
  百度欺诈分>=90: -0.0539
  阿里欺诈分>=95: -0.0539
```

#### 示例5：规则增益分析

**规则增益定义**：规则增益是指在现有策略基础上添加新规则后，策略整体效果的提升程度。它反映了新规则对策略的贡献价值，帮助我们识别高价值规则。

```python
from rulelift import calculate_strategy_gain

# 实际存在的规则策略
actual_rules = df['RULE'].unique()
print("\n场景2: 单个规则作为策略")
single_rule_strategies = {
    rule: [rule] for rule in actual_rules[:3]  # 使用前3个规则作为单个策略
}

gain_matrix, gain_details = calculate_strategy_gain(
    df, 
    rule_col='RULE', 
    user_id_col='USER_ID',
    user_target_col='USER_TARGET',
    strategy_definitions=single_rule_strategies
)
print(f"单个规则策略的增益矩阵:")
print(gain_matrix)
```

**运行结果**：
```
场景2: 单个规则作为策略
单个规则策略的增益矩阵:
                授信通过  阿里欺诈分>=95  百度欺诈分>=90
授信通过       0.000000   0.796356   0.859801
阿里欺诈分>=95  1.255720   0.000000   1.076994
百度欺诈分>=90  1.163059   0.910642   0.000000
```
### 本方法缺陷与优化

本方法有几点缺陷，
首先，对用户评级区分度高度依赖，假如一个AUC为0.9的评分模型那他对于规则下客户的评估就比较可靠，反之亦然。
其次，评级是根据历史通过的数据训练来的，这样或多或少可能存在幸存者偏差的问题，导致预测样本不准。
最后，算法倾向于赋予与评级相关性较高的规则更高的效度值，如果评级很可靠这倒没啥问题。

**改进方法**：推估方法强依赖评级效果，因此评级效果越好，良好拟合实际标签，推估效果就越好。可以通过拒绝推断、大样本减少幸存者偏差，再者可以通过算法优化、加入多方面维度的特征提高评级的拟合及泛化性。

## 四、规则自动挖掘模块介绍

规则自动挖掘模块是 rulelift 的另一个核心功能，旨在解决手动制定规则耗时耗力、难以发现复杂关系的问题。该模块基于用户特征自动挖掘及优化规则，帮助风控团队快速生成高质量的规则集，提升规则系统的整体效果。规则自动挖掘模块提供以下核心功能：
- 特征分析：对所有特征进行全面分析，包括效度指标、相关性分析和分箱分析
- 单特征规则挖掘：支持传统方法、XGBoost方法、卡方检验方法，快速发现单个特征的有效阈值
- 多特征交叉规则挖掘：支持多特征两两交叉，发现特征间的复杂交互关系
- 树模型规则提取：支持决策树、随机森林、卡方决策树、XGBoost、孤立森林5种算法
- 全面评估：支持使用badrate、损失率、lift、recall指标评估规则和特征的有效性
- 可视化支持：提供丰富的可视化功能，包括特征重要性图、决策树结构图、规则评估图、交叉热力图等

### 内置数据集示例介绍

规则挖掘模块提供了内置的 `feas_target.csv` 数据集，用于演示功能：

| 字段名 | 描述 | 类型 | 示例值 |
|--------|------|------|--------|
| ID | 用户唯一标识 | 字符串 | ID20260510020747 |
| CREATE_TIME | 数据创建时间 | 日期 | 25-Apr |
| ALI_FQZSCORE | 阿里欺诈分数 | 数值 | 700 |
| BAIDU_FQZSCORE | 百度欺诈分数 | 数值 | 458 |
| 人行近3个月申请借款次数 | 用户近3个月借款申请次数 | 数值 | 35 |
| ISBAD | 目标变量（坏客户标记） | 0/1 | 1 |


### 0. 特征分析（变量分析）示例
- **技术机制**：对所有特征进行全面分析，包括效度指标、相关性分析和分箱分析
- **核心指标**：IV值、KS值、相关性系数、变量分箱、损失率分布、PSI等
- **可视化支持**：提供变量相关性热力图和分箱分析图
- **适用场景**：规则挖掘前的特征筛选和理解，帮助选择最有效的特征

```python
from rulelift import VariableAnalyzer, load_example_data
import matplotlib.pyplot as plt

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')
print(f"用户特征数据集形状: {feature_df.shape}")
print(f"数据列名: {list(feature_df.columns)}")

# 初始化变量分析器（不使用损失率指标）
print("\n1. 初始化变量分析器（不使用损失率指标）:")
variable_analyzer = VariableAnalyzer(feature_df, exclude_cols=['ID', 'CREATE_TIME','OVD_BAL','AMOUNT'], target_col='ISBAD')
print(f"   分析特征数量: {len(variable_analyzer.features)}")

# 分析所有变量的效度指标
print("\n2. 所有变量效度指标分析:")
var_metrics = variable_analyzer.analyze_all_variables()
print(f"   变量分析结果:")
print(var_metrics)

# 初始化变量分析器（使用损失率指标）
print("\n3. 初始化变量分析器（使用损失率指标）:")
variable_analyzer_with_loss = VariableAnalyzer(
    feature_df, 
    exclude_cols=['ID', 'CREATE_TIME','OVD_BAL','AMOUNT'], 
    target_col='ISBAD',
    amount_col='AMOUNT',
    ovd_bal_col='OVD_BAL'
)
print(f"   分析特征数量: {len(variable_analyzer_with_loss.features)}")

# 变量相关性分析
print("\n4. 变量相关性分析:")
corr_matrix = feature_df[var_metrics['variable']].corr()
print(f"   变量相关性矩阵:")
print(corr_matrix)

# 可视化相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('变量相关性矩阵')
plt.savefig('images/variable_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   变量相关性矩阵图已保存到: images/variable_correlation.png")

# 分析单个变量的分箱情况
print("\n5. 单个变量分箱分析:")
feature = 'ALI_FQZSCORE'  # 选择一个特征
bin_analysis = variable_analyzer.analyze_single_variable(feature, n_bins=10)
print(f"   {feature}特征的分箱分析结果:")
print(bin_analysis)

# 分析单个变量的分箱情况（包含损失率指标）
print("\n6. 单个变量分箱分析（包含损失率指标）:")
bin_analysis_with_loss = variable_analyzer_with_loss.analyze_single_variable(feature, n_bins=10)
print(f"   {feature}特征的分箱分析结果（包含损失率）:")
print(bin_analysis_with_loss.head())

# 可视化变量分箱结果
variable_analyzer.plot_variable_bins(feature, n_bins=10)
plt.savefig('images/variable_bin_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   变量分箱分析图已保存到: images/variable_bin_analysis.png")

# PSI计算示例
print("\n7. PSI计算示例:")
psi_value = variable_analyzer.calculate_psi(feature)
print(f"   {feature}特征的PSI值: {psi_value:.4f}")
```

**运行结果**：
```
用户特征数据集形状: (499, 6)
数据列名: ['ID', 'CREATE_TIME', 'ALI_FQZSCORE', 'BAIDU_FQZSCORE', '人行近3个月申请借款次数', 'ISBAD']

1. 初始化变量分析器（不使用损失率指标）:
   分析特征数量: 3

2. 所有变量效度指标分析:
   变量分析结果:
       variable    iv_value       ks  mean_diff  corr_with_target
0  ALI_FQZSCORE    0.884846  0.652000   0.094000         -0.596000
1  BAIDU_FQZSCORE    0.386249  0.384000   0.040000         -0.356000
2  人行近3个月申请借款次数    0.737160  0.582000   0.084000          0.554000

3. 初始化变量分析器（使用损失率指标）:
   分析特征数量: 3

4. 变量相关性分析:
   变量相关性矩阵:
                   ALI_FQZSCORE  BAIDU_FQZSCORE  人行近3个月申请借款次数
ALI_FQZSCORE           1.000000        0.356000          -0.446000
BAIDU_FQZSCORE         0.356000        1.000000          -0.252000
人行近3个月申请借款次数      -0.446000       -0.252000           1.000000
   变量相关性矩阵图已保存到: images/variable_correlation.png


6. 单个变量分箱分析（包含损失率指标）:
   ALI_FQZSCORE特征的分箱分析结果（包含损失率）:
   bin_id        lower_bound       upper_bound  sample_count  bad_count  \
0       0  515.0000000000000  535.0000000000000             1          1   
1       1  535.0000000000000  555.0000000000000             0          0   
2       2  555.0000000000000  575.0000000000000             0          0   
3       3  575.0000000000000  595.0000000000000             2          2   
4       4  595.0000000000000  615.0000000000000             3          3   
5       5  615.0000000000000  635.0000000000000             6          6   
6       6  635.0000000000000  655.0000000000000            10          9   
7       7  655.0000000000000  675.0000000000000            15         12   
8       8  675.0000000000000  695.0000000000000            20         15   
9       9  695.0000000000000  715.0000000000000            24         18   

    badrate      lift  coverage  loss_rate  loss_lift
0  1.000000  3.261438  0.002004   0.150000   3.261438
1  0.000000  0.000000  0.000000   0.000000   0.000000
2  0.000000  0.000000  0.000000   0.000000   0.000000
3  1.000000  3.261438  0.004008   0.100000   2.174292
4  1.000000  3.261438  0.006012   0.090000   1.961872
5  1.000000  3.261438  0.012024   0.080000   1.740988
6  0.900000  2.935294  0.020040   0.070000   1.523810
7  0.800000  2.609150  0.030060   0.060000   1.309150
8  0.750000  2.446078  0.040080   0.050000   1.089744
9  0.750000  2.446078  0.048096   0.040000   0.871622

   变量分箱分析图已保存到: images/variable_bin_analysis.png

```




### 示例1：单特征规则挖掘

单特征规则挖掘适用于快速发现单个特征的有效阈值，支持传统方法、XGBoost方法和卡方检验方法。

##### 1.1 支持的方法

| 方法 | 代码 | 特点 | 适用场景 |
|------|------|------|----------|
| 传统方法 | `None` | 基于分箱的简单统计 | 快速发现特征阈值 |
| XGBoost方法 | `xgb` | 基于XGBoost算法的规则提取 | 快速挖掘的规则 |
| 卡方检验方法 | `chi2` | 基于卡方检验的规则提取 | 特征选择更严格 |

##### 1.2 核心功能

- **多种方法支持**：支持传统方法、XGBoost方法、卡方检验方法
- **损失率指标**：支持使用损失率指标评估规则的有效性
- **多种评估指标**：支持lift、badrate、precision、recall、f1、loss_rate、loss_lift等多种指标
- **表格形式展示**：规则评估结果以表格形式展示，便于分析和筛选

##### 1.3 传统方法示例

```python
from rulelift import SingleFeatureRuleMiner, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')
print(f"用户特征数据集形状: {feature_df.shape}")
print(f"数据列名: {list(feature_df.columns)}")

# 初始化单特征规则挖掘器（不使用损失率指标）
miner = SingleFeatureRuleMiner(feature_df, target_col='ISBAD', exclude_cols=['ID', 'CREATE_TIME', 'OVD_BAL', 'AMOUNT'])

# 选择一个特征进行分析
feature = 'ALI_FQZSCORE'
print(f"\n分析特征: {feature}")

# 计算单个特征的指标
metrics_df = miner.calculate_single_feature_metrics(feature, num_bins=20)

# 获取规则展示
top_rules = miner.get_top_rules(feature, top_n=5, metric='lift')
print(f"{feature}特征的top 5规则:")
print(top_rules[['rule_description', 'lift', 'badrate', 'sample_ratio']])
```

**运行结果**：
```
用户特征数据集形状: (499, 6)
数据列名: ['ID', 'CREATE_TIME', 'ALI_FQZSCORE', 'BAIDU_FQZSCORE', '人行近3个月申请借款次数', 'ISBAD']

分析特征: ALI_FQZSCORE
ALI_FQZSCORE特征的规则:
           rule_description      lift   badrate  sample_ratio
1  ALI_FQZSCORE <= 515.0000  3.261438  1.000000      0.002004
3  ALI_FQZSCORE <= 635.0000  2.213119  0.678571      0.056112
5  ALI_FQZSCORE <= 665.0000  2.174292  0.666667      0.102204
7  ALI_FQZSCORE <= 688.5000  2.087320  0.640000      0.150301
9  ALI_FQZSCORE <= 705.0000  1.993101  0.611111      0.216433
```
### 示例2：多特征交叉规则挖掘

多特征交叉规则挖掘适用于发现特征间的交互作用，支持多特征两两交叉、多种指标分析和Excel导出。

##### 2.1 核心功能

- **多特征两两交叉**：支持传入多个特征，自动生成所有两两特征组合的交叉矩阵
- **多种指标分析**：支持badrate、count、sample_ratio、lift、loss_rate、loss_lift等多种指标
- **Excel导出**：支持将交叉矩阵导出为Excel文件，方便策略人员分析
- **可视化支持**：通过热力图直观展示特征交叉关系

##### 2.2 两特征交叉示例

```python
from rulelift import MultiFeatureRuleMiner, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')

# 初始化多特征规则挖掘器
multi_miner = MultiFeatureRuleMiner(feature_df, target_col='ISBAD')

# 绘制交叉热力图
plt = multi_miner.plot_cross_heatmap(feature1, feature2, metric='lift')
plt.savefig('cross_feature_heatmap.png', dpi=300, bbox_inches='tight')
print("交叉特征热力图已保存到: cross_feature_heatmap.png")
```


##### 2.3 多特征两两交叉示例

支持传入多个特征，自动生成所有两两特征组合的交叉矩阵，方便策略人员全面分析特征间的交互关系。

```python
# 多特征两两交叉分析示例
features_list = ['ALI_FQZSCORE', 'BAIDU_FQZSCORE', 'NUMBER OF LOAN APPLICATIONS TO PBOC']
cross_matrices_multi = multi_miner.generate_cross_matrices_excel(
    features_list=features_list,
    output_path='cross_analysis_multi_features.xlsx',
    metrics=['badrate', 'count', 'sample_ratio', 'lift'],  # 支持多种指标分析
    binning_method='quantile'  # 支持等频分箱
)
print(f"多特征交叉矩阵Excel文件已保存到: cross_analysis_multi_features.xlsx")
```

##### 2.4 支持的指标说明

多特征交叉分析支持多种指标，帮助策略人员全面评估特征组合的风险水平和业务价值：

| 指标 | 定义 | 业务意义 |
|------|------|----------|
| `badrate` | 坏样本比例 = 坏样本数 / 总样本数 | 直接反映该特征组合下的风险水平 |
| `count` | 样本数量 | 反映该特征组合的覆盖范围 |
| `sample_ratio` | 样本占比 = 该组合样本数 / 总样本数 | 反映该特征组合的业务重要性 |
| `lift` | 提升度 = 该组合badrate / 总样本badrate | 反映该特征组合的风险区分能力，值越大效果越好 |
| `loss_rate` | 损失率 = 损失金额 / 总金额 | 反映该特征组合的实际损失程度（需要提供amount_col和ovd_bal_col） |
| `loss_lift` | 损失提升度 = 该组合loss_rate / 总样本loss_rate | 反映该特征组合的损失区分能力（需要提供amount_col和ovd_bal_col） |

##### 2.5 Excel文件内容示例

生成的Excel文件包含多个sheet，每个sheet对应一个特征组合，例如：
- `ALI_FQZSCORE_x_BAIDU_FQZSCORE`：两个特征的交叉矩阵
- `ALI_FQZSCORE_x_NUMBER OF LOAN APPLICATIONS TO PBOC`：另一个特征组合
- 每个sheet包含以下指标：badrate、count、sample_ratio、lift等
- 策略人员可以根据交叉矩阵中的高lift区域制订规则

**Excel文件内容示例**：（需要在初始化时提供amount_col和ovd_bal_col）

| ALI_FQZSCORE | BAIDU_FQZSCORE | badrate | count | loss_rate | loss_lift |
|--------------|----------------|---------|-------|-----------|-----------|
| (500, 600]   | (300, 400]     | 0.6667  | 15    | 0.4567    | 2.89      |
| (500, 600]   | (400, 500]     | 0.4000  | 25    | 0.3210    | 2.01      |
| (600, 700]   | (300, 400]     | 0.5000  | 20    | 0.2890    | 1.81      |
| (600, 700]   | (400, 500]     | 0.2000  | 30    | 0.1567    | 0.98      |
### 示例3：基于树模型的规则提取

TreeRuleExtractor 提供了统一的树模型规则提取接口，支持多种算法和丰富的配置选项，适用于生成综合性的规则集。

##### 3.1 支持的算法

TreeRuleExtractor 支持以下5种算法：

| 算法 | 代码 | 特点 | 适用场景 |
|------|------|------|----------|
| 决策树 | `dt` | 简单直观，易于解释 | 快速生成规则，适合初步探索 |
| 随机森林 | `rf` | 多树集成，稳定性好 | 需要更稳定、多样性的规则效果 |
| 卡方决策树 | `chi2` | 基于卡方检验分裂 | 特征选择更严格 |
| XGBoost | `xgb` | 梯度提升，性能优秀 | 需要高精度的规则 |
| 孤立森林 | `isf` | 异常检测，无监督 | 发现异常模式 |

##### 3.2 核心功能

- **多种算法支持**：支持决策树、随机森林、卡方决策树、XGBoost、孤立森林5种算法
- **超参数控制**：支持传入超参数控制树、规则复杂度，如max_depth、min_samples_split、min_samples_leaf、n_estimators等
- **业务解释性配置**：支持传入业务字段（feature_trends）挖掘符合业务解释性规则，过滤不符合业务逻辑的规则
- **多种指标评估**：支持多种指标（如lift、badrate、precision、recall、f1、loss_rate、loss_lift）评估挖掘规则的有效性
- **表格形式展示**：规则评估结果以表格形式展示，便于分析和筛选
- **可视化支持**：提供特征重要性图、决策树结构图、规则评估图等可视化功能

##### 3.3 基本使用示例

```python
from rulelift import TreeRuleExtractor, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')

# 初始化TreeRuleExtractor（决策树算法，不使用损失率指标）
tree_miner = TreeRuleExtractor(
    feature_df, 
    target_col='ISBAD', 
    exclude_cols=['ID', 'CREATE_TIME', 'OVD_BAL', 'AMOUNT'],
    algorithm='dt',  # 使用决策树算法
    max_depth=3,  # 树复杂度配置：决策树最大深度
    min_samples_split=20,  # 规则精度配置：分裂节点所需的最小样本数
    min_samples_leaf=10,  # 规则精度配置：叶子节点的最小样本数
    test_size=0.3,  # 测试集比例
    random_state=42  # 随机种子
)

# 训练模型
train_acc, test_acc = tree_miner.train()
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 提取规则
dt_rules = tree_miner.extract_rules()
print(f"提取的规则数量: {len(dt_rules)}")

# 规则评估
eval_results = tree_miner.evaluate_rules()
print(f"规则评估结果（前5条）:")
print(eval_results[['rule', 'test_hit_count', 'test_bad_count', 'test_badrate', 'test_hit_rate', 'test_baseline_badrate', 'test_badrate_after_interception', 'test_badrate_reduction', 'badrate_diff', 'test_precision', 'test_recall', 'test_f1', 'test_lift']].head())
```

**运行结果**：
```
训练集准确率: 0.8223
测试集准确率: 0.7600
提取的规则数量: 7
规则评估结果（前5条）:
                                                rule  test_hit_count  test_bad_count  test_badrate  test_precision  test_recall   test_f1  test_lift       
5  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...              12              11      0.916667        0.916667     0.244444  0.385965   3.055556       
1  NUMBER OF LOAN APPLICATIONS TO PBOC <= 9.5000 ...               9               6      0.666667        0.666667     0.133333  0.222222   2.222222       
4  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...              11               6      0.545455        0.545455     0.133333  0.214286   1.818182       
2  NUMBER OF LOAN APPLICATIONS TO PBOC <= 9.5000 ...              62              15      0.241935        0.241935     0.333333  0.280374   0.806452       
6  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...               9               2      0.222222        0.222222     0.044444  0.074074   0.740741       
```

##### 3.4 使用损失率指标评估

支持使用损失率指标评估规则的有效性，需要提供amount_col（金额字段）和ovd_bal_col（逾期金额字段）。

```python
# 初始化TreeRuleExtractor（使用损失率指标）
tree_miner_with_loss = TreeRuleExtractor(
    feature_df, 
    target_col='ISBAD', 
    exclude_cols=['ID', 'CREATE_TIME', 'OVD_BAL', 'AMOUNT'],
    algorithm='dt',
    max_depth=3, 
    min_samples_split=20,
    min_samples_leaf=10,
    test_size=0.3,
    random_state=42,
    amount_col='AMOUNT',  # 金额字段
    ovd_bal_col='OVD_BAL'  # 逾期金额字段
)

# 训练模型
train_acc, test_acc = tree_miner_with_loss.train()
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 提取规则
dt_rules = tree_miner_with_loss.extract_rules()
print(f"提取的规则数量: {len(dt_rules)}")

# 规则评估（使用损失率指标）
eval_results_with_loss = tree_miner_with_loss.evaluate_rules()
print(f"规则评估结果（前5条，包含损失率指标）:")
print(eval_results_with_loss[['rule', 'train_loss_rate', 'train_loss_lift', 'test_loss_rate', 'test_loss_lift', 'train_lift', 'test_lift']].head())
```

**运行结果**：
```
                                                rule  train_loss_rate  train_loss_lift  test_loss_rate  test_loss_lift  train_lift  test_lift
5  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...         0.047619         2.632478        0.047619        2.632478    3.055556   3.055556
1  NUMBER OF LOAN APPLICATIONS TO PBOC <= 9.5000 ...         0.025000         1.382979        0.025000        1.382979    2.222222   2.222222
4  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...         0.020000         1.106383        0.020000        1.106383    1.818182   1.818182
2  NUMBER OF LOAN APPLICATIONS TO PBOC <= 9.5000 ...         0.015000         0.829787        0.015000        0.829787    0.806452   0.806452
6  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...         0.010000         0.553192        0.010000        0.553192    0.740741   0.740741
```

##### 3.5 使用业务解释性配置挖掘规则

支持传入业务字段（feature_trends）挖掘符合业务解释性规则，过滤不符合业务逻辑的规则。

```python
# 初始化TreeRuleExtractor（XGBoost算法，使用特征趋势过滤）
tree_miner_xgb = TreeRuleExtractor(
    feature_df, 
    target_col='ISBAD', 
    exclude_cols=['ID', 'CREATE_TIME', 'OVD_BAL', 'AMOUNT'],
    algorithm='xgb',  # 使用XGBoost算法
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=10,
    n_estimators=10,
    max_features='sqrt',
    test_size=0.3,
    feature_trends={  # 业务解释性配置：特征与目标标签的正负相关性
        'ALI_FQZSCORE': -1,      # 负相关：分数越低，违约概率越高
        'BAIDU_FQZSCORE': -1,    # 负相关：分数越低，违约概率越高
        'NUMBER OF LOAN APPLICATIONS TO PBOC': 1  # 正相关：申请次数越多，违约概率越高
    },
    random_state=42
)

# 训练模型
train_acc, test_acc = tree_miner_xgb.train()
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 提取规则
xgb_rules = tree_miner_xgb.extract_rules()
print(f"提取的规则数量: {len(xgb_rules)}")

# 规则评估
eval_results = tree_miner_xgb.evaluate_rules()
print(f"规则评估结果（前5条）:")
print(eval_results[['rule', 'test_hit_count', 'test_bad_count', 'test_badrate', 'test_hit_rate', 'test_baseline_badrate', 'test_badrate_after_interception', 'test_badrate_reduction', 'badrate_diff', 'test_precision', 'test_recall', 'test_f1', 'test_lift']].head())
```

**运行结果**：
```
训练集准确率: 0.8166
测试集准确率: 0.7867
提取的规则数量: 31
规则评估结果（前5条）:
                                                 rule  test_hit_count  test_bad_count  test_badrate  test_precision  test_recall   test_f1  test_lift      
28  ALI_FQZSCORE <= 692.5000 AND ALI_FQZSCORE <= 6...              10              10      1.000000        1.000000     0.222222  0.363636   3.333333      
27  NUMBER OF LOAN APPLICATIONS TO PBOC > 9.5000 A...               7               7      1.000000        1.000000     0.155556  0.269231   3.333333      
30  BAIDU_FQZSCORE <= 390.5000 AND BAIDU_FQZSCORE ...               4               4      1.000000        1.000000     0.088889  0.163265   3.333333      
12  NUMBER OF LOAN APPLICATIONS TO PBOC > 10.5000 ...              17              16      0.941176        0.941176     0.355556  0.516129   3.137255      
14  ALI_FQZSCORE <= 692.5000 AND NUMBER OF LOAN AP...              17              16      0.941176        0.941176     0.355556  0.516129   3.137255      
```

##### 3.6 常用字段说明

超参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|----------|------|
| `algorithm` | str | `'dt'` | 算法类型：'dt'、'rf'、'chi2'、'xgb'、'isf' |
| `max_depth` | int | `5` | 决策树最大深度，控制树的复杂度 |
| `min_samples_split` | int | `10` | 分裂节点所需的最小样本数，控制规则精度 |
| `min_samples_leaf` | int | `5` | 叶子节点的最小样本数，控制规则精度 |
| `n_estimators` | int | `10` | 随机森林/XGBoost/孤立森林中树的数量 |
| `max_features` | str | `'sqrt'` | 每棵树分裂时考虑的最大特征数：'sqrt'或'log2' |
| `test_size` | float | `0.3` | 测试集比例 |
| `random_state` | int | `42` | 随机种子，保证结果可复现 |
| `feature_trends` | Dict[str, int] | `None` | 特征与目标标签的正负相关性字典，用于过滤不符合业务解释性的规则 |
| `amount_col` | str | `None` | 金额字段名，用于计算损失率指标 |
| `ovd_bal_col` | str | `None` | 逾期金额字段名，用于计算损失率指标 |

评估指标说明（训练集同理）
| 指标名称 | 类型 | 公式 | 意义 |
|---------|------|------|------|
| test_hit_count | int | - | 测试集上规则命中的样本数 |
| test_bad_count | int | - | 测试集上规则命中的坏样本数 |
| test_good_count | int | - | 测试集上规则命中的好样本数 |
| test_badrate | float | 测试集命中坏样本数 / 测试集命中样本数 | 规则在测试集上命中样本的坏样本率 |
| test_precision | float | 测试集命中坏样本数 / 测试集命中样本数 | 规则在测试集上的精确率 |
| test_recall | float | 测试集命中坏样本数 / 测试集总坏样本数 | 规则在测试集上的召回率 |
| test_f1 | float | 2 * (精确率 * 召回率) / (精确率 + 召回率) | 测试集上精确率和召回率的调和平均值 |
| test_lift | float | 规则坏样本率 / 测试集总坏样本率 | 规则在测试集上的提升度，核心评估指标 |
| test_hit_rate | float | 测试集命中样本数 / 测试集总样本数 | 规则在测试集上的命中率 |
| test_baseline_badrate | float | 测试集总坏样本数 / 测试集总样本数 | 测试集整体坏样本率（基准） |
| test_badrate_after_interception | float | (测试集总坏样本数 - 命中坏样本数) / (测试集总样本数 - 命中样本数) | 规则拦截后剩余样本的坏样本率 |
| test_badrate_reduction | float | [(基准坏样本率 - 拦截后坏样本率) / 基准坏样本率] / 命中率 | 规则对坏样本率的降低幅度 |

| badrate_diff | float | 训练集lift - 测试集lift | 训练集和测试集提升度的差异，衡量规则泛化能力 |

##### 3.7 可视化功能

TreeRuleExtractor 提供了丰富的可视化功能：

```python
# 特征重要性图
tree_miner.plot_feature_importance(save_path='images/tree_feature_importance.png')
print("特征重要性图已保存到: images/tree_feature_importance.png")

# 决策树结构图（仅支持决策树和卡方决策树）
tree_miner.plot_decision_tree(save_path='images/tree_decision_structure.pdf')
print("决策树结构图已保存到: images/tree_decision_structure.pdf")

# 规则评估图
tree_miner.plot_rule_evaluation(save_path='images/tree_rule_evaluation.png')
print("规则评估图已保存到: images/tree_rule_evaluation.png")
```


## 许可证

MIT License

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
