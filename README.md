## rulelift：策略规则有效性分析及自动挖掘规则的Python工具包。

## 一、规则系统的缺陷

在风控领域，规则系统因其配置便捷性和较强的解释性而被广泛应用，但也存在明显的缺陷：

1. **效果监控难**：被规则拒掉的客户没有后续表现数据，无法直接评估规则拦截效果
2. **稳定性差**：规则效果可能随时间漂移，需要定期监控和调整
3. **评估优化缺乏系统性**：手动调整规则耗时耗力，规则之间的相互影响难以评估，容易导致冗余或冲突，陷入局部最优

## 二、rulelift 解决方案

**rulelift** 提供了全面的解决方案，帮助风控团队克服上述挑战：

### 1. 规则智能评估模块
- **无需分流测试**：基于规则命中用户的评级分布即可评估规则效果
- **实时监控**：支持基于生产数据的实时规则效果分析
- **多维度评估**：综合考虑命中率、逾期率、召回率、精确率、lift值、F1分数等指标
- **规则相关性分析**：识别冗余规则，评估规则之间的相互影响
- **策略增益计算**：评估不同规则组合的效果提升

### 2. 规则自动挖掘模块
- **单特征规则挖掘**：自动从单个特征中挖掘有效的风控规则
- **多特征交叉规则挖掘**：发现特征之间的复杂交叉关系
- **决策树规则提取**：从决策树模型中提取可解释的规则
- **可视化支持**：直观展示规则效果和关系

基于对上线规则的评估结果，我们可以及时发现规则效率低下或不稳定的问题，从而及时调整规则阈值或删减。也可以结合规则挖掘，新增有效规则，提升规则系统的整体效果及稳定性。

### 快速开始

```bash
# 使用pip安装（推荐）
pip install rulelift

# 从源码安装
pip install git+https://github.com/aialgorithm/rulelift.git
```

### 基本使用示例

```python
# 加载示例数据
from rulelift import load_example_data, analyze_rules, DecisionTreeRuleExtractor

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
dt_miner = DecisionTreeRuleExtractor(
    feature_df, 
    target_col='ISBAD', 
    exclude_cols=['ID', 'CREATE_TIME'],
    max_depth=3, 
    min_samples_leaf=10
)
# 提取规则
dt_rules = dt_miner.extract_rules()
print(f"从决策树中提取到 {len(dt_rules)} 条规则")
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

#### 1. 单特征规则挖掘
- **技术机制**：对数值型特征进行等频或等宽分箱，计算每个分箱的风险指标
- **核心指标**：badrate（坏样本率）、lift值（风险提升倍数）、coverage（覆盖率）
- **筛选条件**：根据min_coverage和min_badrate等参数筛选有效规则
- **适用场景**：快速发现单个特征的有效阈值，适合初步探索阶段

#### 2. 多特征交叉规则挖掘
- **技术机制**：生成特征组合的交叉矩阵，计算每个交叉组合的风险指标
- **核心指标**：交叉组合的badrate、lift值、样本占比
- **可视化支持**：通过热力图直观展示特征交叉关系
- **适用场景**：发现特征间的交互作用，生成更复杂的规则

#### 3. 决策树规则提取
- **技术机制**：训练决策树模型，从树结构中提取可解释的规则
- **优势**：生成的规则具有良好的可解释性和较高的预测能力
- **优化手段**：通过剪枝等技术控制规则复杂度
- **适用场景**：生成综合性的规则集，适合构建完整的规则策略

### 数据要求

规则挖掘模块需要以下类型的数据：

| 数据类型 | 描述 | 示例 |
|---------|------|------|
| 特征数据 | 用户的各种属性和行为特征 | 信用评分、申请次数、收入水平等 |
| 标签数据 | 用户的实际表现标签 | 逾期/未逾期、欺诈/非欺诈等 |

### 内置数据集

规则挖掘模块提供了内置的 `feas_target.csv` 数据集，用于演示功能：

| 字段名 | 描述 | 类型 | 示例值 |
|--------|------|------|--------|
| ID | 用户唯一标识 | 字符串 | ID20260510020747 |
| CREATE_TIME | 数据创建时间 | 日期 | 25-Apr |
| ALI_FQZSCORE | 阿里欺诈分数 | 数值 | 700 |
| BAIDU_FQZSCORE | 百度欺诈分数 | 数值 | 458 |
| 人行近3个月申请借款次数 | 用户近3个月借款申请次数 | 数值 | 35 |
| ISBAD | 目标变量（坏客户标记） | 0/1 | 1 |

### 功能使用示例

#### 示例1：单特征规则挖掘

单特征规则挖掘适用于快速发现单个特征的有效阈值：

```python
from rulelift import SingleFeatureRuleMiner, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')
print(f"用户特征数据集形状: {feature_df.shape}")
print(f"数据列名: {list(feature_df.columns)}")

# 初始化单特征规则挖掘器
miner = SingleFeatureRuleMiner(feature_df, target_col='ISBAD', exclude_cols=['ID', 'CREATE_TIME'])

# 选择一个特征进行分析
feature = 'ALI_FQZSCORE'
print(f"\n分析特征: {feature}")

# 计算单个特征的指标
metrics_df = miner.calculate_single_feature_metrics(feature, num_bins=20)

# 获取top规则
top_rules = miner.get_top_rules(feature, top_n=5, metric='lift')
print(f"{feature}特征的top 5规则:")
print(top_rules[['rule_description', 'lift', 'badrate', 'sample_ratio']])
```

**运行结果**：
```
用户特征数据集形状: (499, 6)
数据列名: ['ID', 'CREATE_TIME', 'ALI_FQZSCORE', 'BAIDU_FQZSCORE', '人行近3个月申请借款次数', 'ISBAD']

分析特征: ALI_FQZSCORE
ALI_FQZSCORE特征的top 5规则:
           rule_description      lift   badrate  sample_ratio
1  ALI_FQZSCORE <= 515.0000  3.261438  1.000000      0.002004
3  ALI_FQZSCORE <= 635.0000  2.213119  0.678571      0.056112
5  ALI_FQZSCORE <= 665.0000  2.174292  0.666667      0.102204
7  ALI_FQZSCORE <= 688.5000  2.087320  0.640000      0.150301
9  ALI_FQZSCORE <= 705.0000  1.993101  0.611111      0.216433
```

#### 示例2：多特征交叉规则挖掘

多特征交叉规则挖掘适用于发现特征间的交互作用：

```python
from rulelift import MultiFeatureRuleMiner, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')

# 初始化多特征规则挖掘器
multi_miner = MultiFeatureRuleMiner(feature_df, target_col='ISBAD')
    
# 生成交叉规则
feature1 = 'ALI_FQZSCORE'
feature2 = 'BAIDU_FQZSCORE'
print(f"\n生成 {feature1} 和 {feature2} 的交叉规则")

# 获取交叉规则
cross_rules = multi_miner.get_cross_rules(feature1, feature2, top_n=5, metric='lift')
print(f"{feature1}和{feature2}的交叉规则top 5:")
print(cross_rules[['rule_description', 'lift', 'badrate', 'sample_ratio']])

# 绘制交叉热力图
plt = multi_miner.plot_cross_heatmap(feature1, feature2, metric='lift')
plt.savefig('cross_feature_heatmap.png', dpi=300, bbox_inches='tight')
print("交叉特征热力图已保存到: cross_feature_heatmap.png")
```

**运行结果**：
```
生成 ALI_FQZSCORE 和 BAIDU_FQZSCORE 的交叉规则
ALI_FQZSCORE和BAIDU_FQZSCORE的交叉规则top 5:
                                  rule_description      lift   badrate  sample_ratio
90  ALI_FQZSCORE = count AND BAIDU_FQZSCORE = 18.0  5.000000  5.000000           0.1
14    ALI_FQZSCORE = lift AND BAIDU_FQZSCORE = 2.0  3.261438  3.261438           0.1
80  ALI_FQZSCORE = count AND BAIDU_FQZSCORE = 16.0  3.000000  3.000000           0.1
20   ALI_FQZSCORE = count AND BAIDU_FQZSCORE = 4.0  2.000000  2.000000           0.1
30   ALI_FQZSCORE = count AND BAIDU_FQZSCORE = 6.0  2.000000  2.000000           0.1
交叉特征热力图已保存到: cross_feature_heatmap.png
```

#### 示例3：基于决策树的规则提取

决策树规则提取适用于生成综合性的规则集：

```python
from rulelift import DecisionTreeRuleExtractor, load_example_data

# 加载用户特征数据集
feature_df = load_example_data('feas_target.csv')

# 初始化决策树规则提取器
dt_miner = DecisionTreeRuleExtractor(
    feature_df, 
    target_col='ISBAD', 
    exclude_cols=['ID', 'CREATE_TIME'],
    max_depth=3, 
    min_samples_leaf=10
)

# 训练决策树并提取规则
dt_miner.extract_rules()

# 获取规则DataFrame
dt_rules_df = dt_miner.get_rules_as_dataframe()
print(f"决策树提取的规则数量: {len(dt_rules_df)}")
print(f"规则DataFrame列名: {list(dt_rules_df.columns)}")

# 打印规则
dt_miner.print_rules(top_n=3)

# 绘制特征重要性图
dt_miner.plot_feature_importance()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("特征重要性图已保存到: feature_importance.png")
```

**运行结果**：
```
决策树提取的规则数量: 7
规则DataFrame列名: ['rule_id', 'rule', 'predicted_class', 'class_name', 'class_probability', 'sample_count', 'importance', 'class_distribution']
=== Top 3 Rules ===

Rule 5 (Importance: 1.9111):
  人行近3个月申请借款次数 > 10.5000 AND ALI_FQZSCORE <= 807.5000 AND BAIDU_FQZSCORE <= 490.5000 
  Predicted Class: bad (Probability: 0.9556)
  Sample Count: 1
  Class Distribution: {'good': 0.044444444444444446, 'bad': 0.9555555555555556}
  拦截用户数: 11
  坏客户数: 9
  好客户数: 2
  Badrate: 0.8182
  召回率: 0.3462
  Lift: 3.1469
```


## 许可证

MIT License

## 作者

aialgorithm <15880982687@qq.com>

## 版本信息

- 当前版本：1.1.6
- 发布日期：2025-12-25

## 项目地址

- GitHub: https://github.com/aialgorithm/rulelift
- PyPI: https://pypi.org/project/rulelift/

## 后续维护
如有bug或维护建议，请通过GitHub Issues反馈，我们会尽快响应并解决。也可以提交Pull Request（PR）来贡献代码。
- 整合多个规则的评估结果，形成策略级结论
- 增强实际场景数据处理能力
- 结果展示&操作可视化
- 考虑敏感信息，暂无法支持AI大模型




<img width="273" height="259" alt="image" src="https://github.com/user-attachments/assets/925325e4-bfe7-4186-be93-fd01248508b2" />

