## rulelift 一个用于信用风险管理中策略规则有效性分析及监控的Python工具包。

## 背景信息与核心价值

### 规则系统监控的挑战
在风控领域，规则系统因为配置的便利性及较强的解释性，而受到广泛应用。但是规则系统有个弊端，难于监控规则的是否还有成效（因为被规则拒掉的客户是不会有后续表现，不知道客户是好是坏）。一套规则拒绝率可能有70%，通常除了监控拒绝率的稳定性，我们不知道规则拦截效果怎么样。特别规则到一定规模后就更难监控维护了，更别说优化规则的效果了。

传统的解决方案是分流一部分客户不被规则拦截，然后再反过来看下本应该这个规则拦截客户是好是坏。但这种办法比较费钱，本该拒绝的用户放进来容易增加违约风险。

随着业务的快速发展，规则数量不断增加，传统的监控方法在规则效度评估方面面临诸多局限性：

1. **滞后性**：传统方法通常基于历史数据进行事后分析，无法实时反映规则在生产环境中的实际表现
2. **单一维度**：往往只关注规则的拦截率或命中率，忽略了规则对风险区分的实际贡献
3. **缺乏系统性**：难以评估规则之间的相互影响和整体效果
4. **依赖人工经验**：需要大量人工分析和调整，效率低下且容易产生主观偏差
5. **缺乏量化指标**：难以客观衡量规则的真实价值和有效性

### rulelift的核心价值
为解决规则评估的诸多不便，我开源了rulelift库，安装简单：`pip install rulelift`。

rulelift旨在解决上述挑战，为信用风险管理团队提供一个高效、全面、量化的规则效度分析工具：

- **实时监控**：支持基于生产数据的实时规则效度分析
- **多维评估**：综合考虑命中率、逾期率、召回率、精确率、lift值等多个指标
- **系统性视角**：评估规则之间的相关性和整体效果
- **自动化分析**：减少人工干预，提高分析效率和准确性
- **灵活配置**：支持不同业务场景和数据源的适配
- **可解释性强**：提供直观的分析结果，便于业务人员理解和决策
- **成本效益高**：无需分流测试，基于现有数据即可评估规则效果

### 基本思路
rulelift的基本思路是，基于分析规则拦截用户评级分布与整体客户评级分布的差异，估算出规则的效度。当然也支持拿客户现有的逾期情况去评估规则有效性。

在信用风控中，我们常用信用评分卡评级或者风险评级去衡量不同客户的风险。对于一个有效规则来说，它所拦截的客户的风险也应该比全量客户更高些，在评级分布上会存在明显差异。

我们就可以根据评级预期的坏账率去推估这些用户有多坏。

n那我们就可以根据评级预期的坏账率去推估这些用户有多坏。

我们以数据示例说明下，

另外一种方式就是使用客户逾期情况，这种一般用于规则开发时期，拿着历史样本去评估规则效果，原理比较简单。

## 技术原理与缺陷分析

### 效度估算方法的技术原理
rulelift采用双层指标体系进行规则效度评估：

1. **基于用户评级的预估指标**
   - 利用用户评级坏账率（USER_LEVEL_BADRATE）计算规则的预估效果
   - 适用于缺乏实际逾期数据（USER_TARGET）的场景
   - 核心指标：预估命中率、预估逾期率、预估召回率、预估精确率、预估lift值

2. **基于实际逾期的实际指标**
   - 利用真实逾期数据计算规则的实际效果
   - 提供最准确的规则效度评估
   - 核心指标：实际命中率、实际逾期率、实际召回率、实际精确率、实际lift值、F1分数

3. **动态命中率监控**
   - 计算历史平均命中率（base_hit_rate）、当前命中率（current_hit_rate）
   - 引入命中率变异系数（hit_rate_cv）衡量命中率的稳定性
   - 支持通过HIT_DATE字段分析规则命中情况的时间趋势

4. **规则相关性分析**
   - 计算规则之间的相关系数矩阵
   - 识别高度相关的冗余规则
   - 评估规则之间的相互影响

### 主要技术缺陷
通过拒绝推断、泛化强的评分卡
#### 1. 对评级区分度的高度依赖性
- **问题**：当前效度估算方法严重依赖用户评级（USER_LEVEL）的质量和区分度
- **技术机制**：算法基于评级坏账率计算预估指标，评级区分度直接影响效度评估结果
- **影响**：如果评级体系本身区分度不足，会导致所有规则的效度评估结果失真
- **表现形式**：在评级区分度差的情况下，所有规则的预估lift值可能差异不大，无法有效区分规则质量

#### 2. 推断偏差
- **问题**：在缺乏真实Target变量（实际逾期）的情况下，算法存在系统性偏差
- **技术机制**：算法倾向于赋予与评级相关性较高的规则更高的效度值
- **影响**：规则效度评估结果可能偏离实际业务价值，产生"幸存者偏差"
- **表现形式**：与评级高度相关的规则可能获得过高的效度评分，而一些与评级相关性较低但实际有效的规则可能被低估。

#### 3. 其他技术局限性
- **数据依赖性**：需要高质量的历史数据和评级信息
- **计算复杂度**：随着规则数量增加，相关性计算的复杂度呈平方增长
- **实时性限制**：对于大规模数据，实时分析可能存在性能瓶颈

## 功能介绍

rulelift可以帮助您分析信用风险规则的有效性，包括：

- 基于用户评级坏账率（USER_LEVEL_BADRATE）的预估指标
- 基于实际逾期情况（USER_TARGET）的实际指标
- 动态命中率监控（历史命中率、当前命中率、命中率变异系数）
- 规则相关性分析，识别冗余规则
- 核心指标包括：命中率、逾期率、召回率、精确率、lift值、F1分数等
- 支持自定义字段映射
- 输出结构化的分析结果

## 安装方法

```bash
pip install rulelift
```

## 快速开始

### 1. 加载示例数据

```python
from rulelift import load_example_data

# 加载示例数据
df = load_example_data()

# 查看数据结构
df.head()
```

### 2. 分析规则效度（仅实际指标）

```python
from rulelift import analyze_rules

# 分析规则效度（仅使用实际逾期数据）
result = analyze_rules(df, user_target_col='USER_TARGET')

# 查看分析结果
print(result.head())

# 按lift值排序
result_sorted = result.sort_values(by='actual_lift', ascending=False)
print(result_sorted[['rule', 'f1', 'actual_lift']].head())
```

### 3. 分析规则效度（包含命中率监控）

```python
# 分析规则效度（包含命中率计算）
result_with_hitrate = analyze_rules(df, 
                                  user_target_col='USER_TARGET',
                                  hit_date_col='HIT_DATE')

# 查看命中率相关指标
hitrate_cols = ['rule', 'base_hit_rate', 'current_hit_rate', 'hit_rate_cv']
print(result_with_hitrate[hitrate_cols].head())
```

### 4. 分析规则相关性

```python
from rulelift import analyze_rule_correlation

# 计算规则相关性矩阵
correlation_matrix, max_correlation = analyze_rule_correlation(df)

# 查看相关性矩阵
print(correlation_matrix)

# 查看每条规则的最大相关性
print("\n每条规则的最大相关性：")
for rule, corr in max_correlation.items():
    print(f"  {rule}: {corr['max_correlation_value']:.4f}")
```

## API文档

### analyze_rules

```python
def analyze_rules(rule_score, rule_col='RULE', user_id_col='USER_ID', 
                 user_level_badrate_col=None, user_target_col=None,
                 hit_date_col=None)
```

#### 参数

| 参数 | 类型 | 描述 | 默认值 | 必要性 |
|------|------|------|--------|--------|
| `rule_score` | DataFrame | 规则拦截客户信息 | - | 必需 |
| `rule_col` | str | 规则名字段名 | 'RULE' | 可选 |
| `user_id_col` | str | 用户编号字段名 | 'USER_ID' | 可选 |
| `user_level_badrate_col` | str | 用户评级坏账率字段名 | None | 可选 |
| `user_target_col` | str | 用户实际逾期字段名 | None | 可选 |
| `hit_date_col` | str | 命中日期字段名 | None | 可选 |

#### 返回值

- DataFrame，包含所有规则的评估指标，根据输入参数不同，可能包含以下字段：
  - `rule`: 规则名称
  - 预估指标：`estimated_badrate_pred`, `estimated_recall_pred`, `estimated_precision_pred`, `estimated_lift_pred`
  - 实际指标：`actual_badrate`, `actual_recall`, `actual_precision`, `actual_lift`, `f1`
  - 命中率指标：`base_hit_rate`, `current_hit_rate`, `hit_rate_cv`
  - 相关性指标：`max_correlation_value`

### analyze_rule_correlation

```python
def analyze_rule_correlation(rule_score, rule_col='RULE', user_id_col='USER_ID')
```

#### 参数

- `rule_score`: DataFrame，规则拦截客户信息
- `rule_col`: str，规则名字段名，默认值为'RULE'
- `user_id_col`: str，用户编号字段名，默认值为'USER_ID'

#### 返回值

- `correlation_matrix`: DataFrame，规则间相关系数矩阵
- `max_correlation`: dict，每条规则与其他规则的最大相关性

### load_example_data

```python
def load_example_data(file_path=None)
```

#### 参数

- `file_path`: str，示例数据文件路径，默认使用内置示例数据

#### 返回值

- DataFrame，示例数据

## 结果解读指南

### 核心指标解读

| 指标 | 定义 | 意义 | 最佳范围 |
|------|------|------|----------|
| `base_hit_rate` | 历史平均命中率 | 规则的长期平均覆盖比例 | 依业务场景而定 |
| `current_hit_rate` | 当前命中率 | 规则的近期覆盖比例 | 依业务场景而定 |
| `hit_rate_cv` | 命中率变异系数 | 命中率的稳定性，CV = 标准差/均值 | < 0.2（稳定）, > 0.5（不稳定） |
| `actual_lift` | 实际lift值 | 规则命中样本逾期率 / 总样本逾期率 | > 1.0（有效）, > 2.0（高效） |
| `f1` | F1分数 | 2*(精确率*召回率)/(精确率+召回率) | 0-1，越高越好 |
| `max_correlation_value` | 最大相关性 | 与其他规则的最大相关系数 | < 0.5（独立性好）, > 0.8（高度相关） |

### 常见使用场景

#### 1. 评估单一规则的有效性
```python
# 查看特定规则的详细指标
specific_rule = result[result['rule'] == '特定规则名']
print(specific_rule[['rule', 'f1', 'actual_lift', 'hit_rate_cv']])
```

#### 2. 筛选高价值规则
```python
# 筛选lift值大于1.2且F1分数大于0.3的规则
high_value_rules = result[(result['actual_lift'] > 1.2) & (result['f1'] > 0.3)]
```

#### 3. 监控规则稳定性
```python
# 识别命中率不稳定的规则（CV > 0.5）
unstable_rules = result[result['hit_rate_cv'] > 0.5]
```

#### 4. 识别冗余规则
```python
# 识别与其他规则高度相关的规则
redundant_rules = []
for rule, corr in max_correlation.items():
    if abs(corr['max_correlation_value']) > 0.8:
        redundant_rules.append(rule)
print(f"高度相关的冗余规则：{redundant_rules}")
```

## 数据质量要求与最佳实践

### 数据质量要求

1. **完整性**：确保数据中包含唯一的用户标识和规则名称
2. **准确性**：确保评级坏账率字段格式正确，避免百分比字符串
3. **一致性**：确保同一用户在不同规则下的信息一致
4. **时效性**：使用最新的生产数据进行分析
5. **真实性**：确保实际逾期字段（USER_TARGET）准确反映用户的逾期情况

### 最佳实践

1. **分场景使用**：
   - 开发测试阶段：使用预估指标快速评估规则效果
   - 生产环境：结合实际指标和命中率监控全面评估规则

2. **定期分析**：
   - 每周/每月进行规则效度分析
   - 跟踪规则效果的时间变化趋势
   - 及时发现和处理效果下降的规则

3. **结合业务理解**：
   - 不要单纯依赖量化指标，结合业务经验进行判断
   - 考虑规则的业务逻辑和风险覆盖范围
   - 评估规则对整体风险控制的贡献

4. **持续优化**：
   - 根据分析结果调整规则参数
   - 移除无效或冗余规则
   - 新增互补性规则，提高整体风险覆盖

## 示例数据结构

示例数据包含以下字段：

| 字段名 | 描述 | 类型 | 示例值 |
| ---- | ---- | ---- | ------ |
| RULE | 规则名称 | 字符串 | 度小满欺诈分>=90 |
| USER_ID | 用户编号 | 字符串 | ID20221115003665 |
| HIT_DATE | 命中规则日期 | 日期 | 2022-10-01 |
| USER_LEVEL | 用户评级 | 整数 | 1 |
| USER_LEVEL_BADRATE | 用户评级对应的坏账率 | 字符串/数值 | 20.00% / 0.2 |
| USER_TARGET | 用户是否逾期 | 整数 | 1（逾期）/ 0（未逾期） |

## 许可证

MIT License

## 作者

aialgorithm <aialgorithm@example.com>

## 版本信息

当前版本：0.3.0

## 项目地址

GitHub: https://github.com/aialgorithm/rulelift
PyPI: https://pypi.org/project/rulelift/

## TODO
- 整合多个规则的评估结果，形成策略级结论
- 规则全局增益
- 增强实际场景数据处理能力
- 结果展示&操作可视化

## 更新日志

### v0.3.0 (2025-12-17)
- 新增命中率变异系数（hit_rate_cv）用于监控规则稳定性
- 新增F1分数计算，综合评估规则效果
- 优化规则相关性分析，新增最大相关性指标
- 改进命中率计算逻辑，历史命中率基于除当前日期外所有日期的平均值
- 修复了部分bug，提高了代码质量
- 完善了文档，新增了技术原理和缺陷分析内容


<img width="273" height="259" alt="image" src="https://github.com/user-attachments/assets/925325e4-bfe7-4186-be93-fd01248508b2" />

