import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import chi2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class MultiFeatureRuleMiner:
    """
    多特征交叉规则生成类，用于生成双特征交叉分析结果
    
    支持自定义分箱阈值、自动分箱、卡方分箱等多种分箱策略
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'ISBAD', 
                 amount_col: str = None, ovd_bal_col: str = None):
        """
        初始化多特征规则挖掘器
        
        参数:
            df: 输入的数据集
            target_col: 目标字段名，默认为'ISBAD'
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
        """
        self.df = df.copy()
        self.target_col = target_col
        self.amount_col = amount_col
        self.ovd_bal_col = ovd_bal_col
        
        # 筛选出数值型特征
        self.numeric_features = [col for col in self.df.columns 
                                if col != target_col and col != amount_col and col != ovd_bal_col
                                and pd.api.types.is_numeric_dtype(self.df[col])]
        
        # 筛选出类别型特征
        self.categorical_features = [col for col in self.df.columns 
                                    if col != target_col and col != amount_col and col != ovd_bal_col
                                    and not pd.api.types.is_numeric_dtype(self.df[col])]
    
    def _bin_numeric_feature(self, feature: str, max_bins: int = 20, 
                          custom_bins: List[float] = None) -> pd.Series:
        """
        对数值型特征进行分箱处理
        
        参数:
            feature: 特征名
            max_bins: 最大分箱数量，默认为20
            custom_bins: 自定义分箱阈值，默认为None（使用等频分箱）
            
        返回:
            分箱后的特征值（返回分箱范围区间）
        """
        data = self.df[feature].dropna()
        
        if custom_bins is not None:
            # 使用自定义分箱阈值
            bins = custom_bins
            # 使用pd.cut创建分箱区间标签
            binned = pd.cut(data, bins=bins, include_lowest=True, right=True)
        else:
            # 使用等频分箱
            discretizer = KBinsDiscretizer(n_bins=max_bins, encode='ordinal', strategy='quantile')
            discretizer.fit(data.values.reshape(-1, 1))
            # 获取分箱边界
            bin_edges = discretizer.bin_edges_[0]
            # 使用pd.cut创建分箱区间标签
            binned = pd.cut(data, bins=bin_edges, include_lowest=True, right=True)
        
        # 转换为Series，保留原始索引
        binned_series = binned
        
        # 填充缺失值
        result = pd.Series(index=self.df.index, dtype=object)
        result.loc[binned_series.index] = binned_series
        
        return result
    
    def _prepare_feature(self, feature: str, max_unique_threshold: int = 5, 
                       custom_bins: List[float] = None, binning_method: str = 'quantile') -> pd.Series:
        """
        准备特征，对取值较多的特征进行分箱处理
        
        参数:
            feature: 特征名
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            custom_bins: 自定义分箱阈值，默认为None
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            
        返回:
            处理后的特征值
        """
        unique_count = self.df[feature].nunique()
        
        if unique_count > max_unique_threshold:
            # 对数值型特征进行分箱
            if feature in self.numeric_features:
                if custom_bins is not None:
                    # 使用自定义分箱
                    return self._bin_numeric_feature(feature, max_bins=len(custom_bins)-1, 
                                                  custom_bins=custom_bins)
                elif binning_method == 'chi2':
                    # 使用卡方分箱
                    return self._bin_with_chi2(feature, max_bins=max_unique_threshold)
                else:
                    # 使用等频分箱（默认）
                    return self._bin_numeric_feature(feature, max_bins=max_unique_threshold)
            # 对类别型特征，保留前max_unique_threshold个最常见的值，其余归为"其他"
            else:
                top_values = self.df[feature].value_counts().head(max_unique_threshold).index
                return self.df[feature].apply(lambda x: x if x in top_values else '其他')
        else:
            return self.df[feature]
    
    def _bin_with_chi2(self, feature: str, max_bins: int = 5) -> pd.Series:
        """
        使用卡方检验进行分箱
        
        参数:
            feature: 特征名
            max_bins: 最大分箱数量
            
        返回:
            分箱后的特征值（返回分箱范围区间）
        """
        feature_values = self.df[feature].dropna()
        target_values = self.df.loc[feature_values.index, self.target_col]
        
        # 初始分箱
        discretizer = KBinsDiscretizer(n_bins=max_bins, encode='ordinal', strategy='quantile')
        discretizer.fit(feature_values.values.reshape(-1, 1))
        
        # 获取分箱边界
        bin_edges = discretizer.bin_edges_[0]
        
        # 计算卡方分数，合并相似分箱
        # 这里简化处理，实际应用中可以使用更复杂的卡方分箱算法
        # 使用pd.cut创建分箱区间标签
        binned = pd.cut(feature_values, bins=bin_edges, include_lowest=True, right=True)
        
        return pd.Series(binned, index=feature_values.index)
    
    def generate_cross_matrix(self, feature1: str, feature2: str, 
                             max_unique_threshold: int = 5,
                             custom_bins1: List[float] = None,
                             custom_bins2: List[float] = None,
                             binning_method: str = 'quantile') -> pd.DataFrame:
        """
        生成双特征交叉矩阵，包含badrate、样本占比等关键指标
        
        参数:
            feature1: 第一个特征名
            feature2: 第二个特征名
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            custom_bins1: 第一个特征的自定义分箱阈值，默认为None
            custom_bins2: 第二个特征的自定义分箱阈值，默认为None
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            
        返回:
            交叉矩阵，包含各种统计指标
        """
        # 准备两个特征
        prepared_feature1 = self._prepare_feature(feature1, max_unique_threshold, custom_bins1, binning_method)
        prepared_feature2 = self._prepare_feature(feature2, max_unique_threshold, custom_bins2, binning_method)
        
        # 创建交叉表数据
        cross_data = pd.DataFrame({
            'feature1': prepared_feature1,
            'feature2': prepared_feature2,
            'target': self.df[self.target_col]
        })
        
        # 如果有amount_col和ovd_bal_col，添加到cross_data中
        if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
            cross_data['amount'] = self.df[self.amount_col]
            cross_data['ovd_bal'] = self.df[self.ovd_bal_col]
        
        # 计算基本统计量
        # 计算每个交叉组合的样本数
        count_matrix = pd.crosstab(cross_data['feature1'], cross_data['feature2'], 
                                  rownames=[feature1], colnames=[feature2])
        
        # 计算每个交叉组合的坏样本数
        bad_count_matrix = pd.crosstab(cross_data['feature1'], cross_data['feature2'], 
                                      values=cross_data['target'], aggfunc='sum',
                                      rownames=[feature1], colnames=[feature2])
        
        # 计算每个交叉组合的坏样本率（避免除以零）
        badrate_matrix = bad_count_matrix.astype(float)
        for idx in badrate_matrix.index:
            for col in badrate_matrix.columns:
                if count_matrix.loc[idx, col] > 0:
                    badrate_matrix.loc[idx, col] = bad_count_matrix.loc[idx, col] / count_matrix.loc[idx, col]
                else:
                    badrate_matrix.loc[idx, col] = 0.0
        
        # 计算每个交叉组合的样本占比
        total_samples = count_matrix.sum().sum()
        sample_ratio_matrix = count_matrix.copy()
        if total_samples > 0:
            sample_ratio_matrix = count_matrix / total_samples
        else:
            sample_ratio_matrix = sample_ratio_matrix * 0.0
        
        # 计算lift值（避免除以零）
        # 总样本坏样本率
        total_badrate = self.df[self.target_col].mean()
        lift_matrix = badrate_matrix.copy()
        if total_badrate > 0:
            lift_matrix = badrate_matrix / total_badrate
        else:
            lift_matrix = lift_matrix * 0.0
        
        # 计算损失率指标（如果有amount_col和ovd_bal_col）
        loss_rate_matrix = pd.DataFrame(index=count_matrix.index, columns=count_matrix.columns, dtype=float)
        loss_lift_matrix = pd.DataFrame(index=count_matrix.index, columns=count_matrix.columns, dtype=float)
        
        if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
            # 计算整体损失率
            overall_df = self.df[[self.amount_col, self.ovd_bal_col, self.target_col]].dropna()
            if len(overall_df) > 0:
                total_amount_overall = overall_df[self.amount_col].sum()
                if total_amount_overall > 0:
                    # 只统计坏样本的ovd_bal
                    overall_df_bad = overall_df[overall_df[self.target_col] == 1]
                    total_ovd_bal_overall = overall_df_bad[self.ovd_bal_col].sum()
                    overall_loss_rate = total_ovd_bal_overall / total_amount_overall
                    
                    # 计算每个交叉组合的损失率
                    for f1_val in count_matrix.index:
                        for f2_val in count_matrix.columns:
                            mask = (cross_data['feature1'] == f1_val) & (cross_data['feature2'] == f2_val)
                            # 先获取完整的subset，不dropna
                            subset = cross_data[mask]
                            
                            if len(subset) > 0:
                                # 只统计坏样本的ovd_bal
                                subset_bad = subset[subset['target'] == 1].dropna(subset=['amount', 'ovd_bal'])
                                if len(subset_bad) > 0:
                                    total_amount_selected = subset_bad['amount'].sum()
                                    if total_amount_selected > 0:
                                        total_ovd_bal_bad_selected = subset_bad['ovd_bal'].sum()
                                        loss_rate = total_ovd_bal_bad_selected / total_amount_selected
                                        loss_rate_matrix.loc[f1_val, f2_val] = loss_rate
                                        
                                        # 计算损失率提升度
                                        loss_lift = loss_rate / overall_loss_rate if overall_loss_rate > 0 else 0.0
                                        loss_lift_matrix.loc[f1_val, f2_val] = loss_lift
        
        # 整合所有矩阵到一个MultiIndex DataFrame中
        # 创建一个空的MultiIndex DataFrame
        features = [feature1, feature2]
        metrics = ['count', 'bad_count', 'badrate', 'sample_ratio', 'lift']
        
        # 如果有损失率指标，添加到metrics中
        if self.amount_col and self.ovd_bal_col:
            metrics.extend(['loss_rate', 'loss_lift'])
        
        # 生成行索引（feature1的唯一值）
        rows = count_matrix.index
        # 生成列索引（feature2的唯一值 + 指标）
        cols = pd.MultiIndex.from_product([count_matrix.columns, metrics], 
                                         names=[feature2, 'metric'])
        
        # 填充数据
        cross_matrix = pd.DataFrame(index=rows, columns=cols)
        
        for f2_val in count_matrix.columns:
            cross_matrix[(f2_val, 'count')] = count_matrix[f2_val]
            cross_matrix[(f2_val, 'bad_count')] = bad_count_matrix[f2_val]
            cross_matrix[(f2_val, 'badrate')] = badrate_matrix[f2_val]
            cross_matrix[(f2_val, 'sample_ratio')] = sample_ratio_matrix[f2_val]
            cross_matrix[(f2_val, 'lift')] = lift_matrix[f2_val]
            
            if self.amount_col and self.ovd_bal_col:
                cross_matrix[(f2_val, 'loss_rate')] = loss_rate_matrix[f2_val]
                cross_matrix[(f2_val, 'loss_lift')] = loss_lift_matrix[f2_val]
        
        return cross_matrix
    
    def get_cross_rules(self, feature1: str, feature2: str, 
                       top_n: int = 10, metric: str = 'lift',
                       min_samples: int = 10, min_lift: float = 1.1,
                       max_unique_threshold: int = 5,
                       custom_bins1: List[float] = None,
                       custom_bins2: List[float] = None,
                       binning_method: str = 'quantile') -> pd.DataFrame:
        """
        从交叉矩阵中提取top规则，输出DataFrame格式
        
        参数:
            feature1: 第一个特征名
            feature2: 第二个特征名
            top_n: 返回的规则数量，默认为10
            metric: 排序指标，默认为'lift'
            min_samples: 最小样本数过滤，默认为10
            min_lift: 最小lift值过滤，默认为1.1
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            custom_bins1: 第一个特征的自定义分箱阈值，默认为None
            custom_bins2: 第二个特征的自定义分箱阈值，默认为None
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            
        返回:
            包含top规则的DataFrame，包含：feature1_value, feature2_value, count, bad_count, badrate, lift, sample_ratio, rule_description
        """
        # 生成交叉矩阵
        cross_matrix = self.generate_cross_matrix(feature1, feature2, max_unique_threshold, 
                                              custom_bins1, custom_bins2, binning_method)
        
        # 转换为长格式，方便排序
        # 根据pandas版本决定是否使用future_stack参数，确保兼容性
        import pandas as pd
        pd_version = pd.__version__.split('.')
        major_version = int(pd_version[0])
        minor_version = int(pd_version[1])
        
        if major_version >= 2 or (major_version == 1 and minor_version >= 21):
            # 对于pandas 1.21.0+和2.x版本，使用future_stack=True
            long_df = cross_matrix.stack(future_stack=True).reset_index()
        else:
            # 对于旧版本pandas，不使用future_stack参数
            long_df = cross_matrix.stack().reset_index()
        
        # 动态设置正确的列名，避免长度不匹配问题
        num_levels = len(long_df.columns) - 1  # 减去最后一列metric
        if num_levels == 2:
            long_df.rename(columns={long_df.columns[0]: 'feature1_value', long_df.columns[1]: 'feature2_value'}, inplace=True)
        else:
            # 处理更复杂的情况，使用通用列名
            long_df.columns = [f'level_{i}' for i in range(num_levels)] + ['metric']
            # 如果有两个索引级别，使用它们作为特征值
            if num_levels >= 2:
                long_df.rename(columns={long_df.columns[0]: 'feature1_value', long_df.columns[1]: 'feature2_value'}, inplace=True)
            else:
                # 简化处理，只保留metric列
                long_df.rename(columns={long_df.columns[0]: 'feature1_value'}, inplace=True)
                long_df['feature2_value'] = 'default'
        
        # 筛选指定指标的行
        metric_rows = long_df[long_df['metric'] == metric]
        
        # 过滤样本数过少的规则
        if 'count' in metric_rows.columns:
            metric_rows = metric_rows[metric_rows['count'] >= min_samples]
        
        # 过滤lift过低的规则
        if 'lift' in metric_rows.columns:
            metric_rows = metric_rows[metric_rows['lift'] >= min_lift]
        
        # 如果没有匹配的行，返回空DataFrame（但至少返回一个规则）
        if metric_rows.empty:
            # 尝试返回所有规则，按lift排序
            all_rows = long_df[long_df['metric'].notna()].sort_values(by='metric', ascending=False)
            top_rules = all_rows.head(top_n)
            
            # 添加规则描述
            top_rules['rule_description'] = top_rules.apply(
                lambda x: f"{feature1} = {x['feature1_value']} AND {feature2} = {x['feature2_value']}",
                axis=1
            )
            
            # 重新排列列顺序
            result_columns = ['feature1_value', 'feature2_value', 'count', 'bad_count', 'badrate', 'lift', 'sample_ratio', 'rule_description']
            
            # 如果metric是loss_rate或loss_lift，添加这些列
            if metric in ['loss_rate', 'loss_lift']:
                result_columns = ['feature1_value', 'feature2_value', 'count', 'bad_count', 'badrate', 'lift', 'sample_ratio', 'loss_rate', 'loss_lift', 'rule_description']
            
            # 添加缺失的列
            for col in result_columns:
                if col not in top_rules.columns:
                    if col == 'count':
                        top_rules[col] = 0
                    elif col == 'bad_count':
                        top_rules[col] = 0
                    elif col == 'badrate':
                        top_rules[col] = top_rules['metric']
                    elif col == 'sample_ratio':
                        top_rules[col] = 0.1
                    elif col == 'lift':
                        top_rules[col] = top_rules['metric']
                    elif col == 'loss_rate':
                        top_rules[col] = 0.0
                    elif col == 'loss_lift':
                        top_rules[col] = top_rules['metric']
            
            # 只保留指定的列
            top_rules = top_rules[result_columns]
            
            return top_rules
        
        # 按指定指标排序，返回top_n
        top_rules = metric_rows.sort_values(by='metric', ascending=False).head(top_n)
        
        # 添加规则描述
        top_rules['rule_description'] = top_rules.apply(
            lambda x: f"{feature1} = {x['feature1_value']} AND {feature2} = {x['feature2_value']}",
            axis=1
        )
        
        # 重新排列列顺序
        result_columns = ['feature1_value', 'feature2_value', 'count', 'bad_count', 'badrate', 'lift', 'sample_ratio', 'rule_description']
        
        # 如果metric是loss_rate或loss_lift，添加这些列
        if metric in ['loss_rate', 'loss_lift']:
            result_columns = ['feature1_value', 'feature2_value', 'count', 'bad_count', 'badrate', 'lift', 'sample_ratio', 'loss_rate', 'loss_lift', 'rule_description']
        
        # 添加缺失的列
        for col in result_columns:
            if col not in top_rules.columns:
                if col == 'count':
                    top_rules[col] = 0
                elif col == 'bad_count':
                    top_rules[col] = 0
                elif col == 'badrate':
                    top_rules[col] = top_rules['metric']
                elif col == 'sample_ratio':
                    top_rules[col] = 0.1
                elif col == 'lift':
                    top_rules[col] = top_rules['metric']
                elif col == 'loss_rate':
                    top_rules[col] = 0.0
                elif col == 'loss_lift':
                    top_rules[col] = top_rules['metric']
        
        # 只保留指定的列
        top_rules = top_rules[result_columns]
        
        return top_rules
    
    def plot_cross_heatmap(self, feature1: str, feature2: str, 
                          metric: str = 'lift', max_unique_threshold: int = 5,
                          custom_bins1: List[float] = None,
                          custom_bins2: List[float] = None,
                          binning_method: str = 'quantile',
                          figsize: Tuple[int, int] = (12, 10)):
        """
        绘制双特征交叉热力图
        
        参数:
            feature1: 第一个特征名
            feature2: 第二个特征名
            metric: 要可视化的指标，默认为'lift'
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            custom_bins1: 第一个特征的自定义分箱阈值，默认为None
            custom_bins2: 第二个特征的自定义分箱阈值，默认为None
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            figsize: 图表大小
            
        返回:
            matplotlib.pyplot对象
        """
        # 生成交叉矩阵
        cross_matrix = self.generate_cross_matrix(feature1, feature2, max_unique_threshold, 
                                              custom_bins1, custom_bins2, binning_method)
        
        # 提取指定指标的矩阵
        metric_matrix = cross_matrix.xs(metric, level='metric', axis=1)
        
        # 创建热力图
        plt.figure(figsize=figsize)
        sns.heatmap(metric_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   cbar_kws={'label': metric})
        
        plt.title(f'{feature1} vs {feature2} - {metric} Heatmap')
        plt.tight_layout()
        
        return plt
    
    def generate_all_cross_rules(self, top_n: int = 10, metric: str = 'lift',
                                max_unique_threshold: int = 5,
                                binning_method: str = 'quantile',
                                max_feature_pairs: int = 10) -> pd.DataFrame:
        """
        生成所有特征对的交叉规则，并返回top规则
        
        参数:
            top_n: 返回的规则数量，默认为10
            metric: 排序指标，默认为'lift'
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            max_feature_pairs: 最大生成的特征对数量，默认为10
            
        返回:
            包含top规则的DataFrame
        """
        # 生成特征对组合
        from itertools import combinations
        features = self.numeric_features + self.categorical_features
        feature_pairs = list(combinations(features, 2))[:max_feature_pairs]  # 限制特征对数量
        
        all_rules = []
        
        for feature1, feature2 in feature_pairs:
            try:
                rules = self.get_cross_rules(feature1, feature2, top_n=top_n, 
                                            metric=metric, max_unique_threshold=max_unique_threshold,
                                            binning_method=binning_method)
                if not rules.empty:
                    all_rules.append(rules)
            except Exception as e:
                print(f"Error generating rules for {feature1} and {feature2}: {e}")
                continue
        
        if not all_rules:
            return pd.DataFrame()
        
        # 合并所有规则，取top_n
        combined_rules = pd.concat(all_rules, ignore_index=True)
        return combined_rules.sort_values(by=metric, ascending=False).head(top_n)
    
    def generate_cross_matrices_excel(self, features_list: List[str] = None, 
                                    feature1: str = None, feature2: str = None,
                                    max_unique_threshold: int = 5,
                                    custom_bins1: List[float] = None,
                                    custom_bins2: List[float] = None,
                                    binning_method: str = 'quantile',
                                    output_path: str = 'cross_analysis.xlsx',
                                    metrics: List[str] = None):
        """
        生成交叉矩阵并保存为Excel文件
        
        输出格式：每个指标一个矩阵，纵向排列badrate、count、sample_ratio三个矩阵
        
        参数:
            features_list: 特征清单列表，用于多特征两两组合生成矩阵
            feature1: 第一个特征名（兼容旧版本，当features_list为None时使用）
            feature2: 第二个特征名（兼容旧版本，当features_list为None时使用）
            max_unique_threshold: 最大允许的唯一值数量阈值，超过则进行分箱，默认为5
            custom_bins1: 第一个特征的自定义分箱阈值，默认为None（仅当features_list长度为2时有效）
            custom_bins2: 第二个特征的自定义分箱阈值，默认为None（仅当features_list长度为2时有效）
            binning_method: 分箱方法，'quantile'（等频）或'chi2'（卡方），默认为'quantile'
            output_path: Excel输出路径，默认为'cross_analysis.xlsx'
            metrics: 要导出的指标列表，默认为['badrate', 'count', 'sample_ratio']
                     可选：'badrate', 'count', 'bad_count', 'sample_ratio', 'lift', 'loss_rate', 'loss_lift'
        """
        # 兼容旧版本：如果没有提供features_list，使用feature1和feature2
        if features_list is None:
            if feature1 is None or feature2 is None:
                raise ValueError("必须提供features_list或feature1和feature2参数")
            features_list = [feature1, feature2]
        
        # 设置默认指标
        if metrics is None:
            metrics = ['badrate', 'count', 'sample_ratio']
        
        # 验证指标是否有效
        valid_metrics = ['badrate', 'count', 'bad_count', 'sample_ratio', 'lift', 'loss_rate', 'loss_lift']
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"无效的指标 '{metric}'，必须是以下之一: {valid_metrics}")
        
        # 如果只有2个特征，使用自定义分箱（如果提供）
        if len(features_list) == 2:
            # 生成交叉矩阵
            cross_matrix = self.generate_cross_matrix(
                features_list[0], features_list[1], 
                max_unique_threshold=max_unique_threshold,
                custom_bins1=custom_bins1,
                custom_bins2=custom_bins2,
                binning_method=binning_method
            )
            
            # 提取指标矩阵
            matrices_dict = {}
            for metric in metrics:
                try:
                    matrices_dict[metric] = cross_matrix.xs(metric, level='metric', axis=1)
                except KeyError:
                    print(f"Warning: Metric '{metric}' not found in cross_matrix, skipping...")
            
            # 在每个矩阵的第一行第一列插入特征名称
            matrices_with_header = {}
            for metric_name, matrix in matrices_dict.items():
                # 创建包含特征名称的标题行
                matrix.index.name = f'{features_list[0]}_VS_{features_list[1]}'
                
                # 将标题行插入到第一行
                matrices_with_header[metric_name] = matrix
            
            # 创建Excel写入器
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 保存指标矩阵到不同的sheet
                for metric_name, matrix in matrices_with_header.items():
                    if metric_name in ['badrate', 'sample_ratio', 'loss_rate', 'loss_lift']:
                        matrix.to_excel(writer, sheet_name=metric_name, float_format='%.4f')
                    else:
                        matrix.to_excel(writer, sheet_name=metric_name)
            
            print(f"交叉矩阵已保存到: {output_path}")
            for metric_name, matrix in matrices_with_header.items():
                print(f"  - {metric_name}矩阵: {matrix.shape}")
            
            return matrices_with_header
        else:
            # 多特征两两组合生成矩阵
            from itertools import combinations
            feature_pairs = list(combinations(features_list, 2))
            
            print(f"生成 {len(feature_pairs)} 个特征对的交叉矩阵...")
            
            # 创建Excel写入器
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for i, (f1, f2) in enumerate(feature_pairs):
                    # 生成交叉矩阵
                    cross_matrix = self.generate_cross_matrix(
                        f1, f2, 
                        max_unique_threshold=max_unique_threshold,
                        custom_bins1=None,  # 多特征组合时不支持自定义分箱
                        custom_bins2=None,
                        binning_method=binning_method
                    )
                    
                    # 提取指标矩阵
                    matrices_dict = {}
                    for metric in metrics:
                        try:
                            matrices_dict[metric] = cross_matrix.xs(metric, level='metric', axis=1)
                        except KeyError:
                            print(f"Warning: Metric '{metric}' not found in cross_matrix for {f1} vs {f2}, skipping...")
                    
                    # 在每个矩阵的第一行第一列插入特征名称
                    matrices_with_header = {}
                    for metric_name, matrix in matrices_dict.items():
                        # 创建包含特征名称的标题行
                        matrix.index.name = f'{f1}_VS_{f2}'
                        
                        # 将标题行插入到第一行
                        matrices_with_header[metric_name] =  matrix 
                    
                    # 保存指标矩阵到不同的sheet（sheet名称格式：f1_vs_f2_metric）
                    for metric_name, matrix in matrices_with_header.items():
                        # 限制sheet名称长度（Excel sheet名称最多31个字符）
                        sheet_name_base = f"{f1[:10]}_vs_{f2[:10]}"
                        sheet_name = f"{sheet_name_base}_{metric_name}"
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        
                        if metric_name in ['badrate', 'sample_ratio', 'loss_rate', 'loss_lift']:
                            matrix.to_excel(writer, sheet_name=sheet_name, float_format='%.4f')
                        else:
                            matrix.to_excel(writer, sheet_name=sheet_name)
                    
                    print(f"  - 特征对 {f1} vs {f2}: {len(matrices_with_header)} 个指标矩阵")
            
            print(f"交叉矩阵已保存到: {output_path}")
            print(f"  总共生成 {len(feature_pairs)} 个特征对的交叉矩阵")
            
            return None
