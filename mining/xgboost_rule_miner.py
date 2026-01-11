import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostRuleMiner:
    """
    XGBoost规则挖掘类，参考XGBoost算法优化单特征规则挖掘
    
    Attributes:
        df: 输入的数据集
        exclude_cols: 排除的字段名列表
        target_col: 目标字段名，默认为'ISBAD'
        features: 待分析的特征列表
        algorithm: 使用的算法，'xgb'或'chi2'
    """
    
    def __init__(self, df: pd.DataFrame, exclude_cols: List[str] = None, 
                 target_col: str = 'ISBAD', algorithm: str = 'xgb', precision: float = 0.1,
                 amount_col: str = None, ovd_bal_col: str = None):
        """
        初始化XGBoost规则挖掘器
        
        参数:
            df: 输入的数据集
            exclude_cols: 排除的字段名列表
            target_col: 目标字段名，默认为'ISBAD'
            algorithm: 使用的算法，'xgb'或'chi2'，默认为'xgb'
            precision: 精度控制参数，默认为0.1（10%的精度）
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
        """
        self.df = df.copy()
        self.target_col = target_col
        self.algorithm = algorithm
        self.precision = precision
        self.amount_col = amount_col
        self.ovd_bal_col = ovd_bal_col
        
        # 排除指定字段和非数值字段
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.exclude_cols.append(self.target_col)
        if self.amount_col:
            self.exclude_cols.append(self.amount_col)
        if self.ovd_bal_col:
            self.exclude_cols.append(self.ovd_bal_col)
        
        # 筛选出数值型特征
        self.features = [col for col in self.df.columns 
                         if col not in self.exclude_cols 
                         and pd.api.types.is_numeric_dtype(self.df[col])]
    
    def _calculate_metrics(self, feature: str, threshold: float, operator: str = '>=') -> Dict[str, float]:
        """
        计算单个阈值下的统计指标
        
        参数:
            feature: 特征名
            threshold: 阈值
            operator: 操作符，支持'>='或'<='
            
        返回:
            包含各类统计指标的字典
        """
        # 根据操作符筛选数据
        if operator == '>=':
            selected = self.df[self.df[feature] >= threshold]
        else:  # '<='
            selected = self.df[self.df[feature] <= threshold]
        
        total = len(self.df)
        selected_count = len(selected)
        
        # 总样本中的坏样本数
        total_bad = self.df[self.target_col].sum()
        # 选中样本中的坏样本数
        selected_bad = selected[self.target_col].sum()
        
        # 计算指标
        hit_rate = selected_count / total if total > 0 else 0  # 命中率
        badrate = selected_bad / selected_count if selected_count > 0 else 0  # 坏账率
        
        # 总坏账率
        total_badrate = total_bad / total if total > 0 else 0
        
        # lift值
        lift = badrate / total_badrate if total_badrate > 0 else 0
        
        # 召回率
        recall = selected_bad / total_bad if total_bad > 0 else 0
        
        # 精确率
        precision = selected_bad / selected_count if selected_count > 0 else 0
        
        # F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算损失率指标（如果有amount_col和ovd_bal_col）
        loss_rate = 0.0
        loss_lift = 0.0
        
        if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
            # 计算选中样本的损失率
            selected_df = selected[[self.amount_col, self.ovd_bal_col, self.target_col]].dropna()
            if len(selected_df) > 0:
                total_amount_selected = selected_df[self.amount_col].sum()
                if total_amount_selected > 0:
                    total_ovd_bal_bad_selected = selected_df[selected_df[self.target_col] == 1][self.ovd_bal_col].sum()
                    loss_rate = total_ovd_bal_bad_selected / total_amount_selected
                    
                    # 计算整体损失率
                    overall_df = self.df[[self.amount_col, self.ovd_bal_col, self.target_col]].dropna()
                    if len(overall_df) > 0:
                        total_amount_overall = overall_df[self.amount_col].sum()
                        if total_amount_overall > 0:
                            total_ovd_bal_overall = overall_df[self.ovd_bal_col].sum()
                            overall_loss_rate = total_ovd_bal_overall / total_amount_overall
                            loss_lift = loss_rate / overall_loss_rate if overall_loss_rate > 0 else 0.0
        
        return {
            'feature': feature,
            'threshold': threshold,
            'operator': operator,
            'sample_count': selected_count,
            'sample_ratio': hit_rate,
            'badrate': badrate,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'lift': lift,
            'loss_rate': loss_rate,
            'loss_lift': loss_lift
        }
    
    def _calculate_xgb_importance(self, feature: str, n_bins: int = 100) -> np.ndarray:
        """
        计算XGBoost风格的特征重要性（基于改进的信息增益算法）
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为20
            
        返回:
            各阈值的重要性分数
        """
        # 计算特征值分布
        feature_values = self.df[feature].dropna().values
        target_values = self.df.loc[self.df[feature].dropna().index, self.target_col].values
        
        # 等频分箱
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        bins = discretizer.fit_transform(feature_values.reshape(-1, 1)).flatten()
        
        # 获取分箱边界
        thresholds = sorted(set(discretizer.bin_edges_[0]))
        
        # 计算每个阈值的重要性（基于改进的梯度提升思路）
        importance_scores = []
        for threshold in thresholds:
            # 计算左右子集的统计信息
            left_mask = feature_values < threshold
            right_mask = feature_values >= threshold
            
            left_bad = target_values[left_mask].sum()
            right_bad = target_values[right_mask].sum()
            
            left_count = left_mask.sum()
            right_count = right_mask.sum()
            
            # 计算信息增益（改进版）
            total_bad = left_bad + right_bad
            total_count = left_count + right_count
            
            if total_count == 0:
                importance = 0
            else:
                # 计算基尼不纯度
                left_impurity = 1 - (left_bad / left_count) ** 2 if left_count > 0 else 0
                right_impurity = 1 - (right_bad / right_count) ** 2 if right_count > 0 else 0
                parent_impurity = 1 - (total_bad / total_count) ** 2 if total_count > 0 else 0
                
                # 信息增益（改进版，考虑样本权重）
                left_weight = left_count / total_count
                right_weight = right_count / total_count
                importance = parent_impurity - left_weight * left_impurity - right_weight * right_impurity
            
            importance_scores.append(importance)
        
        return np.array(importance_scores)
    
    def _calculate_chi2_score(self, feature: str, threshold: float) -> float:
        """
        计算卡方检验分数
        
        参数:
            feature: 特征名
            threshold: 阈值
            
        返回:
            卡方检验分数
        """
        # 创建列联表
        feature_data = self.df[[feature, self.target_col]].dropna()
        feature_values = feature_data[feature].values
        target_values = feature_data[self.target_col].values
        
        # 根据阈值分割
        left_mask = feature_values < threshold
        right_mask = feature_values >= threshold
        
        # 创建2x2列联表
        try:
            contingency_table = pd.crosstab(
                pd.Series(['left' if m else 'right' for m in left_mask]),
                pd.Series(target_values)
            )
            
            # 计算卡方检验
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            return chi2_stat
        except Exception as e:
            print(f"Warning: Chi2 calculation failed for threshold {threshold}: {e}")
            return 0.0
    
    def _find_optimal_thresholds(self, feature: str, n_bins: int = 200) -> List[float]:
        """
        寻找最优阈值（改进版）
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为200（受precision参数控制）
            
        返回:
            最优阈值列表
        """
        # 根据precision参数调整分箱数量
        if self.algorithm in ['xgb', 'chi2']:
            # 使用precision参数控制分箱精度
            n_bins = max(10, int(n_bins * self.precision))
        
        # 对特征值进行分箱，获取阈值列表
        feature_values = self.df[feature].dropna().values
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        discretizer.fit(feature_values.reshape(-1, 1))
        
        # 获取分箱边界作为阈值
        thresholds = sorted(set(discretizer.bin_edges_[0]))
        
        if self.algorithm == 'xgb':
            # 使用XGBoost风格的重要性排序
            importance_scores = self._calculate_xgb_importance(feature, n_bins)
            # 选择重要性最高的阈值
            optimal_thresholds = [thresholds[i] for i in np.argsort(importance_scores)[-10:]]  # 增加到10个
        elif self.algorithm == 'chi2':
            # 使用卡方检验选择最佳阈值
            chi2_scores = [self._calculate_chi2_score(feature, threshold) for threshold in thresholds]
            # 选择卡方分数最高的阈值
            optimal_thresholds = [thresholds[i] for i in np.argsort(chi2_scores)[-10:]]  # 增加到10个
        else:
            # 默认使用所有阈值
            optimal_thresholds = thresholds
        
        return optimal_thresholds
    
    def analyze_feature(self, feature: str, n_bins: int = 2000) -> pd.DataFrame:
        """
        分析单个特征的不同阈值下的效度分布
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为20
            
        返回:
            包含所有阈值指标的DataFrame
        """
        if feature not in self.features:
            raise ValueError(f"Feature {feature} is not in the list of numeric features.")
        
        # 寻找最优阈值
        optimal_thresholds = self._find_optimal_thresholds(feature, n_bins)
        
        # 对每个最优阈值计算>=和<=两种情况的指标
        results = []
        for threshold in optimal_thresholds:
            results.append(self._calculate_metrics(feature, threshold, '>='))
            results.append(self._calculate_metrics(feature, threshold, '<='))
        
        df_result = pd.DataFrame(results)
        
        # 添加规则描述列
        df_result['rule_description'] = df_result.apply(
            lambda x: f"{x['feature']} {x['operator']} {x['threshold']:.4f}",
            axis=1
        )
        
        return df_result
    
    def analyze_all_features(self, n_bins: int = 20) -> Dict[str, pd.DataFrame]:
        """
        分析所有特征的不同阈值下的效度分布
        
        参数:
            n_bins: 分箱数量，默认为20
            
        返回:
            字典，键为特征名，值为包含该特征所有阈值指标的DataFrame
        """
        results = {}
        for feature in self.features:
            results[feature] = self.analyze_feature(feature, n_bins)
        return results
    
    def get_top_rules(self, feature: str = None, top_n: int = 10, metric: str = 'lift') -> pd.DataFrame:
        """
        获取单个特征的top规则或所有特征的top规则
        
        参数:
            feature: 特征名，默认为None（获取所有特征的top规则）
            top_n: 返回的规则数量，默认为10
            metric: 排序指标，默认为'lift'
            
        返回:
            包含top规则的DataFrame
        """
        if feature is not None:
            # 获取单个特征的top规则
            feature_results = self.analyze_feature(feature)
            top_rules = feature_results.sort_values(by=metric, ascending=False).head(top_n)
            return top_rules
        else:
            # 获取所有特征的top规则
            all_results = self.analyze_all_features()
            df_all = pd.concat(all_results.values(), ignore_index=True)
            top_rules = df_all.sort_values(by=metric, ascending=False).head(top_n)
            return top_rules
    
    def plot_feature_importance(self, feature: str, n_bins: int = 20, figsize: Tuple[int, int] = (12, 6)):
        """
        绘制特征重要性图
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为20
            figsize: 图表大小
        """
        if self.algorithm == 'xgb':
            # 计算XGBoost风格的重要性
            importance_scores = self._calculate_xgb_importance(feature, n_bins)
            
            # 获取阈值
            feature_values = self.df[feature].dropna().values
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            discretizer.fit(feature_values.reshape(-1, 1))
            thresholds = sorted(set(discretizer.bin_edges_[0]))
            
            # 创建图表
            plt.figure(figsize=figsize)
            plt.plot(thresholds, importance_scores, marker='o', linewidth=2)
            plt.title(f'Feature {feature} - XGBoost Importance')
            plt.xlabel(f'{feature} Threshold')
            plt.ylabel('Importance Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        else:
            # 计算卡方分数
            feature_values = self.df[feature].dropna().values
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            discretizer.fit(feature_values.reshape(-1, 1))
            thresholds = sorted(set(discretizer.bin_edges_[0]))
            
            chi2_scores = [self._calculate_chi2_score(feature, threshold) for threshold in thresholds]
            
            # 创建图表
            plt.figure(figsize=figsize)
            plt.plot(thresholds, chi2_scores, marker='o', linewidth=2)
            plt.title(f'Feature {feature} - Chi2 Score')
            plt.xlabel(f'{feature} Threshold')
            plt.ylabel('Chi2 Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        return plt
    
    def plot_threshold_analysis(self, feature: str, metric: str = 'lift', n_bins: int = 20, 
                               figsize: Tuple[int, int] = (12, 6)):
        """
        绘制特征阈值分析图
        
        参数:
            feature: 特征名
            metric: 要可视化的指标，默认为'lift'
            n_bins: 分箱数量，默认为20
            figsize: 图表大小
        """
        # 分析特征
        df_result = self.analyze_feature(feature, n_bins)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 分别绘制>=和<=两种情况
        for operator in ['>=', '<=']:
            subset = df_result[df_result['operator'] == operator]
            plt.plot(subset['threshold'], subset[metric], label=f'{operator}', marker='o', markersize=5)
        
        plt.title(f'Feature {feature} - {metric} Analysis ({self.algorithm.upper()})')
        plt.xlabel(f'{feature} Threshold')
        plt.ylabel(metric.capitalize())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt
