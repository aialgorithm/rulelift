import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

class SingleFeatureRuleMiner:
    """
    单特征规则挖掘类，用于对数据各特征的不同阈值进行效度分布分析
    
    Attributes:
        df: 输入的数据集
        exclude_cols: 排除的字段名列表
        target_col: 目标字段名，默认为'ISBAD'
        features: 待分析的特征列表
    """
    
    def __init__(self, df: pd.DataFrame, exclude_cols: List[str] = None, target_col: str = 'ISBAD', 
                 amount_col: str = None, ovd_bal_col: str = None):
        """
        初始化单特征规则挖掘器
        
        参数:
            df: 输入的数据集
            exclude_cols: 排除的字段名列表
            target_col: 目标字段名，默认为'ISBAD'
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
        """
        self.df = df.copy()
        self.target_col = target_col
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
            'total_samples': total,
            'selected_samples': selected_count,
            'hit_rate': hit_rate,
            'total_bad': total_bad,
            'selected_bad': selected_bad,
            'badrate': badrate,
            'total_badrate': total_badrate,
            'lift': lift,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'loss_rate': loss_rate,
            'loss_lift': loss_lift
        }
    
    def analyze_feature(self, feature: str, n_bins: int = 20) -> pd.DataFrame:
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
        
        # 对特征值进行分箱，获取阈值列表
        data = self.df[feature].dropna().values.reshape(-1, 1)
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        discretizer.fit(data)
        
        # 获取分箱边界作为阈值
        thresholds = sorted(set(discretizer.bin_edges_[0]))
        
        # 过滤掉最小值和最大值，避免无意义的规则
        if len(thresholds) > 2:
            thresholds = thresholds[1:-1]  # 去掉最小值和最大值
        
        # 对每个阈值智能选择操作符，避免无意义规则
        results = []
        for i, threshold in enumerate(thresholds):
            # 只计算有意义的方向
            if i > 0:  # 不是最小值，可以计算<=
                results.append(self._calculate_metrics(feature, threshold, '<='))
            if i < len(thresholds) - 1:  # 不是最大值，可以计算>=
                results.append(self._calculate_metrics(feature, threshold, '>='))
        
        return pd.DataFrame(results)
    
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
        
        plt.title(f'Feature {feature} - {metric} Analysis')
        plt.xlabel(f'{feature} Threshold')
        plt.ylabel(metric.capitalize())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt
    
    def plot_badrate_vs_hitrate(self, feature: str, n_bins: int = 20, 
                               figsize: Tuple[int, int] = (12, 6)):
        """
        绘制坏账率 vs 命中率散点图
        
        参数:
            feature: 特征名
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
            plt.scatter(subset['hit_rate'], subset['badrate'], 
                       label=f'{operator}', alpha=0.7, s=50)
        
        plt.title(f'Feature {feature} - Badrate vs Hitrate')
        plt.xlabel('Hit Rate')
        plt.ylabel('Badrate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt
    
    # 兼容example.py的别名方法
    def calculate_single_feature_metrics(self, feature: str, num_bins: int = 20) -> pd.DataFrame:
        """
        计算单个特征的不同阈值的效度指标（别名方法，兼容example.py）
        
        参数:
            feature: 特征名
            num_bins: 分箱数量，默认为20
            
        返回:
            包含各阈值统计指标的DataFrame
        """
        return self.analyze_feature(feature, n_bins=num_bins)
    
    def plot_feature_metrics(self, feature: str, metric: str = 'lift', num_bins: int = 20, 
                           figsize: Tuple[int, int] = (12, 8)):
        """
        绘制单个特征的指标分布图（别名方法，兼容example.py）
        
        参数:
            feature: 特征名
            metric: 要可视化的指标，默认为'lift'
            num_bins: 分箱数量，默认为20
            figsize: 图表大小
            
        返回:
            matplotlib.pyplot对象
        """
        return self.plot_threshold_analysis(feature, metric=metric, n_bins=num_bins, figsize=figsize)
    
    def get_top_rules(self, feature: str = None, top_n: int = 10, metric: str = 'lift', 
                    min_samples: int = 10) -> pd.DataFrame:
        """
        获取单个特征的top规则或所有特征的top规则
        
        参数:
            feature: 特征名，默认为None（获取所有特征的top规则）
            top_n: 返回的规则数量，默认为10
            metric: 排序指标，默认为'lift'
            min_samples: 最小样本数过滤，默认为10
            
        返回:
            包含top规则的DataFrame
        """
        if feature is not None:
            # 获取单个特征的top规则
            feature_results = self.analyze_feature(feature)
            
            # 过滤样本数过少的规则
            if 'selected_samples' in feature_results.columns:
                feature_results = feature_results[feature_results['selected_samples'] >= min_samples]
            
            # 过滤lift接近1的规则（无区分度）
            if 'lift' in feature_results.columns:
                feature_results = feature_results[feature_results['lift'] > 1.1]
            
            top_rules = feature_results.sort_values(by=metric, ascending=False).head(top_n)
            
            # 添加规则描述
            top_rules['rule_description'] = top_rules.apply(
                lambda x: f"{feature} {x['operator']} {x['threshold']:.4f}",
                axis=1
            )
            
            # 添加sample_ratio列作为hit_rate的别名
            top_rules['sample_ratio'] = top_rules['hit_rate']
            
            return top_rules
        else:
            # 获取所有特征的top规则
            all_results = self.analyze_all_features()
            df_all = pd.concat(all_results.values(), ignore_index=True)
            
            # 过滤样本数过少的规则
            if 'selected_samples' in df_all.columns:
                df_all = df_all[df_all['selected_samples'] >= min_samples]
            
            # 过滤lift接近1的规则（无区分度）
            if 'lift' in df_all.columns:
                df_all = df_all[df_all['lift'] > 1.1]
            
            top_rules = df_all.sort_values(by=metric, ascending=False).head(top_n)
            
            # 添加规则描述
            top_rules['rule_description'] = top_rules.apply(
                lambda x: f"{x['feature']} {x['operator']} {x['threshold']:.4f}",
                axis=1
            )
            
            # 添加sample_ratio列作为hit_rate的别名
            top_rules['sample_ratio'] = top_rules['hit_rate']
            
            return top_rules
