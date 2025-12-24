import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_auc_score

class VariableAnalyzer:
    """
    变量分析模块，用于计算变量的效度指标和分箱分析
    
    Attributes:
        df: 输入的数据集
        exclude_cols: 排除的字段名列表
        target_col: 目标字段名，默认为'ISBAD'
        features: 待分析的特征列表
    """
    
    def __init__(self, df: pd.DataFrame, exclude_cols: List[str] = None, target_col: str = 'ISBAD'):
        """
        初始化变量分析器
        
        参数:
            df: 输入的数据集
            exclude_cols: 排除的字段名列表
            target_col: 目标字段名，默认为'ISBAD'
        """
        self.df = df.copy()
        self.target_col = target_col
        
        # 排除指定字段和非数值字段
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.exclude_cols.append(self.target_col)
        
        # 筛选出数值型特征
        self.features = [col for col in self.df.columns 
                         if col not in self.exclude_cols 
                         and pd.api.types.is_numeric_dtype(self.df[col])]
    
    def calculate_iv(self, feature: str, n_bins: int = 10) -> float:
        """
        计算信息值(Information Value)
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为10
            
        返回:
            float，信息值
        """
        # 对特征值进行等频分箱
        df = self.df[[feature, self.target_col]].dropna()
        df['bin'] = pd.qcut(df[feature], q=n_bins, duplicates='drop', labels=False)
        
        # 计算各分箱的好坏样本数
        bin_stats = df.groupby('bin').agg({
            self.target_col: ['count', 'sum']
        }).reset_index()
        bin_stats.columns = ['bin', 'total', 'bad']
        bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
        
        # 计算总体好坏样本数
        total_bad = bin_stats['bad'].sum()
        total_good = bin_stats['good'].sum()
        
        # 计算各分箱的WOE和IV
        bin_stats['bad_rate'] = bin_stats['bad'] / total_bad
        bin_stats['good_rate'] = bin_stats['good'] / total_good
        bin_stats['woe'] = np.log((bin_stats['bad_rate'] + 1e-10) / (bin_stats['good_rate'] + 1e-10))
        bin_stats['iv'] = (bin_stats['bad_rate'] - bin_stats['good_rate']) * bin_stats['woe']
        
        return bin_stats['iv'].sum()
    
    def calculate_ks(self, feature: str, n_bins: int = 10) -> float:
        """
        计算KS统计量
        
        参数:
            feature: 特征名
            n_bins: 分箱数量，默认为10
            
        返回:
            float，KS值
        """
        # 对特征值进行等频分箱
        df = self.df[[feature, self.target_col]].dropna()
        df['bin'] = pd.qcut(df[feature], q=n_bins, duplicates='drop', labels=False)
        
        # 计算各分箱的统计信息
        bin_stats = df.groupby('bin').agg({
            self.target_col: ['count', 'sum']
        }).reset_index()
        bin_stats.columns = ['bin', 'total', 'bad']
        bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
        
        # 计算累积分布
        bin_stats['cum_bad'] = bin_stats['bad'].cumsum()
        bin_stats['cum_good'] = bin_stats['good'].cumsum()
        
        # 计算累积百分比
        total_bad = bin_stats['bad'].sum()
        total_good = bin_stats['good'].sum()
        bin_stats['cum_bad_rate'] = bin_stats['cum_bad'] / total_bad
        bin_stats['cum_good_rate'] = bin_stats['cum_good'] / total_good
        
        # 计算KS值
        bin_stats['ks'] = abs(bin_stats['cum_bad_rate'] - bin_stats['cum_good_rate'])
        
        return bin_stats['ks'].max()
    
    def calculate_auc(self, feature: str) -> float:
        """
        计算AUC值
        
        参数:
            feature: 特征名
            
        返回:
            float，AUC值
        """
        df = self.df[[feature, self.target_col]].dropna()
        
        try:
            auc = roc_auc_score(df[self.target_col], df[feature])
        except ValueError:
            auc = 0.5  # 单类别情况，AUC为0.5
        
        return auc
    
    def analyze_all_variables(self) -> pd.DataFrame:
        """
        分析所有变量的效度指标
        
        返回:
            DataFrame，包含所有变量的效度指标
        """
        results = []
        
        for feature in self.features:
            # 计算各指标
            iv = self.calculate_iv(feature)
            ks = self.calculate_ks(feature)
            auc = self.calculate_auc(feature)
            
            # 添加到结果列表
            results.append({
                'variable': feature,
                'iv': iv,
                'ks': ks,
                'auc': auc
            })
        
        return pd.DataFrame(results).sort_values(by='iv', ascending=False)
    
    def analyze_single_variable(self, variable: str, n_bins: int = 10) -> pd.DataFrame:
        """
        分析单个变量的分箱情况
        
        参数:
            variable: 变量名
            n_bins: 分箱数量，默认为10
            
        返回:
            DataFrame，包含各分箱的统计信息
        """
        if variable not in self.features:
            raise ValueError(f"Variable {variable} is not in the list of numeric features.")
        
        # 对特征值进行等频分箱
        df = self.df[[variable, self.target_col]].dropna()
        df['bin'] = pd.qcut(df[variable], q=n_bins, duplicates='drop', labels=False)
        
        # 获取分箱边界
        bins = pd.qcut(df[variable], q=n_bins, duplicates='drop').unique()
        bin_edges = [bin.left for bin in bins] + [bins[-1].right]
        
        # 计算各分箱的统计信息
        bin_stats = df.groupby('bin').agg({
            variable: ['min', 'max'],
            self.target_col: ['count', 'sum']
        }).reset_index()
        bin_stats.columns = ['bin', 'min', 'max', 'total', 'bad']
        
        # 计算额外统计指标
        bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
        bin_stats['badrate'] = bin_stats['bad'] / bin_stats['total']
        
        # 计算累积统计量
        bin_stats['cum_total'] = bin_stats['total'].cumsum()
        bin_stats['cum_bad'] = bin_stats['bad'].cumsum()
        bin_stats['cum_good'] = bin_stats['good'].cumsum()
        bin_stats['cum_badrate'] = bin_stats['cum_bad'] / bin_stats['cum_total']
        
        # 计算KS值
        total_bad = bin_stats['bad'].sum()
        total_good = bin_stats['good'].sum()
        bin_stats['cum_bad_pct'] = bin_stats['cum_bad'] / total_bad
        bin_stats['cum_good_pct'] = bin_stats['cum_good'] / total_good
        bin_stats['ks'] = abs(bin_stats['cum_bad_pct'] - bin_stats['cum_good_pct'])
        
        return bin_stats
    
    def plot_variable_bins(self, variable: str, n_bins: int = 10) -> Any:
        """
        可视化变量分箱结果
        
        参数:
            variable: 变量名
            n_bins: 分箱数量，默认为10
            
        返回:
            matplotlib.pyplot对象
        """
        import matplotlib.pyplot as plt
        
        # 获取分箱统计信息
        bin_stats = self.analyze_single_variable(variable, n_bins)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制badrate和cum_badrate
        ax1.bar(bin_stats['bin'], bin_stats['badrate'], label='Bad Rate', alpha=0.7)
        ax1.plot(bin_stats['bin'], bin_stats['cum_badrate'], label='Cumulative Bad Rate', 
                marker='o', color='red', linewidth=2)
        ax1.set_title(f'{variable} - Bin Analysis')
        ax1.set_ylabel('Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制KS曲线
        ax2.plot(bin_stats['bin'], bin_stats['cum_bad_pct'], label='Cumulative Bad %', 
                marker='o', linewidth=2)
        ax2.plot(bin_stats['bin'], bin_stats['cum_good_pct'], label='Cumulative Good %', 
                marker='o', linewidth=2)
        ax2.plot(bin_stats['bin'], bin_stats['ks'], label='KS', 
                marker='o', color='red', linewidth=2)
        ax2.set_title(f'{variable} - KS Analysis')
        ax2.set_xlabel('Bin')
        ax2.set_ylabel('Percentage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return plt
