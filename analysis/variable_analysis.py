import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_auc_score

class VariableAnalyzer:
    """
    变量分析模块，用于计算变量的效度指标、PSI和分箱分析
    
    Attributes:
        df: 输入的数据集
        exclude_cols: 排除的字段名列表
        target_col: 目标字段名，默认为'ISBAD'
        features: 待分析的特征列表
    """
    
    def __init__(self, df: pd.DataFrame, exclude_cols: List[str] = None, target_col: str = 'ISBAD', amount_col: str = None, ovd_bal_col: str = None):
        """
        初始化变量分析器
        
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
    
    def calculate_missing_rate(self, feature: str) -> float:
        """
        计算特征的缺失率
        
        参数:
            feature: 特征名
            
        返回:
            float，缺失率
        """
        total_count = len(self.df[feature])
        missing_count = self.df[feature].isna().sum()
        return missing_count / total_count if total_count > 0 else 0.0
    
    def calculate_single_value_rate(self, feature: str) -> float:
        """
        计算特征的单值率
        
        参数:
            feature: 特征名
            
        返回:
            float，单值率
        """
        total_count = len(self.df[feature])
        single_value_count = self.df[feature].value_counts().iloc[0] if len(self.df[feature].value_counts()) > 0 else 0
        return single_value_count / total_count if total_count > 0 else 0.0
    
    def calculate_psi(self, feature: str, baseline_df: pd.DataFrame = None, current_df: pd.DataFrame = None) -> float:
        """
        计算PSI（Population Stability Index）
        
        参数:
            feature: 特征名
            baseline_df: 基准数据集，默认为None（使用当前数据的前半部分）
            current_df: 当前数据集，默认为None（使用当前数据的后半部分）
            
        返回:
            float，PSI值
        """
        # 如果没有提供基准数据集，使用当前数据的前半部分作为基准
        if baseline_df is None:
            split_idx = len(self.df) // 2
            baseline_df = self.df.iloc[:split_idx]
            current_df = self.df.iloc[split_idx:]
        else:
            # 使用提供的基准和当前数据集
            baseline_df = baseline_df.copy()
            current_df = current_df.copy()
        
        # 确保两个数据集都有相同的列
        common_cols = [col for col in baseline_df.columns if col in current_df.columns]
        baseline_df = baseline_df[common_cols]
        current_df = current_df[common_cols]
        
        # 计算PSI
        psi_value = 0.0
        for col in common_cols:
            if col == feature:
                # 计算基准分布
                baseline_dist = baseline_df[col].value_counts(normalize=True)
                current_dist = current_df[col].value_counts(normalize=True)
                
                # 合并分布
                combined_dist = pd.concat([baseline_dist, current_dist], axis=1).fillna(0)
                
                # 计算PSI
                psi = ((combined_dist.iloc[1] - combined_dist.iloc[0]) ** 2).sum()
                psi_value += psi
        
        return psi_value / len(common_cols) if len(common_cols) > 0 else 0.0
    
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
            # 单类别情况，AUC为0.5
            auc = 0.5
        
        # 如果AUC小于0.5，取1-AUC
        if auc < 0.5:
            auc = 1 - auc
        
        return auc
    
    def calculate_mean_diff(self, feature: str) -> float:
        """
        计算特征均值差异
        
        参数:
            feature: 特征名
            
        返回:
            float，均值差异
        """
        df = self.df[[feature, self.target_col]].dropna()
        
        # 计算特征在好坏样本中的均值差异
        bad_mean = df[df[self.target_col] == 1][feature].mean()
        good_mean = df[df[self.target_col] == 0][feature].mean()
        
        return bad_mean - good_mean
    
    def calculate_corr_with_target(self, feature: str) -> float:
        """
        计算特征与目标变量的相关性
        
        参数:
            feature: 特征名
            
        返回:
            float，相关系数
        """
        df = self.df[[feature, self.target_col]].dropna()
        
        # 计算相关性
        corr = df.corr().loc[feature, self.target_col]
        
        return corr
    
    def calculate_loss_rate(self, feature: str, amount_col: str = None, ovd_bal_col: str = None) -> float:
        """
        计算损失率（金额口径）
        
        参数:
            feature: 特征名
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
            
        返回:
            float，损失率
        """
        if amount_col is None or amount_col not in self.df.columns:
            return 0.0
        
        if ovd_bal_col is None or ovd_bal_col not in self.df.columns:
            return 0.0
        
        df = self.df[[feature, self.target_col, amount_col, ovd_bal_col]].dropna()
        
        if len(df) == 0:
            return 0.0
        
        total_amount = df[amount_col].sum()
        if total_amount == 0:
            return 0.0
        
        total_ovd_bal_bad = df[df[self.target_col] == 1][ovd_bal_col].sum()
        
        loss_rate = total_ovd_bal_bad / total_amount
        
        return loss_rate
    
    def calculate_loss_lift(self, feature: str, amount_col: str = None, ovd_bal_col: str = None) -> float:
        """
        计算损失率提升度（金额口径）
        
        参数:
            feature: 特征名
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
            
        返回:
            float，损失率提升度
        """
        if amount_col is None or amount_col not in self.df.columns:
            return 0.0
        
        if ovd_bal_col is None or ovd_bal_col not in self.df.columns:
            return 0.0
        
        df = self.df[[feature, self.target_col, amount_col, ovd_bal_col]].dropna()
        
        if len(df) == 0:
            return 0.0
        
        total_amount = df[amount_col].sum()
        if total_amount == 0:
            return 0.0
        
        # 计算当前特征的损失率
        total_ovd_bal_bad = df[df[self.target_col] == 1][ovd_bal_col].sum()
        feature_loss_rate = total_ovd_bal_bad / total_amount
        
        # 计算整体损失率
        total_ovd_bal_all = df[ovd_bal_col].sum()
        overall_loss_rate = total_ovd_bal_all / total_amount
        
        # 计算损失率提升度
        loss_lift = feature_loss_rate / overall_loss_rate if overall_loss_rate > 0 else 0.0
        
        return loss_lift
    
    def analyze_all_variables(self) -> pd.DataFrame:
        """
        分析所有变量的效度指标和统计信息
        
        返回:
            DataFrame，包含所有变量的效度指标和统计信息
        """
        results = []
        
        for feature in self.features:
            # 计算各指标
            iv = self.calculate_iv(feature)
            ks = self.calculate_ks(feature)
            auc = self.calculate_auc(feature)
            missing_rate = self.calculate_missing_rate(feature)
            single_value_rate = self.calculate_single_value_rate(feature)
            mean_diff = self.calculate_mean_diff(feature)
            corr_with_target = self.calculate_corr_with_target(feature)
            psi = self.calculate_psi(feature)
            
            # 计算统计信息
            feature_data = self.df[feature]
            min_value = feature_data.min()
            max_value = feature_data.max()
            median_value = feature_data.median()
            
            # 添加到结果列表
            results.append({
                'variable': feature,
                'iv': iv,
                'ks': ks,
                'auc': auc,
                'missing_rate': missing_rate,
                'single_value_rate': single_value_rate,
                'min_value': min_value,
                'max_value': max_value,
                'median_value': median_value,
                'mean_diff': mean_diff,
                'corr_with_target': corr_with_target,
                'psi': psi
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
        bin_stats['pct'] = bin_stats['total']/bin_stats['total'].sum()
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
        
        # 计算损失率指标（如果有amount_col和ovd_bal_col）
        if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
            # 为每个分箱计算损失率
            bin_stats['loss_rate'] = 0.0
            bin_stats['loss_lift'] = 0.0
            
            # 计算整体损失率
            overall_df = self.df[[self.amount_col, self.ovd_bal_col, self.target_col]].dropna()
            if len(overall_df) > 0:
                total_amount_overall = overall_df[self.amount_col].sum()
                if total_amount_overall > 0:
                    total_ovd_bal_overall = overall_df[self.ovd_bal_col].sum()
                    overall_loss_rate = total_ovd_bal_overall / total_amount_overall
                    
                    # 为每个分箱计算损失率和损失率提升度
                    for bin_idx in bin_stats['bin']:
                        # 获取该分箱的样本
                        bin_mask = (df['bin'] == bin_idx)
                        bin_indices = df[bin_mask].index
                        
                        # 计算该分箱的损失率
                        bin_subset = self.df.loc[bin_indices][[self.amount_col, self.ovd_bal_col, self.target_col]].dropna()
                        if len(bin_subset) > 0:
                            total_amount_bin = bin_subset[self.amount_col].sum()
                            if total_amount_bin > 0:
                                total_ovd_bal_bad_bin = bin_subset[bin_subset[self.target_col] == 1][self.ovd_bal_col].sum()
                                loss_rate = total_ovd_bal_bad_bin / total_amount_bin
                                loss_lift = loss_rate / overall_loss_rate if overall_loss_rate > 0 else 0.0
                                
                                bin_stats.loc[bin_stats['bin'] == bin_idx, 'loss_rate'] = loss_rate
                                bin_stats.loc[bin_stats['bin'] == bin_idx, 'loss_lift'] = loss_lift
        
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
