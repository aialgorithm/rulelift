import pandas as pd
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    """计算Population Stability Index
    
    参数：
    - expected: Series，预期分布数据
    - actual: Series，实际分布数据
    - buckets: int，分箱数量
    
    返回：
    - float，PSI值
    """
    # 确保输入是Series
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    
    # 分箱处理
    try:
        # 使用等频分箱
        expected_ranks, bins = pd.qcut(expected, buckets, labels=False, duplicates='drop', retbins=True)
        # 使用相同的分箱边界处理实际数据
        actual_ranks = pd.cut(actual, bins=bins, labels=False, include_lowest=True)
    except ValueError:
        # 处理异常情况，如数据量过少
        expected_ranks = pd.Series(0, index=expected.index)
        actual_ranks = pd.Series(0, index=actual.index)
        bins = [expected.min(), expected.max()]
    
    # 计算分布
    expected_dist = expected_ranks.value_counts(normalize=True).sort_index()
    actual_dist = actual_ranks.value_counts(normalize=True).sort_index()
    
    # 合并分布，确保所有分箱都存在
    all_bins = sorted(set(expected_dist.index) | set(actual_dist.index))
    expected_dist = expected_dist.reindex(all_bins, fill_value=0)
    actual_dist = actual_dist.reindex(all_bins, fill_value=0)
    
    # 计算PSI
    psi_values = (actual_dist - expected_dist) * np.log((actual_dist + 1e-10) / (expected_dist + 1e-10))
    psi = psi_values.sum()
    
    return psi

def calculate_rule_psi(rule_score, rule_col, hit_date_col, user_id_col, time_splits=None):
    """计算规则在不同时期的PSI
    
    参数：
    - rule_score: DataFrame，原始数据
    - rule_col: str，规则名字段名
    - hit_date_col: str，命中日期字段名
    - user_id_col: str，用户编号字段名
    - time_splits: list，时间分割点，默认为按月份分割
    
    返回：
    - DataFrame，规则PSI值
    """
    # 确保日期字段是datetime类型
    rule_score[hit_date_col] = pd.to_datetime(rule_score[hit_date_col])
    
    # 按规则和日期分组，计算每个规则每天的命中率
    daily_hit_rate = rule_score.groupby([rule_col, pd.Grouper(key=hit_date_col, freq='D')])[user_id_col].nunique().reset_index()
    daily_hit_rate = daily_hit_rate.rename(columns={user_id_col: 'hit_count'})
    
    # 计算每天总用户数
    daily_total = rule_score.groupby(pd.Grouper(key=hit_date_col, freq='D'))[user_id_col].nunique().reset_index()
    daily_total = daily_total.rename(columns={user_id_col: 'total_count'})
    
    # 合并数据，计算命中率
    daily_metrics = pd.merge(daily_hit_rate, daily_total, on=hit_date_col, how='left')
    daily_metrics['hit_rate'] = daily_metrics['hit_count'] / daily_metrics['total_count']
    
    # 按规则计算PSI
    psi_results = []
    
    for rule in daily_metrics[rule_col].unique():
        # 获取该规则的命中率数据
        rule_data = daily_metrics[daily_metrics[rule_col] == rule].sort_values(hit_date_col)
        
        if len(rule_data) < 2:
            continue
        
        # 以第一个时间段为基准，计算后续每个时间段的PSI
        base_period = rule_data.iloc[0][hit_date_col]
        base_rate = rule_data.iloc[0]['hit_rate']
        
        for i in range(1, len(rule_data)):
            current_period = rule_data.iloc[i][hit_date_col]
            current_rate = rule_data.iloc[i]['hit_rate']
            
            # 计算PSI
            psi_value = calculate_psi([base_rate], [current_rate])
            
            psi_results.append({
                'rule': rule,
                'base_period': base_period,
                'current_period': current_period,
                'psi': psi_value,
                'base_hit_rate': base_rate,
                'current_hit_rate': current_rate
            })
    
    return pd.DataFrame(psi_results)

def calculate_rule_stability(rule_score, rule_col, hit_date_col, user_id_col):
    """计算规则稳定性指标
    
    参数：
    - rule_score: DataFrame，原始数据
    - rule_col: str，规则名字段名
    - hit_date_col: str，命中日期字段名
    - user_id_col: str，用户编号字段名
    
    返回：
    - dict，规则稳定性指标
    """
    # 确保日期字段是datetime类型
    rule_score[hit_date_col] = pd.to_datetime(rule_score[hit_date_col])
    
    # 按月份分组
    rule_score['month'] = rule_score[hit_date_col].dt.to_period('M')
    
    # 计算每个规则每月的命中率
    monthly_metrics = rule_score.groupby([rule_col, 'month']).agg(
        hit_count=pd.NamedAgg(column=user_id_col, aggfunc='nunique'),
        total_count=pd.NamedAgg(column=user_id_col, aggfunc=lambda x: rule_score[rule_score['month'] == x.name[1]][user_id_col].nunique())
    ).reset_index()
    
    monthly_metrics['hit_rate'] = monthly_metrics['hit_count'] / monthly_metrics['total_count']
    
    stability_results = {}
    
    for rule in monthly_metrics[rule_col].unique():
        rule_data = monthly_metrics[monthly_metrics[rule_col] == rule].sort_values('month')
        
        if len(rule_data) < 2:
            continue
        
        # 计算命中率标准差
        hit_rate_std = rule_data['hit_rate'].std()
        
        # 计算命中率变异系数
        hit_rate_mean = rule_data['hit_rate'].mean()
        hit_rate_cv = hit_rate_std / hit_rate_mean if hit_rate_mean > 0 else 0
        
        # 计算相邻月份命中率变化率
        monthly_changes = rule_data['hit_rate'].pct_change().dropna()
        max_monthly_change = monthly_changes.max()
        min_monthly_change = monthly_changes.min()
        avg_monthly_change = monthly_changes.mean()
        
        stability_results[rule] = {
            'hit_rate_std': hit_rate_std,
            'hit_rate_cv': hit_rate_cv,
            'max_monthly_change': max_monthly_change,
            'min_monthly_change': min_monthly_change,
            'avg_monthly_change': avg_monthly_change,
            'months_analyzed': len(rule_data)
        }
    
    return stability_results

def calculate_long_term_stability(rule_score, rule_col, hit_date_col, user_id_col, window_size=30):
    """计算规则长期稳定性
    
    参数：
    - rule_score: DataFrame，原始数据
    - rule_col: str，规则名字段名
    - hit_date_col: str，命中日期字段名
    - user_id_col: str，用户编号字段名
    - window_size: int，滚动窗口大小（天）
    
    返回：
    - dict，规则长期稳定性指标
    """
    # 确保日期字段是datetime类型
    rule_score[hit_date_col] = pd.to_datetime(rule_score[hit_date_col])
    
    # 按日期和规则分组，计算每天的命中用户数
    daily_hits = rule_score.groupby([hit_date_col, rule_col])[user_id_col].nunique().reset_index()
    daily_hits = daily_hits.rename(columns={user_id_col: 'hit_count'})
    
    # 计算每天总用户数
    daily_total = rule_score.groupby(hit_date_col)[user_id_col].nunique().reset_index()
    daily_total = daily_total.rename(columns={user_id_col: 'total_count'})
    
    # 合并数据，计算命中率
    daily_metrics = pd.merge(daily_hits, daily_total, on=hit_date_col, how='left')
    daily_metrics['hit_rate'] = daily_metrics['hit_count'] / daily_metrics['total_count']
    
    long_term_stability = {}
    
    for rule in daily_metrics[rule_col].unique():
        rule_data = daily_metrics[daily_metrics[rule_col] == rule].sort_values(hit_date_col)
        
        if len(rule_data) < window_size:
            continue
        
        # 计算滚动命中率均值和标准差
        rule_data.set_index(hit_date_col, inplace=True)
        rolling_mean = rule_data['hit_rate'].rolling(window=window_size).mean()
        rolling_std = rule_data['hit_rate'].rolling(window=window_size).std()
        
        # 计算滚动变异系数
        rolling_cv = rolling_std / rolling_mean
        
        long_term_stability[rule] = {
            'rolling_mean': rolling_mean.dropna().tolist(),
            'rolling_std': rolling_std.dropna().tolist(),
            'rolling_cv': rolling_cv.dropna().tolist(),
            'dates': rolling_mean.dropna().index.tolist(),
            'window_size': window_size
        }
    
    return long_term_stability
