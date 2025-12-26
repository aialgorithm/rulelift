import pandas as pd
import numpy as np
from ..utils.validation import _validate_columns
from ..utils.data_processing import preprocess_data
from ..metrics import (
    calculate_estimated_metrics,
    calculate_actual_metrics,
    calculate_rule_correlation
)


def get_user_rule_matrix(df, rule_col, user_id_col):
    """获取用户-规则矩阵

    参数：
    - df: DataFrame，原始数据
    - rule_col: str，规则名字段名
    - user_id_col: str，用户编号字段名

    返回：
    - DataFrame，用户-规则矩阵，行是用户，列是规则，值为1表示命中
    """
    # 使用更高效的crosstab实现，避免多次聚合操作
    # 先去重，避免用户-规则对重复计数
    unique_pairs = df[[user_id_col, rule_col]].drop_duplicates()
    
    # 使用crosstab创建用户-规则矩阵
    user_rule_df = pd.crosstab(
        index=unique_pairs[user_id_col],
        columns=unique_pairs[rule_col]
    )
    
    # 确保矩阵中只有0和1
    user_rule_df = (user_rule_df > 0).astype(int)
    return user_rule_df


def analyze_rules(rule_score, rule_col='RULE', user_id_col='USER_ID',
                 user_level_badrate_col=None, user_target_col=None,
                 hit_date_col=None, metrics=None):
    """分析规则效度

    参数：
    - rule_score: DataFrame，规则拦截客户信息
    - rule_col: str，规则名字段名，默认值为'RULE'
    - user_id_col: str，用户编号字段名，默认值为'USER_ID'
    - user_level_badrate_col: str，用户评级坏账率字段名，可选
    - user_target_col: str，用户实际逾期字段名，可选
    - hit_date_col: str，命中日期字段名，用于命中率计算，可选
    - metrics: list，指定要计算的指标列表，可选，默认计算所有指标

    返回：
    - DataFrame，包含所有规则的评估指标
    """
    # 验证必需列
    required_columns = [rule_col, user_id_col]
    if user_level_badrate_col:
        required_columns.append(user_level_badrate_col)
    if user_target_col:
        required_columns.append(user_target_col)
    if hit_date_col:
        required_columns.append(hit_date_col)
    
    _validate_columns(rule_score, required_columns)
    
    # 预处理数据
    rule_score = preprocess_data(rule_score, user_level_badrate_col)
    
    # 获取用户-规则矩阵
    user_rule_df = get_user_rule_matrix(rule_score, rule_col, user_id_col)
    
    # 计算指标
    estimated_metrics = {}
    actual_metrics = {}
    hit_rate_metrics = {}
    correlation_metrics = {}
    
    # 计算预估指标
    if user_level_badrate_col:
        estimated_metrics = calculate_estimated_metrics(rule_score, user_rule_df, user_id_col, user_level_badrate_col)
    
    # 计算实际指标
    if user_target_col:
        actual_metrics = calculate_actual_metrics(rule_score, user_rule_df, user_id_col, user_target_col)
    
    # 计算命中率相关指标（仅当传入hit_date_col时执行）
    if hit_date_col:
        # 确保日期字段是datetime类型
        #rule_score[hit_date_col] = pd.to_datetime(rule_score[hit_date_col])
        
        # 获取所有日期
        all_dates = rule_score[hit_date_col].unique().tolist()
        all_dates.sort()
        
        # 当天日期（取最新日期）
        current_date = all_dates[-1]
        
        # 计算当天命中率
        current_data = rule_score[rule_score[hit_date_col] == current_date]
        current_user_rule = get_user_rule_matrix(current_data, rule_col, user_id_col)
        current_hit_counts = current_user_rule.sum()
        current_total_users = len(current_user_rule)
        current_hit_rates = current_hit_counts / current_total_users
        
        # 计算历史命中率（取除当前日期外所有日期的平均值）
        history_dates = all_dates[:-1]  # 除当前日期外的所有日期
        history_daily_hit_rates = []
        
        for date in history_dates:
            date_data = rule_score[rule_score[hit_date_col] == date]
            date_user_rule = get_user_rule_matrix(date_data, rule_col, user_id_col)
            date_hit_counts = date_user_rule.sum()
            date_total_users = len(date_user_rule)
            history_daily_hit_rates.append(date_hit_counts / date_total_users)
        
        # 计算所有日期的命中率（包括当前日期）
        all_daily_hit_rates = []
        for date in all_dates:
            date_data = rule_score[rule_score[hit_date_col] == date]
            date_user_rule = get_user_rule_matrix(date_data, rule_col, user_id_col)
            date_hit_counts = date_user_rule.sum()
            date_total_users = len(date_user_rule)
            all_daily_hit_rates.append(date_hit_counts / date_total_users)
        
        # 转换为DataFrame，方便计算
        history_daily_hit_rates_df = pd.DataFrame(history_daily_hit_rates)
        all_daily_hit_rates_df = pd.DataFrame(all_daily_hit_rates)
        
        # 计算历史命中率（平均值）
        base_hit_rates = history_daily_hit_rates_df.mean(axis=0)
        
        # 计算命中率变异系数（CV）
        hit_rate_means = all_daily_hit_rates_df.mean(axis=0)
        hit_rate_stds = all_daily_hit_rates_df.std(axis=0)
        hit_rate_cv = hit_rate_stds / hit_rate_means
        
        # 整合命中率指标
        for rule in user_rule_df.columns:
            hit_rate_metrics[rule] = {
                'base_hit_rate': base_hit_rates.get(rule, 0),
                'current_hit_rate': current_hit_rates.get(rule, 0),
                'hit_rate_cv': hit_rate_cv.get(rule, 0)
            }
    
    # 计算规则相关性指标
    _, correlation_metrics = analyze_rule_correlation(rule_score, rule_col, user_id_col)
    
    # 整合所有指标
    all_rules = set(estimated_metrics.keys()) | set(actual_metrics.keys()) | set(hit_rate_metrics.keys()) | set(correlation_metrics.keys())
    
    rule_info = []
    for rule in all_rules:
        # 整合指标
        metrics_dict = {
            'rule': rule,
            **estimated_metrics.get(rule, {}),
            **actual_metrics.get(rule, {}),
            **hit_rate_metrics.get(rule, {}),
            **correlation_metrics.get(rule, {})
        }
        
        # 移除不需要的指标
        if 'hit_rate_pred' in metrics_dict:
            del metrics_dict['hit_rate_pred']
        
        rule_info.append(metrics_dict)
    
    # 转换为DataFrame
    result_df = pd.DataFrame(rule_info)
    
    # 计算总样本数和总逾期率（仅在调试模式下打印）
    total_samples = len(user_rule_df)
    
    # 如果指定了指标列表，只返回指定的指标
    if metrics:
        # 确保rule列始终包含
        metrics = ['rule'] + [m for m in metrics if m != 'rule']
        # 过滤存在的列
        result_df = result_df[[col for col in metrics if col in result_df.columns]]
    
    return result_df


def analyze_rule_correlation(rule_score, rule_col, user_id_col):
    """分析规则间相关性

    参数：
    - rule_score: DataFrame，规则拦截客户信息
    - rule_col: str，规则名字段名，默认值为'RULE'
    - user_id_col: str，用户编号字段名，默认值为'USER_ID'

    返回：
    - DataFrame，规则间相关系数矩阵
    - dict，每条规则与其他规则的最大相关性
    """
    # 获取用户-规则矩阵
    user_rule_df = get_user_rule_matrix(rule_score, rule_col, user_id_col)
    
    # 计算规则间相关性
    correlation_matrix = calculate_rule_correlation(user_rule_df)
    
    # 计算每条规则与其他规则的最大相关性
    max_correlation = {}
    for rule in correlation_matrix.columns:
        # 排除与自身的相关性
        other_correlations = correlation_matrix[rule].drop(rule)
        max_correlation[rule] = {
            'max_correlation_value': other_correlations.max()
        }
    
    return correlation_matrix, max_correlation
