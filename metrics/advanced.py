import pandas as pd
import numpy as np

def calculate_strategy_pair_gain(user_rule_df, user_target, strategy_a_rules, strategy_b_rules):
    """
    计算两两策略间的增益值
    
    参数：
    - user_rule_df: DataFrame，用户-规则矩阵
    - user_target: dict或Series，用户实际逾期情况
    - strategy_a_rules: list，策略A的规则列表
    - strategy_b_rules: list，策略B的规则列表
    
    返回：
    - dict，策略间增益指标：
      - gain_users: 策略A在策略B之后新增的拦截用户数
      - gain_bads: 新增拦截用户中的坏客户数
      - gain_badrate: 新增拦截用户的坏账率
      - gain_lift: 新增拦截用户坏账率相对于策略B坏账率的增益（倍数）
      - gain_coverage: 新增拦截用户占总样本的比例
      - gain_recall: 新增拦截用户中的坏客户占总坏客户的比例
      - b_hit_users: 策略B拦截的用户数
      - b_badrate: 策略B拦截用户的坏账率
    """
    # 确保用户目标是Series类型，便于向量化操作
    if isinstance(user_target, dict):
        user_target = pd.Series(user_target)
    
    # 过滤掉不存在的规则，确保只处理实际存在的规则
    existing_rules_a = [rule for rule in strategy_a_rules if rule in user_rule_df.columns]
    existing_rules_b = [rule for rule in strategy_b_rules if rule in user_rule_df.columns]
    
    # 计算总样本数和总逾期样本数
    total_samples = len(user_rule_df)
    total_actual_bads = user_target.sum()
    
    # 策略B拦截的用户
    if existing_rules_b:
        b_hit_mask = user_rule_df[existing_rules_b].sum(axis=1) > 0
        b_hit_users = user_rule_df[b_hit_mask].index
        b_hit_count = len(b_hit_users)
    else:
        # 如果策略B没有有效规则，返回默认结果
        return {
            'gain_users': 0,
            'gain_bads': 0,
            'gain_badrate': 0,
            'gain_lift': 0,
            'gain_coverage': 0,
            'gain_recall': 0,
            'b_hit_users': 0,
            'b_badrate': 0
        }
    
    if b_hit_count == 0:
        return {
            'gain_users': 0,
            'gain_bads': 0,
            'gain_badrate': 0,
            'gain_lift': 0,
            'gain_coverage': 0,
            'gain_recall': 0,
            'b_hit_users': 0,
            'b_badrate': 0
        }
    
    # 策略A拦截的用户
    if existing_rules_a:
        a_hit_mask = user_rule_df[existing_rules_a].sum(axis=1) > 0
        a_hit_users = user_rule_df[a_hit_mask].index
        a_hit_count = len(a_hit_users)
    else:
        # 如果策略A没有有效规则，返回默认结果
        return {
            'gain_users': 0,
            'gain_bads': 0,
            'gain_badrate': 0,
            'gain_lift': 0,
            'gain_coverage': 0,
            'gain_recall': 0,
            'b_hit_users': b_hit_count,
            'b_badrate': 0
        }
    
    # 策略A在策略B之后新增的拦截用户：被策略A拦截但未被策略B拦截的用户
    new_intercept_users = [user for user in a_hit_users if user not in b_hit_users]
    new_intercept_count = len(new_intercept_users)
    
    if new_intercept_count == 0:
        return {
            'gain_users': 0,
            'gain_bads': 0,
            'gain_badrate': 0,
            'gain_lift': 0,
            'gain_coverage': 0,
            'gain_recall': 0,
            'b_hit_users': b_hit_count,
            'b_badrate': 0
        }
    
    # 计算策略B拦截用户的坏账率
    b_bads = user_target.loc[b_hit_users].sum()
    b_badrate = b_bads / b_hit_count if b_hit_count > 0 else 0
    
    # 计算新增拦截用户的坏账率
    new_intercept_bads = user_target.loc[new_intercept_users].sum()
    new_intercept_badrate = new_intercept_bads / new_intercept_count if new_intercept_count > 0 else 0
    
    # 计算增益指标
    # gain_lift: 新增拦截用户坏账率相对于策略B坏账率的增益（倍数）
    gain_lift = new_intercept_badrate / b_badrate if b_badrate > 0 else 0
    gain_coverage = new_intercept_count / total_samples if total_samples > 0 else 0
    gain_recall = new_intercept_bads / total_actual_bads if total_actual_bads > 0 else 0
    
    return {
        'gain_users': new_intercept_count,
        'gain_bads': new_intercept_bads,
        'gain_badrate': new_intercept_badrate,
        'gain_lift': gain_lift,
        'gain_coverage': gain_coverage,
        'gain_recall': gain_recall,
        'b_hit_users': b_hit_count,
        'b_badrate': b_badrate
    }
