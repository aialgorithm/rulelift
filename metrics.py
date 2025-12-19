import pandas as pd
import numpy as np

def calculate_estimated_metrics(rule_score, user_rule_df, user_id_col, user_level_badrate_col):
    """计算基于USER_LEVEL_BADRATE的预估指标
    
    参数：
    - rule_score: DataFrame，原始数据
    - user_rule_df: DataFrame，用户-规则矩阵
    - user_id_col: str，用户编号字段名
    - user_level_badrate_col: str，用户评级坏账率字段名
    
    返回：
    - dict，规则预估指标
    """
    # 获取唯一规则列表
    rules = user_rule_df.columns.tolist()
    
    # 计算总样本数
    total_samples = len(user_rule_df)
    
    # 计算每个用户的预估逾期概率
    user_badrate = rule_score.groupby(user_id_col)[user_level_badrate_col].first().to_dict()
    
    # 计算总预估逾期样本数
    total_estimated_bads = sum(user_badrate[user_id] for user_id in user_rule_df.index)
    
    estimated_metrics = {}
    
    for rule in rules:
        # 命中该规则的用户
        hit_users = user_rule_df[user_rule_df[rule] == 1].index
        hit_count = len(hit_users)
        
        if hit_count == 0:
            continue
        
        # 命中规则的预估逾期样本数
        estimated_bads = sum(user_badrate[user_id] for user_id in hit_users)
        
        # 计算指标
        estimated_badrate = estimated_bads / hit_count
        estimated_recall = estimated_bads / total_estimated_bads
        estimated_precision = estimated_bads / hit_count
        estimated_lift = estimated_badrate / (total_estimated_bads / total_samples)
        
        estimated_metrics[rule] = {
            'estimated_badrate_pred': estimated_badrate,
            'estimated_recall_pred': estimated_recall,
            'estimated_precision_pred': estimated_precision,
            'estimated_lift_pred': estimated_lift
        }
    
    return estimated_metrics

def calculate_actual_metrics(rule_score, user_rule_df, user_id_col, user_target_col):
    """计算基于USER_TARGET的实际指标
    
    参数：
    - rule_score: DataFrame，原始数据
    - user_rule_df: DataFrame，用户-规则矩阵
    - user_id_col: str，用户编号字段名
    - user_target_col: str，用户实际逾期字段名
    
    返回：
    - dict，规则实际指标
    """
    # 获取唯一规则列表
    rules = user_rule_df.columns.tolist()
    
    # 计算总样本数
    total_samples = len(user_rule_df)
    
    # 计算每个用户的实际逾期情况
    user_target = rule_score.groupby(user_id_col)[user_target_col].first().to_dict()
    
    # 计算总实际逾期样本数
    total_actual_bads = sum(user_target[user_id] for user_id in user_rule_df.index)
    
    actual_metrics = {}
    
    for rule in rules:
        # 命中该规则的用户
        hit_users = user_rule_df[user_rule_df[rule] == 1].index
        hit_count = len(hit_users)
        
        if hit_count == 0:
            continue
        
        # 命中规则的实际逾期样本数
        actual_bads = sum(user_target[user_id] for user_id in hit_users)
        
        # 计算指标
        actual_badrate = actual_bads / hit_count if hit_count > 0 else 0
        actual_recall = actual_bads / total_actual_bads if total_actual_bads > 0 else 0
        actual_precision = actual_bads / hit_count if hit_count > 0 else 0
        actual_lift = actual_badrate / (total_actual_bads / total_samples) if total_samples > 0 else 0
        
        # 计算F1分数，使用更简洁的字段名
        f1_score = 2 * (actual_precision * actual_recall) / (actual_precision + actual_recall) if (actual_precision + actual_recall) > 0 else 0
        
        actual_metrics[rule] = {
            'actual_badrate': actual_badrate,
            'actual_recall': actual_recall,
            'actual_precision': actual_precision,
            'actual_lift': actual_lift,
            'f1': f1_score  # 使用简洁的字段名
        }
    
    return actual_metrics

def calculate_rule_correlation(user_rule_df):
    """计算规则间相关性
    
    参数：
    - user_rule_df: DataFrame，用户-规则矩阵
    
    返回：
    - DataFrame，规则间相关系数矩阵
    """
    return user_rule_df.corr()
