import pandas as pd
import numpy as np
from ..utils.validation import _validate_columns
from ..metrics import calculate_strategy_pair_gain


def calculate_strategy_gain(rule_score, rule_col='RULE', user_id_col='USER_ID',
                           user_target_col='ISBAD', strategy_definitions=None, 
                           metric='gain_lift'):
    """计算两两策略间的增益值

    参数：
    - rule_score: DataFrame，规则拦截客户信息
    - rule_col: str，规则名字段名，默认值为'RULE'
    - user_id_col: str，用户编号字段名，默认值为'USER_ID'
    - user_target_col: str，用户实际逾期字段名，默认值为'ISBAD'
    - strategy_definitions: dict，策略定义，键为策略名，值为规则列表
    - metric: str，用于矩阵显示的主要指标，可选值：'gain_lift', 'gain_badrate', 'gain_users', 'gain_bads', 'gain_coverage', 'gain_recall'

    返回：
    - DataFrame，两两策略间的增益矩阵
    - dict，详细的策略间增益指标
    """
    # 验证必需列
    _validate_columns(rule_score, [rule_col, user_id_col, user_target_col])
    
    # 验证指标是否有效
    valid_metrics = ['gain_lift', 'gain_badrate', 'gain_users', 'gain_bads', 'gain_coverage', 'gain_recall']
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")
    
    # 获取用户-规则矩阵
    from .rule_analysis import get_user_rule_matrix
    user_rule_df = get_user_rule_matrix(rule_score, rule_col, user_id_col)
    
    # 获取用户实际逾期情况（直接使用Series，避免转换为dict）
    user_target = rule_score.groupby(user_id_col)[user_target_col].first()
    
    # 如果没有提供策略定义，则将每个规则作为单独的策略
    if strategy_definitions is None:
        strategy_definitions = {rule: [rule] for rule in user_rule_df.columns}
    
    # 计算两两策略间的增益
    strategy_names = list(strategy_definitions.keys())
    gain_matrix = pd.DataFrame(0.0, index=strategy_names, columns=strategy_names)  # 使用float类型初始化
    gain_details = {}
    
    # 计算策略B之后策略A的增益，即策略A在策略B之后新增的拦截用户的增益
    for strategy_a in strategy_names:
        gain_details[strategy_a] = {}
        for strategy_b in strategy_names:
            # 计算策略B执行后，策略A的增益
            gain = calculate_strategy_pair_gain(
                user_rule_df, 
                user_target, 
                strategy_definitions[strategy_a], 
                strategy_definitions[strategy_b]
            )
            
            # 将主要增益指标存入矩阵
            gain_matrix.loc[strategy_a, strategy_b] = gain[metric]
            
            # 保存详细增益信息
            gain_details[strategy_a][strategy_b] = gain
    
    return gain_matrix, gain_details
