import pandas as pd


def preprocess_data(df, user_level_badrate_col=None):
    """预处理数据
    
    参数：
    - df: DataFrame，原始数据
    - user_level_badrate_col: str，用户评级坏账率字段名
    
    返回：
    - DataFrame，预处理后的数据
    """
    df = df.copy()
    
    # 处理百分比字符串为浮点数
    if user_level_badrate_col and user_level_badrate_col in df.columns:
        if df[user_level_badrate_col].dtype == 'object':
            df[user_level_badrate_col] = df[user_level_badrate_col].str.rstrip('%').astype(float) / 100
    
    return df
