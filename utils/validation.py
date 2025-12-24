def _validate_columns(df, required_columns):
    """验证数据列是否完整
    
    参数：
    - df: DataFrame，待验证数据
    - required_columns: list，必需的列名列表
    
    返回：
    - bool，验证结果
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列：{missing_columns}")
    return True
