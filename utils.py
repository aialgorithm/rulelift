import pandas as pd
import os

def load_example_data(file_path='./data/hit_rule_info.csv'):
    """加载示例数据
    
    参数：
    - file_path: str，示例数据文件路径
    
    返回：
    - DataFrame，示例数据
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"示例数据文件不存在：{file_path}")
    
    # 读取数据
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        return df
    except Exception as e:
        raise Exception(f"读取示例数据失败：{e}")

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
