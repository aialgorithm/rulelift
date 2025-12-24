import pandas as pd
import os

def load_example_data(data_name='hit_rule_info', file_path=None):
    """加载示例数据
    
    参数：
    - data_name: str，示例数据名称，可选值：'hit_rule_info'、'feas_target'、'hit_rule_info.csv'、'feas_target.csv'
    - file_path: str，自定义示例数据文件路径
    
    返回：
    - DataFrame，示例数据
    """
    # 移除文件名中的扩展名，以便匹配数据名称
    data_name = os.path.splitext(data_name)[0]
    
    # 确定数据文件路径
    if file_path is None:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_name == 'feas_target':
            file_path = os.path.join(current_dir, 'data', 'feas_target.csv')
        else:  # 默认加载hit_rule_info
            file_path = os.path.join(current_dir, 'data', 'hit_rule_info.csv')
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"示例数据文件不存在：{file_path}")
    
    # 读取数据
    try:
        # 尝试不同编码
        encodings = ['gbk', 'utf-8', 'latin-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception(f"无法读取文件 {file_path}，尝试了多种编码均失败")
        
        return df
    except Exception as e:
        raise Exception(f"读取示例数据失败：{e}")
