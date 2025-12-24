import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_rule_comparison(rules_df: pd.DataFrame, metrics: List[str] = ['lift', 'badrate', 'hit_rate'],
                        figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None):
    """
    绘制规则比较图
    
    参数:
        rules_df: 规则数据框，包含rule和要比较的指标列
        metrics: 要比较的指标列表
        figsize: 图表大小
        save_path: 保存路径，如'./rule_comparison.png'
    """
    # 确保规则名称存在
    if 'rule' not in rules_df.columns:
        raise ValueError("DataFrame must contain 'rule' column")
    
    # 确保所有指标存在
    for metric in metrics:
        if metric not in rules_df.columns:
            raise ValueError(f"DataFrame must contain '{metric}' column")
    
    # 创建子图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics):
        sns.barplot(x='rule', y=metric, data=rules_df, ax=axes[i], palette='viridis')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'Rule Comparison - {metric.capitalize()}')
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for p in axes[i].patches:
            height = p.get_height()
            axes[i].annotate(f'{height:.4f}',
                           xy=(p.get_x() + p.get_width() / 2., height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_rule_network(rules_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 15), save_path: Optional[str] = None):
    """
    绘制规则网络图
    
    参数:
        rules_df: 规则数据框，包含rule和lift列
        figsize: 图表大小
        save_path: 保存路径，如'./rule_network.png'
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for _, row in rules_df.iterrows():
        G.add_node(row['rule'], lift=row['lift'], 
                  badrate=row.get('badrate', 0),
                  hit_rate=row.get('hit_rate', 0))
    
    # 添加边（这里简单模拟，实际应根据规则间的关联关系）
    # 例如，根据规则的条件重叠度或相关性
    for i, row1 in rules_df.iterrows():
        for j, row2 in rules_df.iterrows():
            if i != j:
                # 简单模拟：lift高的规则指向lift低的规则
                if row1['lift'] > row2['lift']:
                    G.add_edge(row1['rule'], row2['rule'], weight=row1['lift'] - row2['lift'])
    
    # 设置节点大小和颜色
    node_sizes = [1000 * (G.nodes[node]['lift'] + 1) for node in G.nodes()]
    node_colors = [G.nodes[node]['lift'] for node in G.nodes()]
    
    # 绘制网络图
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          cmap='viridis', alpha=0.8)
    
    # 绘制边
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, 
                                  arrowsize=10)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Lift Value')
    
    plt.title('Rule Network Visualization')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_rule_distribution(rules_df: pd.DataFrame, metric: str = 'lift',
                          figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None):
    """
    绘制规则分布直方图
    
    参数:
        rules_df: 规则数据框
        metric: 要绘制分布的指标
        figsize: 图表大小
        save_path: 保存路径，如'./rule_distribution.png'
    """
    plt.figure(figsize=figsize)
    
    # 绘制直方图和密度曲线
    sns.histplot(rules_df[metric], kde=True, bins=20, color='skyblue', edgecolor='black')
    
    # 添加统计信息
    mean_val = rules_df[metric].mean()
    median_val = rules_df[metric].median()
    std_val = rules_df[metric].std()
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.4f}')
    
    plt.title(f'Distribution of Rule {metric.capitalize()}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_lift_precision_scatter(rules_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None):
    """
    绘制Lift-Precision散点图
    
    参数:
        rules_df: 规则数据框，包含lift和precision列
        figsize: 图表大小
        save_path: 保存路径，如'./lift_precision_scatter.png'
    """
    # 确保必要列存在
    if 'lift' not in rules_df.columns or 'precision' not in rules_df.columns:
        raise ValueError("DataFrame must contain 'lift' and 'precision' columns")
    
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    scatter = sns.scatterplot(x='precision', y='lift', data=rules_df, 
                             size='hit_rate' if 'hit_rate' in rules_df.columns else None,
                             sizes=(50, 500), alpha=0.7, hue='badrate' if 'badrate' in rules_df.columns else None)
    
    # 添加趋势线
    sns.regplot(x='precision', y='lift', data=rules_df, scatter=False, color='red', line_kws={'linestyle': '--'})
    
    plt.title('Lift vs Precision Scatter Plot')
    plt.xlabel('Precision')
    plt.ylabel('Lift')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def save_figure(plt_obj: plt, save_path: str, dpi: int = 300):
    """
    保存图表
    
    参数:
        plt_obj: matplotlib.pyplot对象
        save_path: 保存路径
        dpi: 分辨率
    """
    plt_obj.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt_obj.close()

def save_rule_report(rules_df: pd.DataFrame, report_path: str = './rule_report'):
    """
    保存规则报告，包含多个可视化图表
    
    参数:
        rules_df: 规则数据框
        report_path: 报告保存路径前缀
    """
    # 1. 规则比较图
    metrics = ['lift', 'badrate', 'hit_rate'] if all(m in rules_df.columns for m in ['lift', 'badrate', 'hit_rate']) else ['lift']
    plt1 = plot_rule_comparison(rules_df, metrics=metrics)
    save_figure(plt1, f'{report_path}_comparison.png')
    
    # 2. 规则分布图
    plt2 = plot_rule_distribution(rules_df, metric='lift')
    save_figure(plt2, f'{report_path}_distribution.png')
    
    # 3. Lift-Precision散点图（如果有precision列）
    if 'precision' in rules_df.columns:
        plt3 = plot_lift_precision_scatter(rules_df)
        save_figure(plt3, f'{report_path}_lift_precision.png')
    
    # 4. 规则网络图
    plt4 = plot_rule_network(rules_df)
    save_figure(plt4, f'{report_path}_network.png')
    
    # 5. 保存规则数据框为CSV
    rules_df.to_csv(f'{report_path}_data.csv', index=False, encoding='utf-8')

def save_figure(plt_obj: plt, save_path: str, dpi: int = 300):
    """
    保存图表
    
    参数:
        plt_obj: matplotlib.pyplot对象
        save_path: 保存路径
        dpi: 分辨率
    """
    plt_obj.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt_obj.close()
