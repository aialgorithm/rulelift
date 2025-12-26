import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.tree import export_graphviz
import graphviz

class RuleVisualizer:
    """
    规则可视化类，用于生成各种规则可视化图表
    """
    
    def __init__(self, dpi: int = 300):
        """
        初始化规则可视化器
        
        参数:
            dpi: 图像分辨率，默认为300
        """
        self.dpi = dpi
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_rule_comparison(self, rules_df: pd.DataFrame, metrics: List[str] = ['lift', 'badrate', 'sample_count'],
                           figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制规则比较图
        
        参数:
            rules_df: 包含规则信息的DataFrame
            metrics: 要比较的指标列表，默认为['lift', 'badrate', 'sample_count']
            figsize: 图表大小，默认为(15, 10)
            save_path: 保存路径，如'./rule_comparison.png'，默认为None
            
        返回:
            图表对象
        """
        # 验证规则DataFrame包含必要列
        if 'rule_description' not in rules_df.columns:
            raise ValueError("规则DataFrame必须包含'rule_description'列")
        
        for metric in metrics:
            if metric not in rules_df.columns:
                raise ValueError(f"规则DataFrame必须包含{metric}列")
        
        # 创建子图
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        # 绘制每个指标的条形图
        for i, metric in enumerate(metrics):
            # 修复FutureWarning：将x变量分配给hue
            sns.barplot(x='rule_description', y=metric, hue='rule_description', data=rules_df, ax=axes[i], palette='viridis', dodge=False)
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'Rule Comparison - {metric.capitalize()}')
            # 设置x轴标签旋转和对齐
            axes[i].tick_params(axis='x', rotation=45)
            plt.setp(axes[i].get_xticklabels(), ha='right')
            
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
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_rule_distribution(self, rules_df: pd.DataFrame, metric: str = 'lift',
                             figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制规则分布直方图
        
        参数:
            rules_df: 包含规则信息的DataFrame
            metric: 要绘制分布的指标，默认为'lift'
            figsize: 图表大小，默认为(10, 6)
            save_path: 保存路径，如'./rule_distribution.png'，默认为None
            
        返回:
            图表对象
        """
        # 验证规则DataFrame包含必要列
        if metric not in rules_df.columns:
            raise ValueError(f"规则DataFrame必须包含{metric}列")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制直方图
        sns.histplot(rules_df[metric], kde=True, ax=ax, bins=20, color='skyblue', edgecolor='black')
        
        # 添加统计信息
        mean_val = rules_df[metric].mean()
        median_val = rules_df[metric].median()
        std_val = rules_df[metric].std()
        
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.4f}')
        
        ax.set_title(f'Distribution of Rule {metric.capitalize()}')
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_lift_precision_scatter(self, rules_df: pd.DataFrame, size_col: str = 'sample_count',
                                  figsize: Tuple[int, int] = (10, 8), save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制Lift-Precision散点图
        
        参数:
            rules_df: 包含规则信息的DataFrame
            size_col: 用于点大小的列，默认为'sample_count'
            figsize: 图表大小，默认为(10, 8)
            save_path: 保存路径，如'./lift_precision_scatter.png'，默认为None
            
        返回:
            图表对象
        """
        # 验证规则DataFrame包含必要列
        for col in ['lift', 'precision', size_col]:
            if col not in rules_df.columns:
                raise ValueError(f"规则DataFrame必须包含{col}列")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制散点图
        scatter = ax.scatter(rules_df['precision'], rules_df['lift'], 
                           s=rules_df[size_col] / rules_df[size_col].max() * 1000, 
                           alpha=0.7, c=rules_df['badrate'], cmap='viridis')
        
        # 添加颜色条
        cbar = fig.colorbar(scatter)
        cbar.set_label('Badrate')
        
        ax.set_title('Lift vs Precision Scatter Plot')
        ax.set_xlabel('Precision')
        ax.set_ylabel('Lift')
        ax.grid(True, alpha=0.3)
        
        # 添加平均线
        ax.axhline(rules_df['lift'].mean(), color='red', linestyle='--', alpha=0.5)
        ax.axvline(rules_df['precision'].mean(), color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_decision_tree(self, model: any, feature_cols: List[str], class_names: List[str] = ['Good', 'Bad'],
                         max_depth: int = 5, figsize: Tuple[int, int] = (20, 15), save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制决策树
        
        参数:
            model: 训练好的决策树模型
            feature_cols: 特征列名列表
            class_names: 类别名称列表，默认为['Good', 'Bad']
            max_depth: 绘制的最大深度，默认为5
            figsize: 图表大小，默认为(20, 15)
            save_path: 保存路径，如'./decision_tree.png'，默认为None
            
        返回:
            图表对象
        """
        # 使用graphviz绘制决策树
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_cols,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth
        )
        
        # 生成图表
        graph = graphviz.Source(dot_data)
        
        # 保存图表
        if save_path:
            graph.render(filename=save_path.replace('.png', ''), format='png', cleanup=True)
        
        return graph
    
    def plot_rule_network(self, rules: List[Dict[str, any]], figsize: Tuple[int, int] = (15, 15),
                        save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制规则网络图
        
        参数:
            rules: 包含规则信息的字典列表
            figsize: 图表大小，默认为(15, 15)
            save_path: 保存路径，如'./rule_network.png'，默认为None
            
        返回:
            图表对象
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for i, rule in enumerate(rules):
            # 计算节点大小，基于样本数和重要性
            node_size = rule.get('samples', 1) * rule.get('importance', 1)
            node_color = rule.get('badrate', 0)  # 使用坏账率作为节点颜色
            
            G.add_node(f'Rule {i+1}', 
                      size=node_size,
                      color=node_color,
                      lift=rule.get('lift', 1),
                      badrate=rule.get('badrate', 0),
                      precision=rule.get('precision', 0))
        
        # 添加边（这里简单模拟，实际应根据规则间的关联关系）
        for i in range(len(rules)):
            for j in range(i+1, len(rules)):
                # 简单模拟边权重，基于规则lift值的差异
                weight = abs(rules[i].get('lift', 1) - rules[j].get('lift', 1))
                if weight > 0.5:  # 只添加权重较大的边
                    G.add_edge(f'Rule {i+1}', f'Rule {j+1}', weight=weight)
        
        # 绘制网络图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算节点位置
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # 获取节点大小和颜色
        node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, 
                             cmap='viridis', alpha=0.8)
        
        # 绘制边
        edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5, arrowsize=10)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Badrate')
        
        ax.set_title('Rule Network Visualization')
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(self, data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None) -> plt.Axes:
        """
        绘制热力图
        
        参数:
            data: 要绘制热力图的数据，通常是相关性矩阵或交叉表
            figsize: 图表大小，默认为(12, 10)
            save_path: 保存路径，如'./heatmap.png'，默认为None
            
        返回:
            图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlBu', ax=ax, square=True)
        
        ax.set_title('Correlation Heatmap')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def export_rules(self, rules_df: pd.DataFrame, export_path: str, export_format: str = 'csv'):
        """
        导出规则信息
        
        参数:
            rules_df: 包含规则信息的DataFrame
            export_path: 导出路径，如'./rules'
            export_format: 导出格式，支持'csv'、'json'、'excel'，默认为'csv'
        """
        if export_format == 'csv':
            rules_df.to_csv(f'{export_path}.csv', index=False, encoding='utf-8-sig')
        elif export_format == 'json':
            rules_df.to_json(f'{export_path}.json', orient='records', force_ascii=False, indent=2)
        elif export_format == 'excel':
            rules_df.to_excel(f'{export_path}.xlsx', index=False, engine='openpyxl')
        else:
            raise ValueError(f"不支持的导出格式: {export_format}，支持的格式有: 'csv', 'json', 'excel'")
    
    def generate_rule_report(self, rules_df: pd.DataFrame, report_path: str = './rule_report'):
        """
        生成规则报告，包含多种可视化图表
        
        参数:
            rules_df: 包含规则信息的DataFrame
            report_path: 报告保存路径，默认为'./rule_report'
        """
        # 生成规则比较图
        self.plot_rule_comparison(rules_df, save_path=f'{report_path}_comparison.png')
        
        # 生成规则分布图
        self.plot_rule_distribution(rules_df, save_path=f'{report_path}_distribution.png')
        
        # 如果包含lift和precision列，生成散点图
        if 'lift' in rules_df.columns and 'precision' in rules_df.columns:
            self.plot_lift_precision_scatter(rules_df, save_path=f'{report_path}_lift_precision.png')
        
        # 导出规则数据
        self.export_rules(rules_df, report_path, export_format='csv')
        self.export_rules(rules_df, report_path, export_format='json')
        self.export_rules(rules_df, report_path, export_format='excel')

# 简化的API函数
def plot_rule_comparison(rules_df: pd.DataFrame, metrics: List[str] = ['lift', 'badrate', 'sample_count'],
                        figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制规则比较图的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_rule_comparison(rules_df, metrics, figsize, save_path)

def plot_rule_distribution(rules_df: pd.DataFrame, metric: str = 'lift',
                          figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制规则分布直方图的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_rule_distribution(rules_df, metric, figsize, save_path)

def plot_lift_precision_scatter(rules_df: pd.DataFrame, size_col: str = 'sample_count',
                               figsize: Tuple[int, int] = (10, 8), save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制Lift-Precision散点图的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_lift_precision_scatter(rules_df, size_col, figsize, save_path)

def plot_decision_tree(model: any, feature_cols: List[str], class_names: List[str] = ['Good', 'Bad'],
                      max_depth: int = 5, figsize: Tuple[int, int] = (20, 15), save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制决策树的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_decision_tree(model, feature_cols, class_names, max_depth, figsize, save_path)

def plot_rule_network(rules: List[Dict[str, any]], figsize: Tuple[int, int] = (15, 15),
                     save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制规则网络图的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_rule_network(rules, figsize, save_path)

def plot_heatmap(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None) -> plt.Axes:
    """
    绘制热力图的简化API
    """
    visualizer = RuleVisualizer()
    return visualizer.plot_heatmap(data, figsize, save_path)

def generate_rule_report(rules_df: pd.DataFrame, report_path: str = './rule_report'):
    """
    生成规则报告的简化API
    """
    visualizer = RuleVisualizer()
    visualizer.generate_rule_report(rules_df, report_path)
