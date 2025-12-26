import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DecisionTreeRuleExtractor:
    """
    基于决策树的规则提取类，用于从训练好的决策树模型中自动提取可解释的规则集
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'ISBAD', exclude_cols: List[str] = None, max_depth: int = 5, 
                 min_samples_split: int = 10, min_samples_leaf: int = 5, 
                 test_size: float = 0.2, random_state: int = 42):
        """
        初始化决策树规则提取器
        
        参数:
            df: 输入的数据集
            target_col: 目标字段名，默认为'ISBAD'
            exclude_cols: 排除的字段名列表，默认为None
            max_depth: 决策树最大深度，默认为5
            min_samples_split: 分裂节点所需的最小样本数，默认为10
            min_samples_leaf: 叶子节点的最小样本数，默认为5
            test_size: 测试集比例，默认为0.2
            random_state: 随机种子，默认为42
        """
        self.df = df.copy()
        self.target_col = target_col
        self.exclude_cols = exclude_cols or []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.test_size = test_size
        self.random_state = random_state
        
        # 准备特征和目标变量，排除指定列
        drop_cols = [self.target_col] + self.exclude_cols
        self.X = self.df.drop(columns=drop_cols)
        self.y = self.df[self.target_col]
        
        # 处理类别型特征
        self._encode_categorical_features()
        
        # 初始化决策树模型
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            criterion='gini'  # 使用基尼系数作为分裂标准
        )
        
        # 训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # 规则提取结果
        self.rules = []
        self.rule_importance = {}
    
    def _encode_categorical_features(self):
        """
        对类别型特征进行编码
        """
        self.encoders = {}
        self.X_encoded = self.X.copy()
        
        for col in self.X.columns:
            if not pd.api.types.is_numeric_dtype(self.X[col]):
                le = LabelEncoder()
                # 处理缺失值
                self.X_encoded[col] = self.X[col].fillna('missing')
                self.X_encoded[col] = le.fit_transform(self.X_encoded[col])
                self.encoders[col] = le
    
    def train(self) -> Tuple[float, float]:
        """
        训练决策树模型
        
        返回:
            训练集准确率和测试集准确率
        """
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 计算准确率
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        
        return train_accuracy, test_accuracy
    
    def _extract_rules_from_tree(self, tree: DecisionTreeClassifier, feature_names: List[str], 
                               class_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        从决策树中提取规则
        
        参数:
            tree: 训练好的决策树模型
            feature_names: 特征名称列表
            class_names: 类别名称列表，默认为None
            
        返回:
            包含规则信息的字典列表
        """
        if class_names is None:
            class_names = [str(i) for i in range(tree.n_classes_)]
        
        # 获取决策树的结构
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value
        
        # 遍历树结构，提取规则
        rules = []
        stack = [(0, [])]  # (node_id, path)
        
        while stack:
            node_id, path = stack.pop()
            
            # 如果是叶子节点
            if children_left[node_id] == children_right[node_id]:
                # 计算规则信息
                class_distribution = value[node_id][0]
                predicted_class = class_distribution.argmax()
                class_probability = class_distribution[predicted_class] / class_distribution.sum()
                
                # 构建规则
                rule = {
                    'rule_id': len(rules) + 1,
                    'conditions': path.copy(),
                    'predicted_class': predicted_class,
                    'class_name': class_names[predicted_class],
                    'class_probability': float(class_probability),
                    'sample_count': int(class_distribution.sum()),
                    'class_distribution': {class_names[i]: float(class_distribution[i]) 
                                          for i in range(len(class_names))}
                }
                rules.append(rule)
            else:
                # 非叶子节点，添加左右子节点到栈中
                # 右子节点（大于阈值）
                right_path = path.copy()
                right_path.append({
                    'feature': feature_names[feature[node_id]],
                    'threshold': float(threshold[node_id]),
                    'operator': '>'
                })
                stack.append((children_right[node_id], right_path))
                
                # 左子节点（小于等于阈值）
                left_path = path.copy()
                left_path.append({
                    'feature': feature_names[feature[node_id]],
                    'threshold': float(threshold[node_id]),
                    'operator': '<='
                })
                stack.append((children_left[node_id], left_path))
        
        return rules
    
    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        提取决策树规则
        
        返回:
            包含规则信息的字典列表
        """
        # 确保模型已经训练
        if not hasattr(self.model, 'tree_'):
            self.train()
        
        # 提取规则
        self.rules = self._extract_rules_from_tree(
            self.model,
            feature_names=self.X.columns.tolist(),
            class_names=['good', 'bad'] if self.y.nunique() == 2 else None
        )
        
        # 计算规则重要性
        self._calculate_rule_importance()
        
        # 对规则按重要性排序
        self.rules.sort(key=lambda x: self.rule_importance.get(x['rule_id'], 0), reverse=True)
        
        return self.rules
    
    def _get_rule_desc(self, rule):
        """
        获取规则描述字符串
        
        参数:
            rule: 规则字典
            
        返回:
            规则描述字符串
        """
        conditions = []
        for cond in rule['conditions']:
            if cond['operator'] == '<=':
                conditions.append(f"{cond['feature']} <= {cond['threshold']:.4f}")
            else:
                conditions.append(f"{cond['feature']} > {cond['threshold']:.4f}")
        return " AND ".join(conditions)
    
    def _calculate_rule_importance(self):
        """
        计算规则重要性
        """
        # 计算每个规则的重要性，基于样本数量和置信度
        for rule in self.rules:
            # 重要性 = 样本数量 * 置信度 * 预测为坏样本的权重（如果是二分类问题）
            weight = 1.0
            if rule['predicted_class'] == 1:  # 如果预测为坏样本，增加权重
                weight = 2.0
            
            importance = rule['sample_count'] * rule['class_probability'] * weight
            self.rule_importance[rule['rule_id']] = importance
    
    def get_rules_as_dataframe(self) -> pd.DataFrame:
        """
        将规则转换为DataFrame格式
        
        返回:
            包含规则信息的DataFrame
        """
        if not self.rules:
            self.extract_rules()
        
        # 将规则转换为DataFrame
        rule_data = []
        for rule in self.rules:
            # 构建规则描述
            conditions = []
            for cond in rule['conditions']:
                if cond['operator'] == '<=':
                    conditions.append(f"{cond['feature']} <= {cond['threshold']:.4f}")
                else:
                    conditions.append(f"{cond['feature']} > {cond['threshold']:.4f}")
            
            rule_desc = " AND ".join(conditions)
            
            rule_data.append({
                'rule_id': rule['rule_id'],
                'rule': rule_desc,
                'predicted_class': rule['predicted_class'],
                'class_name': rule['class_name'],
                'class_probability': rule['class_probability'],
                'sample_count': rule['sample_count'],
                'importance': self.rule_importance[rule['rule_id']],
                'class_distribution': str(rule['class_distribution'])
            })
        
        return pd.DataFrame(rule_data)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        返回:
            包含特征重要性的DataFrame
        """
        if not hasattr(self.model, 'feature_importances_'):
            self.train()
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        })
        
        return feature_importance.sort_values(by='importance', ascending=False)
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (12, 6), save_path: str = None):
        """
        绘制特征重要性图
        
        参数:
            figsize: 图表大小
            save_path: 保存路径，如'./feature_importance.png'
            
        返回:
            matplotlib.pyplot对象
        """
        feature_importance = self.get_feature_importance()
        
        plt.figure(figsize=figsize)
        # 修复FutureWarning：将y变量分配给hue并设置legend=False
        sns.barplot(x='importance', y='feature', hue='feature', data=feature_importance, palette='viridis', dodge=False)
        # 仅在图例存在时移除它，避免警告
        legend = plt.gca().get_legend()
        if legend:
            legend.remove()
        plt.title('Feature Importance from Decision Tree')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def evaluate_rules(self, rules_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        评估规则在测试集上的表现
        
        参数:
            rules_df: 规则DataFrame，如果为None则使用当前提取的规则
            
        返回:
            包含规则评估结果的DataFrame
        """
        if rules_df is None:
            rules_df = self.get_rules_as_dataframe()
        
        # 评估每条规则在测试集上的表现
        evaluation_results = []
        
        # 计算整个测试集的坏客户总数，用于计算召回率
        total_test_bads = sum(self.y_test)
        
        for _, rule_row in rules_df.iterrows():
            rule_id = rule_row['rule_id']
            rule_desc = rule_row['rule']
            
            # 初始化掩码为全True
            mask = pd.Series(True, index=self.X_test.index)
            
            # 解析规则描述并应用条件
            conditions = rule_desc.split(' AND ')
            for condition in conditions:
                # 解析条件: feature operator value
                if ' > ' in condition:
                    feature, value = condition.split(' > ')
                    value = float(value)
                    mask = mask & (self.X_test[feature] > value)
                elif ' <= ' in condition:
                    feature, value = condition.split(' <= ')
                    value = float(value)
                    mask = mask & (self.X_test[feature] <= value)
                elif ' < ' in condition:
                    feature, value = condition.split(' < ')
                    value = float(value)
                    mask = mask & (self.X_test[feature] < value)
                elif ' >= ' in condition:
                    feature, value = condition.split(' >= ')
                    value = float(value)
                    mask = mask & (self.X_test[feature] >= value)
            
            # 应用掩码到测试集
            rule_applied = self.X_test[mask]
            
            if len(rule_applied) == 0:
                continue
            
            # 获取对应的真实标签
            y_true = self.y_test[rule_applied.index]
            y_pred = [rule_row['predicted_class']] * len(y_true)
            
            # 计算评估指标
            total = len(y_true)
            bad_count = sum(y_true)  # 拦截用户中的坏客户数
            good_count = total - bad_count  # 拦截用户中的好客户数
            
            true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
            false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
            true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
            false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
            
            # 命中率
            hit_rate = total / len(self.y_test)
            
            # 坏账率
            badrate = bad_count / total if total > 0 else 0
            
            # 精确率 = 真正例 / 预测为正例的数量
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            
            # 召回率 = 真正例 / 测试集总坏客户数 （修正为整个测试集的召回率）
            recall = true_positive / total_test_bads if total_test_bads > 0 else 0
            
            # F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # lift值
            total_badrate = sum(self.y_test) / len(self.y_test) if len(self.y_test) > 0 else 0
            lift = badrate / total_badrate if total_badrate > 0 else 0
            
            evaluation_results.append({
                'rule_id': rule_id,
                'rule': rule_desc,
                'hit_rate': hit_rate,
                'badrate': badrate,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'lift': lift,
                'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative,
                'sample_count': total,
                'bad_count': bad_count,  # 新增：拦截用户中的坏客户数
                'good_count': good_count   # 新增：拦截用户中的好客户数
            })
        
        return pd.DataFrame(evaluation_results)
    
    def print_rules(self, top_n: int = 10):
        """
        打印规则，包含拦截用户数、badrate、lift和召回率等关键指标
        
        参数:
            top_n: 打印的规则数量，默认为10
        """
        if not self.rules:
            self.extract_rules()
        
        # 评估规则，获取关键指标
        evaluation_df = self.evaluate_rules()
        
        # 如果没有评估结果，直接返回
        if evaluation_df.empty:
            print("没有可评估的规则")
            return
        
        # 创建规则描述到规则信息的映射
        rule_desc_to_info = {}
        for rule in self.rules:
            conditions = []
            for cond in rule['conditions']:
                if cond['operator'] == '<=':
                    conditions.append(f"{cond['feature']} <= {cond['threshold']:.4f}")
                else:
                    conditions.append(f"{cond['feature']} > {cond['threshold']:.4f}")
            rule_desc = " AND ".join(conditions)
            rule_desc_to_info[rule_desc] = rule
        
        # 在评估结果中添加预测类别列
        evaluation_df['predicted_class'] = evaluation_df['rule'].apply(lambda x: rule_desc_to_info[x]['predicted_class'])
        
        # 过滤出预测为坏客户的规则（predicted_class=1）
        evaluation_df_bad = evaluation_df[evaluation_df['predicted_class'] == 1]
        
        # 按lift倒排序规则，并过滤掉lift为0的规则
        evaluation_df_sorted = evaluation_df_bad[evaluation_df_bad['lift'] > 0].sort_values(by='lift', ascending=False)
        
        # 获取top_n规则
        top_rules = evaluation_df_sorted.head(top_n)
        
        print(f"=== Top {min(top_n, len(top_rules))} Rules ===\n")
        
        for _, eval_data in top_rules.iterrows():
            rule_desc = eval_data['rule']
            rule_info = rule_desc_to_info.get(rule_desc)
            
            if rule_info:
                print(f"Rule {rule_info['rule_id']} (Importance: {self.rule_importance[rule_info['rule_id']]:.4f}):")
                print(f"  {rule_desc}")
                print(f"  Predicted Class: {rule_info['class_name']} (Probability: {rule_info['class_probability']:.4f})")
                print(f"  Sample Count: {rule_info['sample_count']}")
                print(f"  Class Distribution: {rule_info['class_distribution']}")
                
                # 打印关键指标，去除精确率，加入lift
                print(f"  拦截用户数: {int(eval_data['sample_count'])}")
                print(f"  坏客户数: {int(eval_data['bad_count'])}")
                print(f"  好客户数: {int(eval_data['good_count'])}")
                print(f"  Badrate: {eval_data['badrate']:.4f}")
                print(f"  召回率: {eval_data['recall']:.4f}")
                print(f"  Lift: {eval_data['lift']:.4f}")
            
            print()
    
    def plot_decision_tree(self, save_path: str = 'decision_tree.png') -> None:
        """
        绘制决策树结构并保存到本地文件
        
        参数:
            save_path: 保存路径，默认'decision_tree.png'
        """
        if not hasattr(self.model, 'tree_'):
            self.train()
        
        # 创建DOT数据，直接在DOT数据中设置字体
        dot_data = export_graphviz(
            self.model,
            out_file=None,
            feature_names=self.X.columns.tolist(),
            class_names=['good', 'bad'] if self.y.nunique() == 2 else None,
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=True,
            node_ids=True,
            proportion=True
        )
        
        # 添加全局字体设置到DOT数据
        dot_data_lines = dot_data.split('\n')
        # 替换为带有中文字体设置的graph定义
        dot_data = '\n'.join([
            'digraph Tree {',
            'graph [fontname="SimHei"];',
            'node [fontname="SimHei"];',
            'edge [fontname="SimHei"];'
        ] + dot_data_lines[1:])
        
        # 创建图形对象
        graph = graphviz.Source(dot_data)
        
        # 保存图片到本地，不直接显示
        graph.render(save_path.replace('.png', ''), format='png', cleanup=True)
        print(f"决策树图片已保存到: {save_path}")
    
    def plot_rule_evaluation(self, save_path: str = 'rule_evaluation.png') -> None:
        """
        绘制规则评估结果的可视化图表
        
        参数:
            save_path: 保存路径，默认'rule_evaluation.png'
        """
        # 评估规则
        evaluation_df = self.evaluate_rules()
        
        if evaluation_df.empty:
            print("没有可评估的规则")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 规则拦截用户数
        axes[0, 0].bar(evaluation_df['rule_id'], evaluation_df['sample_count'])
        axes[0, 0].set_title('规则拦截用户数')
        axes[0, 0].set_xlabel('规则ID')
        axes[0, 0].set_ylabel('拦截用户数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 规则badrate
        axes[0, 1].bar(evaluation_df['rule_id'], evaluation_df['badrate'])
        axes[0, 1].set_title('规则Badrate')
        axes[0, 1].set_xlabel('规则ID')
        axes[0, 1].set_ylabel('Badrate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 规则精确率
        axes[1, 0].bar(evaluation_df['rule_id'], evaluation_df['precision'])
        axes[1, 0].set_title('规则精确率')
        axes[1, 0].set_xlabel('规则ID')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 规则召回率
        axes[1, 1].bar(evaluation_df['rule_id'], evaluation_df['recall'])
        axes[1, 1].set_title('规则召回率')
        axes[1, 1].set_xlabel('规则ID')
        axes[1, 1].set_ylabel('召回率')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片到本地，不直接显示
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"规则评估图片已保存到: {save_path}")
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        获取模型性能指标
        
        返回:
            包含模型性能指标的字典
        """
        if not hasattr(self.model, 'tree_'):
            self.train()
        
        # 计算训练集和测试集的性能指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_train_prob = self.model.predict_proba(self.X_train)[:, 1]
        y_test_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        performance = {
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'train_precision': precision_score(self.y_train, y_train_pred),
            'test_precision': precision_score(self.y_test, y_test_pred),
            'train_recall': recall_score(self.y_train, y_train_pred),
            'test_recall': recall_score(self.y_test, y_test_pred),
            'train_f1': f1_score(self.y_train, y_train_pred),
            'test_f1': f1_score(self.y_test, y_test_pred),
            'train_roc_auc': roc_auc_score(self.y_train, y_train_prob),
            'test_roc_auc': roc_auc_score(self.y_test, y_test_prob)
        }
        
        return performance
