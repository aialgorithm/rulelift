import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from scipy.stats import chi2_contingency
from typing import List, Dict, Any, Tuple, Union
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TreeRuleExtractor:
    """
    统一的树模型规则提取类，支持多种算法
    
    支持的算法：
    - 'dt': 决策树（Decision Tree）
    - 'rf': 随机森林（Random Forest）
    - 'chi2': 卡方随机森林（Chi-square Random Forest）- 先对特征进行卡方分箱预处理，然后使用随机森林
    - 'gbdt': 梯度提升树（Gradient Boosting Decision Tree）
    - 'xgb': 已废弃，请使用'gbdt'（向后兼容）
    - 'isf': 孤立森林（Isolation Forest）
    
    注意：chi2算法采用卡方分箱预处理 + 随机森林的方式实现，不需要复杂的自定义决策树分裂逻辑。
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'ISBAD', exclude_cols: List[str] = None,
                 algorithm: str = 'dt', max_depth: int = 3, min_samples_split: int = 10, 
                 min_samples_leaf: Union[int, float] = 5, n_estimators: int = 10, max_features: str = 'sqrt',
                 test_size: float = 0.3, random_state: int = 42,
                 amount_col: str = None, ovd_bal_col: str = None,
                 feature_trends: Dict[str, int] = None,
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 min_bin_ratio: float = 0.05,
                 isf_weights: Dict[str, float] = None):
        """
        初始化树规则提取器
        
        参数:
            df: 输入的数据集
            target_col: 目标字段名，默认为'ISBAD'
            exclude_cols: 排除的字段名列表，默认为None
            algorithm: 算法类型，'dt'、'rf'、'chi2'、'gbdt'、'isf'，默认为'dt'
                     注意：'xgb'已废弃，请使用'gbdt'（向后兼容）
            max_depth: 决策树最大深度，默认为3
            min_samples_split: 分裂节点所需的最小样本数，默认为10
            min_samples_leaf: 叶子节点的最小样本数，默认为5。可以是整数或小数，如果为小于1的小数，则自动转换为训练样本数的整数（如0.1转换为训练样本数的10%）
            n_estimators: 随机森林/GBDT/孤立森林中树的数量，默认为10
            max_features: 每棵树分裂时考虑的最大特征数，'sqrt'或'log2'，默认为'sqrt'
            test_size: 测试集比例，默认为0.3
            random_state: 随机种子，默认为42
            amount_col: 金额字段名，默认为None
            ovd_bal_col: 逾期金额字段名，默认为None
            feature_trends: 特征与目标标签的正负相关性字典，{特征名: 1或-1}，默认为None
                          1表示正相关（特征越大，违约概率越高），只保留大于阈值的规则
                          -1表示负相关（特征越小，违约概率越高），只保留小于等于阈值的规则
            learning_rate: GBDT学习率，默认为0.1
            subsample: GBDT子采样比例，默认为1.0
            min_bin_ratio: 卡方分箱的最小样本占比，默认为0.05（仅适用于chi2算法）
            isf_weights: 孤立森林规则权重的字典，默认为None。
                        可选键: 'purity' (坏客户纯度), 'anomaly' (异常分数), 'sample' (样本数量), 'hit' (异常坏客户命中比例)
        """
        if df is None or df.empty:
            raise ValueError("输入的数据集不能为空")
            
        self.df = df.copy(deep=False).reset_index(drop=True)
        self.target_col = target_col
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"目标字段 '{self.target_col}' 不在数据集中")
            
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.algorithm = algorithm
        if self.algorithm == 'xgb':
            self.algorithm = 'gbdt'
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.amount_col = amount_col
        self.ovd_bal_col = ovd_bal_col
        self.feature_trends = feature_trends
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_bin_ratio = min_bin_ratio
        
        # 初始化孤立森林权重
        default_isf_weights = {
            'purity': 0.5,
            'anomaly': 0.3,
            'sample': 0.15,
            'hit': 0.05
        }
        if isf_weights:
            default_isf_weights.update(isf_weights)
        self.isf_weights = default_isf_weights
        
        # 向后兼容性处理：'xgb' -> 'gbdt'
        self.algorithm = self.algorithm.lower()
        if self.algorithm == 'xgb':
            warnings.warn(
                "算法'xgb'已废弃，请使用'gbdt'。'xgb'将在未来版本中移除。",
                DeprecationWarning,
                stacklevel=2
            )
            self.algorithm = 'gbdt'
        
        # 验证算法类型
        valid_algorithms = ['dt', 'rf', 'chi2', 'gbdt', 'isf']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{self.algorithm}'. Must be one of {valid_algorithms}")
        
        # 准备特征和目标变量，排除指定列
        drop_cols = [self.target_col] + self.exclude_cols
        if self.amount_col:
            drop_cols.append(self.amount_col)
        if self.ovd_bal_col:
            drop_cols.append(self.ovd_bal_col)
        self.X = self.df.drop(columns=drop_cols)
        self.y = self.df[self.target_col]
        
        # 处理类别型特征
        self._encode_categorical_features()
        
        # 处理NaN值
        self.X = self.X.fillna(0)
        
        # 初始化模型
        self.model = self._initialize_model()
        
        # 训练集和测试集
        if self.algorithm != 'isf':
            # 孤立森林不需要划分训练集和测试集
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_encoded, self.y, test_size=test_size, random_state=random_state
            )
            
            # 处理min_samples_leaf小数参数
            if isinstance(self.min_samples_leaf, float) and self.min_samples_leaf < 1:
                original_min_samples_leaf = self.min_samples_leaf
                self.min_samples_leaf = int(len(self.X_train) * self.min_samples_leaf)
                print(f"min_samples_leaf参数为小数{original_min_samples_leaf:.2%}，自动转换为训练样本数的整数: {self.min_samples_leaf}")
            
            # 对chi2算法进行卡方分箱预处理
            if self.algorithm == 'chi2':
                print("正在进行卡方分箱预处理...")
                self.X_train = self._preprocess_features_chi2(self.X_train, self.y_train)
                # 保存训练集的分箱边界
                self.chi2_bin_edges = {}
                for col in self.X_train.columns:
                    if pd.api.types.is_numeric_dtype(self.X[col]):
                        # 对训练集进行分箱，获取分箱边界
                        feature_values = self.X_train[col].dropna()
                        target_values = self.y_train.loc[feature_values.index]
                        
                        # 初始分箱：使用等频分箱
                        max_bins = min(50, len(feature_values.unique()))
                        if max_bins >= 2:
                            discretizer = KBinsDiscretizer(n_bins=max_bins, encode='ordinal', strategy='quantile')
                            bins = discretizer.fit_transform(feature_values.values.reshape(-1, 1)).flatten()
                            bin_edges_col = discretizer.bin_edges_[0]
                            
                            # 使用卡方检验合并相似分箱
                            while len(bin_edges_col) > 2:
                                chi2_scores = []
                                for i in range(len(bin_edges_col) - 2):
                                    merged_bins = bins.copy()
                                    merged_bins[merged_bins == i+1] = i
                                    
                                    try:
                                        contingency_table = pd.crosstab(merged_bins, target_values)
                                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                                        chi2_scores.append((i, chi2_stat))
                                    except:
                                        chi2_scores.append((i, 0))
                                
                                if chi2_scores:
                                    min_chi2_idx, min_chi2_score = min(chi2_scores, key=lambda x: x[1])
                                    
                                    bin_counts = pd.Series(bins).value_counts().sort_index()
                                    total_samples = len(bins)
                                    min_required_samples = total_samples * self.min_bin_ratio
                                    
                                    if min_chi2_score < 3.841 and bin_counts.iloc[min_chi2_idx] >= min_required_samples:
                                        bins[bins > min_chi2_idx] -= 1
                                        bin_edges_col = np.delete(bin_edges_col, min_chi2_idx + 1)
                                    else:
                                        break
                                else:
                                    break
                            
                            # 保存分箱边界
                            self.chi2_bin_edges[col] = bin_edges_col
                
                self.X_test = self._preprocess_features_chi2(self.X_test, self.y_test, bin_edges=self.chi2_bin_edges)
                print(f"卡方分箱预处理完成")
        else:
            self.X_train = self.X_encoded
            self.y_train = self.y
            self.X_test = None
            self.y_test = None
            
            # 处理min_samples_leaf小数参数（孤立森林）
            if isinstance(self.min_samples_leaf, float) and self.min_samples_leaf < 1:
                original_min_samples_leaf = self.min_samples_leaf
                self.min_samples_leaf = int(len(self.X_train) * self.min_samples_leaf)
                print(f"min_samples_leaf参数为小数{original_min_samples_leaf:.2%}，自动转换为训练样本数的整数: {self.min_samples_leaf}")
        
        # 规则提取结果
        self.rules = []
        self.all_rules = []
        self.rule_importance = {}
    
    def _encode_categorical_features(self):
        """
        对类别型特征进行编码
        """
        self.encoders = {}
        self.X_encoded = self.X.copy(deep=False)
        
        # 处理NaN值
        self.X_encoded = self.X_encoded.fillna(0)
        
        for col in self.X.columns:
            if not pd.api.types.is_numeric_dtype(self.X[col]):
                le = LabelEncoder()
                # 处理缺失值
                self.X_encoded[col] = self.X[col].fillna('missing')
                self.X_encoded[col] = le.fit_transform(self.X_encoded[col])
                self.encoders[col] = le
        
        # 更新X为编码后的数据
        self.X = self.X_encoded
    
    def _preprocess_features_chi2(self, X: pd.DataFrame, y: pd.Series, bin_edges: Dict[str, np.ndarray] = None) -> pd.DataFrame:
        """
        使用卡方检验对特征进行分箱预处理
        
        参数:
            X: 特征数据
            y: 目标变量
            bin_edges: 分箱边界字典（可选），如果提供则直接使用训练集的分箱边界
            
        返回:
            分箱后的特征数据（原始特征值替换为卡方分箱的上限）
        """
        X_binned = X.copy(deep=False)
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # 跳过非数值型特征
                continue
            
            feature_values = X[col].dropna()
            target_values = y.loc[feature_values.index]
            
            # 如果提供了训练集的分箱边界，直接使用
            if bin_edges and col in bin_edges:
                bin_edges_col = bin_edges[col]
                binned = pd.cut(feature_values, bins=bin_edges_col, right=True)
            else:
                # 初始分箱：使用等频分箱
                max_bins = min(10, len(feature_values.unique()))
                if max_bins < 2:
                    continue
                
                discretizer = KBinsDiscretizer(n_bins=max_bins, encode='ordinal', strategy='quantile')
                bins = discretizer.fit_transform(feature_values.values.reshape(-1, 1)).flatten()
                
                # 获取分箱边界
                bin_edges_col = discretizer.bin_edges_[0]
                
                # 使用卡方检验合并相似分箱
                while len(bin_edges_col) > 2:
                    # 计算相邻分箱的卡方分数
                    chi2_scores = []
                    for i in range(len(bin_edges_col) - 2):
                        # 合并分箱i和i+1
                        merged_bins = bins.copy()
                        merged_bins[merged_bins == i+1] = i
                        
                        # 创建列联表
                        try:
                            contingency_table = pd.crosstab(merged_bins, target_values)
                            
                            # 计算卡方检验
                            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                            chi2_scores.append((i, chi2_stat))
                        except:
                            chi2_scores.append((i, 0))
                    
                    # 找到卡方分数最小的相邻分箱对（最相似的分箱）
                    if chi2_scores:
                        min_chi2_idx, min_chi2_score = min(chi2_scores, key=lambda x: x[1])
                        
                        # 检查是否满足最小样本占比
                        bin_counts = pd.Series(bins).value_counts().sort_index()
                        total_samples = len(bins)
                        min_required_samples = total_samples * self.min_bin_ratio
                        
                        # 如果最小分箱的样本数满足要求，且卡方分数低于阈值，则合并分箱
                        if min_chi2_score < 3.841 and bin_counts.iloc[min_chi2_idx] >= min_required_samples:
                            # 合并分箱
                            bins[bins > min_chi2_idx] -= 1
                            # 移除分箱边界
                            bin_edges_col = np.delete(bin_edges_col, min_chi2_idx + 1)
                        else:
                            break
                    else:
                        break
            
            # 使用最终的分箱边界创建分箱区间
            binned = pd.cut(feature_values, bins=bin_edges_col, right=True)
            
            # 将原始特征值替换为卡方分箱的上限
            if hasattr(binned, 'cat'):
                # 从IntervalIndex获取上限
                bin_upper_bounds = [interval.right for interval in binned.cat.categories]
                bin_mapping = {interval.right: interval.right for interval in binned.cat.categories}
                X_binned[col] = binned.apply(lambda x: bin_mapping.get(x.right, x.right) if pd.notna(x) else np.nan)
            else:
                # 如果没有categories属性，使用原始值
                X_binned[col] = X[col]
        
        return X_binned
    
    def _initialize_model(self):
        """
        根据算法类型初始化模型
        """
        if self.algorithm == 'dt':
            # 决策树
            model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                criterion='gini'
            )
        elif self.algorithm == 'rf':
            # 随机森林
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=True
            )
        elif self.algorithm == 'chi2':
            # 卡方随机森林（先对特征进行卡方分箱预处理，然后使用随机森林）
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=True
            )
        elif self.algorithm == 'gbdt':
            # 梯度提升树（GBDT）
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                learning_rate=self.learning_rate,
                subsample=self.subsample
            )
        elif self.algorithm == 'isf':
            # 孤立森林（用于异常检测）
            # 调整contamination参数，从0.1降到0.05，增加异常样本数量
            model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=min(512, len(self.X_encoded)),  # 从256增加到512
                contamination=0.05,  # 从0.1降到0.05
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return model
    
    def train(self) -> Tuple[float, float]:
        """
        训练模型
        
        返回:
            训练集准确率和测试集准确率（对于孤立森林，返回异常分数统计）
        """
        try:
            if self.algorithm == 'isf':
                # 孤立森林训练
                self.model.fit(self.X_train)
                # 计算异常分数
                anomaly_scores = self.model.score_samples(self.X_train)
                # 返回统计信息
                mean_score = np.mean(anomaly_scores)
                std_score = np.std(anomaly_scores)
                print(f"孤立森林训练完成")
                print(f"  平均异常分数: {mean_score:.4f}")
                print(f"  标准差: {std_score:.4f}")
                return mean_score, std_score
            else:
                # 其他算法训练
                self.model.fit(self.X_train, self.y_train)
                
                # 计算准确率
                train_accuracy = self.model.score(self.X_train, self.y_train)
                test_accuracy = self.model.score(self.X_test, self.y_test)
                
                return train_accuracy, test_accuracy
        except Exception as e:
            print(f"训练模型时发生错误: {str(e)}")
            raise
    
    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        提取规则
        
        返回:
            规则列表
        """
        try:
            if self.algorithm == 'dt':
                self.rules = self._extract_rules_from_tree(self.model, tree_id=0)
            elif self.algorithm == 'rf':
                self.all_rules = self._extract_rules_from_forest()
            elif self.algorithm == 'chi2':
                # chi2算法使用的是RandomForestClassifier，应该调用_extract_rules_from_forest
                self.all_rules = self._extract_rules_from_forest()
            elif self.algorithm == 'gbdt':
                # GBDT：从所有树中提取规则
                self.all_rules = self._extract_rules_from_gbdt()
            elif self.algorithm == 'isf':
                self.rules = self._extract_rules_from_isolation_forest()
            
            # 计算规则重要性
            if self.algorithm in ['dt']:
                self.rule_importance = {rule['rule_id']: self._calculate_rule_importance(rule) 
                                      for rule in self.rules}
            elif self.algorithm in ['rf', 'chi2', 'gbdt']:
                for rule in self.all_rules:
                    rule['importance'] = self._calculate_rule_importance(rule)
            elif self.algorithm == 'isf':
                for rule in self.rules:
                    rule['importance'] = self._calculate_rule_importance(rule)
            
            return self.rules if self.algorithm not in ['rf', 'chi2', 'gbdt'] else self.all_rules
        except Exception as e:
            print(f"提取规则时发生错误: {str(e)}")
            raise
    
    def _extract_rules_from_tree(self, tree_model, tree_id: int = 0) -> List[Dict[str, Any]]:
        """
        从单棵决策树中提取规则
        
        参数:
            tree_model: 决策树模型
            tree_id: 树的ID
            
        返回:
            规则列表
        """
        tree_ = tree_model.tree_
        feature_names = self.X.columns.tolist()
        
        rules = []
        
        def recurse(node, current_conditions):
            if tree_.feature[node] != -2:  # 不是叶子节点
                feature_name = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]
                
                # 左子树（<=）
                left_conditions = current_conditions + [{
                    'feature': feature_name,
                    'threshold': threshold,
                    'operator': '<='
                }]
                recurse(tree_.children_left[node], left_conditions)
                
                # 右子树（>）
                right_conditions = current_conditions + [{
                    'feature': feature_name,
                    'threshold': threshold,
                    'operator': '>'
                }]
                recurse(tree_.children_right[node], right_conditions)
            else:  # 叶子节点
                # 获取类别分布
                value = tree_.value[node][0]
                total_samples = value.sum()
                
                # 计算该叶子节点路径上样本的实际坏客户比例
                badrate = self._calculate_leaf_node_badrate(current_conditions)
                
                class_distribution = {
                    'good': 1.0 - badrate,
                    'bad': badrate
                }
                
                # 对于GBDT回归树，使用坏客户比例作为判断依据
                predicted_class = 1 if badrate > 0.5 else 0
                class_name = 'good' if predicted_class == 0 else 'bad'
                class_probability = badrate
                
                # 创建规则
                rule = {
                    'rule_id': len(rules),
                    'conditions': current_conditions,
                    'predicted_class': predicted_class,
                    'class_name': class_name,
                    'class_probability': class_probability,
                    'sample_count': int(total_samples),
                    'class_distribution': class_distribution,
                    'tree_id': tree_id,
                    'prediction_value': None
                }
                
                # 根据feature_trends过滤规则条件
                if self.feature_trends:
                    filtered_conditions = []
                    for condition in current_conditions:
                        feature = condition['feature']
                        operator = condition['operator']
                        
                        # 如果该特征在feature_trends中定义
                        if feature in self.feature_trends:
                            trend = self.feature_trends[feature]
                            
                             # 正相关：只保留大于（>）的条件
                            if trend == 1 and (operator == '<=' or operator == '<'):
                                continue  # 剔除不符合的条件
                            
                            # 负相关：只保留小于等于（<=）的条件
                            if trend == -1 and (operator == '>' or operator == '>='):
                                continue  # 剔除不符合的条件
                        
                        filtered_conditions.append(condition)
                    
                    # 如果过滤后没有条件，跳过该规则
                    if len(filtered_conditions) == 0:
                        return
                    
                    # 使用过滤后的条件生成规则
                    rule['conditions'] = filtered_conditions
                
                rules.append(rule)
        
        recurse(0, [])
        return rules
    
    def _calculate_leaf_node_badrate(self, conditions: List[Dict[str, Any]]) -> float:
        """
        计算叶子节点路径上样本的坏客户比例
        
        参数:
            conditions: 规则条件列表
            
        返回:
            坏客户比例
        """
        mask = np.ones(len(self.X_train), dtype=bool)
        for condition in conditions:
            feature = condition['feature']
            threshold = condition['threshold']
            operator = condition['operator']
            
            # 处理category类型（卡方分箱后的特征）
            feature_values = self.X_train[feature]
            if pd.api.types.is_categorical_dtype(feature_values):
                # 将category类型转换为float类型
                feature_values = feature_values.astype(float)
            
            if operator == '<=':
                mask &= (feature_values <= threshold)
            elif operator == '>':
                mask &= (feature_values > threshold)
            elif operator == '==':
                mask &= (feature_values == threshold)
        
        hit_count = mask.sum()
        if hit_count == 0:
            return 0.0
        
        hit_bad = self.y_train[mask].sum()
        badrate = hit_bad / hit_count if hit_count > 0 else 0
        
        return badrate
    
    def _filter_rules_by_feature_trends(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据特征趋势过滤规则
        
        参数:
            rules: 规则列表
        
        返回:
            过滤后的规则列表
        """
        if not self.feature_trends:
            return rules
        
        filtered_rules = []
        for rule in rules:
            # 检查规则中的所有条件是否符合feature_trends
            valid_rule = True
            for condition in rule['conditions']:
                feature = condition['feature']
                operator = condition['operator']
                
                if feature in self.feature_trends:
                    trend = self.feature_trends[feature]
                    
                    # 正相关：只保留大于（>）的条件
                    if trend == 1 and (operator == '<=' or operator == '<'):
                        valid_rule = False
                        break
                    
                    # 负相关：只保留小于等于（<=）的条件
                    if trend == -1 and (operator == '>' or operator == '>='):
                        valid_rule = False
                        break
            
            if valid_rule:
                filtered_rules.append(rule)
        
        return filtered_rules
    
    def _extract_rules_from_forest(self) -> List[Dict[str, Any]]:
        """
        从随机森林中提取规则
        
        返回:
            规则列表
        """
        all_rules = []
        
        for i, tree in enumerate(self.model.estimators_):
            tree_rules = self._extract_rules_from_tree(tree, tree_id=i)
            all_rules.extend(tree_rules)
        
        # 根据feature_trends过滤规则
        if self.feature_trends:
            all_rules = self._filter_rules_by_feature_trends(all_rules)
        
        return all_rules
    
    def _extract_rules_from_gbdt(self) -> List[Dict[str, Any]]:
        """
        从梯度提升树（GBDT）中提取规则
        
        返回:
            规则列表
        """
        all_rules = []
        
        # GradientBoostingClassifier.estimators_是形状为(n_estimators, 1)的numpy数组
        # 每个元素是一个DecisionTreeRegressor对象
        for i in range(self.model.n_estimators_):
            tree = self.model.estimators_[i, 0]
            tree_rules = self._extract_rules_from_tree(tree, tree_id=i)
            all_rules.extend(tree_rules)
        
        print(f"   从GBDT中提取的原始规则数量: {len(all_rules)}")
        
        # 规则筛选：过滤掉在训练集上命中样本过少的规则
        filtered_rules = []
        for rule in all_rules:
            # 计算该规则在训练集上的表现
            mask = np.ones(len(self.X_train), dtype=bool)
            for condition in rule['conditions']:
                feature = condition['feature']
                threshold = condition['threshold']
                operator = condition['operator']
                
                # 处理category类型（卡方分箱后的特征）
                feature_values = self.X_train[feature]
                if pd.api.types.is_categorical_dtype(feature_values):
                    # 将category类型转换为float类型
                    feature_values = feature_values.astype(float)
                
                if operator == '<=':
                    mask &= (feature_values <= threshold)
                elif operator == '>':
                    mask &= (feature_values > threshold)
                elif operator == '==':
                    mask &= (feature_values == threshold)
            
            hit_count = mask.sum()
            
            # 过滤掉命中样本过少的规则（至少命中5个样本，从10降低到5）
            if hit_count < 5:
                continue
            
            # 计算坏客户纯度
            hit_bad = self.y_train[mask].sum()
            badrate = hit_bad / hit_count if hit_count > 0 else 0
            
            # 降低坏客户纯度要求（从0.05降到0.01），增加规则数量
            if badrate < 0.01:
                continue
            
            # 更新规则的样本数量和坏客户纯度
            rule['sample_count'] = int(hit_count)
            rule['class_probability'] = badrate
            
            filtered_rules.append(rule)
        
        print(f"   过滤后的规则数量: {len(filtered_rules)}")
        
        # 根据feature_trends过滤规则
        if self.feature_trends:
            filtered_rules = self._filter_rules_by_feature_trends(filtered_rules)
            print(f"   特征趋势过滤后的规则数量: {len(filtered_rules)}")
        
        # 规则去重：基于规则条件进行去重
        unique_rules = []
        seen_rules = set()
        
        for rule in filtered_rules:
            # 生成规则条件的唯一标识
            conditions_str = '|'.join([
                f"{cond['feature']}{cond['operator']}{cond['threshold']:.4f}"
                for cond in sorted(rule['conditions'], key=lambda x: x['feature'])
            ])
            
            if conditions_str not in seen_rules:
                seen_rules.add(conditions_str)
                unique_rules.append(rule)
        
        print(f"   去重后的规则数量: {len(unique_rules)}")
        
        # 计算规则重要性
        for rule in unique_rules:
            rule['importance'] = self._calculate_rule_importance(rule)
        
        # 按重要性排序
        unique_rules.sort(key=lambda x: x['importance'], reverse=True)
        
        # 保留top 100条规则（增加规则数量）
        return unique_rules[:100]
    
    def _extract_rules_from_isolation_forest(self) -> List[Dict[str, Any]]:
        """
        从孤立森林中提取规则（直接从树结构递归提取规则，限制深度3）
        
        返回:
            规则列表
        """
        # 计算异常分数
        anomaly_scores = self.model.score_samples(self.X_train)
        
        # 使用contamination超参数来确定异常阈值（设置宽松些，默认0.1）
        contamination = 0.3  # 从0.05提高到0.3，增加异常样本数量
        anomaly_threshold = np.percentile(anomaly_scores, int(contamination * 100))
        
        # 直接筛选出异常且为坏客户（ISBAD=1）的样本
        bad_anomaly_mask = (anomaly_scores < anomaly_threshold) & (self.y_train == 1)
        bad_anomaly_indices = np.where(bad_anomaly_mask)[0]
        bad_anomaly_scores = anomaly_scores[bad_anomaly_indices]
        
        # 如果没有异常的坏客户样本，返回空规则列表
        if len(bad_anomaly_indices) == 0:
            return []
        
        # 孤立森林的树结构
        feature_names = self.X.columns.tolist()
        rules = []
        
        # 递归遍历孤立森林中的每棵树，提取从根节点到叶子节点的路径作为规则
        for tree_id, estimator in enumerate(self.model.estimators_):
            tree = estimator.tree_
            
            def extract_rules_from_tree_node(node_id, current_conditions, current_depth):
                """
                递归提取孤立森林树的规则
                
                参数:
                    node_id: 当前节点ID
                    current_conditions: 当前路径的条件列表
                    current_depth: 当前深度
                """
                # 如果深度超过3，停止递归
                if current_depth > 3:
                    return []
                
                # 如果是叶子节点
                if tree.feature[node_id] == -2:
                    # 计算该规则在训练集上的表现
                    mask = np.ones(len(self.X_train), dtype=bool)
                    for condition in current_conditions:
                        feature = condition['feature']
                        threshold = condition['threshold']
                        operator = condition['operator']
                        
                        if operator == '<=':
                            mask &= (self.X_train[feature] <= threshold)
                        elif operator == '>':
                            mask &= (self.X_train[feature] > threshold)
                    
                    hit_count = mask.sum()
                    
                    # 降低过滤阈值（从5降到1），允许命中1个样本的规则
                    if hit_count < 1:
                        return []
                    
                    hit_bad = self.y_train[mask].sum()
                    hit_good = hit_count - hit_bad
                    
                    # 计算坏客户纯度
                    badrate = hit_bad / hit_count if hit_count > 0 else 0
                    

                    
                    # 计算该规则命中的异常坏客户比例
                    mask_bad_anomaly = np.zeros(len(bad_anomaly_indices), dtype=bool)
                    for i, idx in enumerate(bad_anomaly_indices):
                        match = True
                        for condition in current_conditions:
                            feature = condition['feature']
                            threshold = condition['threshold']
                            operator = condition['operator']
                            
                            if operator == '<=':
                                match &= (self.X_train.iloc[idx][feature] <= threshold)
                            elif operator == '>':
                                match &= (self.X_train.iloc[idx][feature] > threshold)
                        
                        mask_bad_anomaly[i] = match
                    
                    bad_anomaly_hit_ratio = mask_bad_anomaly.sum() / len(bad_anomaly_indices) if len(bad_anomaly_indices) > 0 else 0
                    
                    # 计算平均异常分数
                    avg_anomaly_score = bad_anomaly_scores[mask_bad_anomaly].mean() if mask_bad_anomaly.sum() > 0 else 0
                    
                    # 创建规则
                    rule = {
                        'rule_id': len(rules),
                        'conditions': current_conditions,
                        'predicted_class': 1,  # 预测为坏客户
                        'class_name': 'bad',
                        'class_probability': badrate,
                        'sample_count': int(hit_count),
                        'class_distribution': {
                            'good': hit_good / hit_count if hit_count > 0 else 0,
                            'bad': badrate
                        },
                        'tree_id': tree_id,
                        'anomaly_score': avg_anomaly_score,
                        'bad_anomaly_hit_ratio': bad_anomaly_hit_ratio
                    }
                    
                    return [rule]
                else:
                    # 非叶子节点，递归遍历左右子树
                    feature_name = feature_names[tree.feature[node_id]]
                    threshold = tree.threshold[node_id]
                    
                    # 左子树（<=）
                    left_conditions = current_conditions + [{
                        'feature': feature_name,
                        'threshold': threshold,
                        'operator': '<='
                    }]
                    left_rules = extract_rules_from_tree_node(
                        tree.children_left[node_id], 
                        left_conditions, 
                        current_depth + 1
                    )
                    
                    # 右子树（>）
                    right_conditions = current_conditions + [{
                        'feature': feature_name,
                        'threshold': threshold,
                        'operator': '>'
                    }]
                    right_rules = extract_rules_from_tree_node(
                        tree.children_right[node_id], 
                        right_conditions, 
                        current_depth + 1
                    )
                    
                    return left_rules + right_rules
            
            # 从根节点开始递归提取规则
            tree_rules = extract_rules_from_tree_node(0, [], 0)
            rules.extend(tree_rules)
        
        # 按综合重要性排序：优先异常分数更高且纯度更高的规则
        # 使用配置的权重计算最终得分
        rules.sort(key=lambda x: (
            x['class_probability'] * self.isf_weights['purity'] +
            (1 - x['anomaly_score']) * self.isf_weights['anomaly'] +
            min(x['sample_count'] / 100, 1.0) * self.isf_weights['sample'] +
            x['bad_anomaly_hit_ratio'] * self.isf_weights['hit']
        ), reverse=True)
        
        # 根据feature_trends过滤规则
        if self.feature_trends:
            # 统计使用大于（>）条件的规则数量
            gt_rules = [rule for rule in rules if any(cond['operator'] == '>' for cond in rule['conditions'])]
            
            rules = self._filter_rules_by_feature_trends(rules)
        
        # 保留top 200条规则
        return rules[:200]
    
    def _calculate_rule_importance(self, rule: Dict[str, Any]) -> float:
        """
        计算规则重要性
        
        参数:
            rule: 规则字典
            
        返回:
            重要性分数
        """
        sample_count = rule['sample_count']
        class_probability = rule['class_probability']
        
        # 权重：预测为坏样本时权重更高
        if rule['class_name'] == 'bad':
            weight = 2.0
        elif rule['class_name'] == 'anomaly':
            weight = 1.5
        else:
            weight = 1.0
        
        importance = sample_count * class_probability * weight
        return importance
    
    def evaluate_rules(self) -> pd.DataFrame:
        """
        评估规则在测试集上的表现
        
        返回:
            包含评估结果的DataFrame
        """
        if self.algorithm == 'isf':
            raise ValueError("孤立森林不支持规则评估功能")
        
        if self.algorithm in ['rf', 'gbdt', 'chi2']:
            rules_to_evaluate = self.all_rules
        else:
            rules_to_evaluate = self.rules
        
        if not rules_to_evaluate:
            print("没有规则可以评估")
            return pd.DataFrame()

        print(f"   评估的规则数量: {len(rules_to_evaluate)}")
        results = []
        
        # 训练集总坏样本数
        total_bad_train = self.y_train.sum()
        # 测试集总坏样本数
        total_bad_test = self.y_test.sum()
        
        # 计算整体损失率（如果有amount_col和ovd_bal_col）
        overall_loss_rate_train = 0.0
        overall_loss_rate_test = 0.0
        
        if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
            # 训练集整体损失率（仅删除amount和ovd_bal的缺失值）
            train_df = self.df.iloc[self.X_train.index][[self.amount_col, self.ovd_bal_col, self.target_col]].dropna(subset=[self.amount_col, self.ovd_bal_col])
            if len(train_df) > 0:
                total_amount_train = train_df[self.amount_col].sum()
                if total_amount_train > 0:
                    # 只统计坏样本的ovd_bal
                    train_df_bad = train_df[train_df[self.target_col] == 1]
                    total_ovd_bal_train = train_df_bad[self.ovd_bal_col].sum()
                    overall_loss_rate_train = total_ovd_bal_train / total_amount_train
                else:
                    overall_loss_rate_train = 0.0
            else:
                overall_loss_rate_train = 0.0
            
            # 测试集整体损失率（仅删除amount和ovd_bal的缺失值）
            test_df = self.df.iloc[self.X_test.index][[self.amount_col, self.ovd_bal_col, self.target_col]].dropna(subset=[self.amount_col, self.ovd_bal_col])
            if len(test_df) > 0:
                total_amount_test = test_df[self.amount_col].sum()
                if total_amount_test > 0:
                    # 只统计坏样本的ovd_bal
                    test_df_bad = test_df[test_df[self.target_col] == 1]
                    total_ovd_bal_test = test_df_bad[self.ovd_bal_col].sum()
                    overall_loss_rate_test = total_ovd_bal_test / total_amount_test
                else:
                    overall_loss_rate_test = 0.0
            else:
                overall_loss_rate_test = 0.0
        
        # 整体badrate（基准badrate）
        baseline_badrate_train = self.y_train.mean() if len(self.y_train) > 0 else 0.0
        baseline_badrate_test = self.y_test.mean() if len(self.y_test) > 0 else 0.0
        
        for rule in rules_to_evaluate:
            # 在训练集上应用规则
            mask_train = np.ones(len(self.X_train), dtype=bool)
            for condition in rule['conditions']:
                feature = condition['feature']
                threshold = condition['threshold']
                operator = condition['operator']
                
                # 处理category类型（卡方分箱后的特征）
                feature_values = self.X_train[feature]
                if pd.api.types.is_categorical_dtype(feature_values):
                    # 将category类型转换为float类型
                    feature_values = feature_values.astype(float)
                
                if operator == '<=':
                    mask_train &= (feature_values <= threshold)
                elif operator == '>':
                    mask_train &= (feature_values > threshold)
                elif operator == '==':
                    mask_train &= (feature_values == threshold)
            
            # 计算训练集上的指标
            hit_count_train = mask_train.sum()
            if hit_count_train == 0:
                hit_bad_train = 0
                hit_good_train = 0
                badrate_train = 0
                precision_train = 0
                recall_train = 0
                f1_train = 0
                lift_train = 0
                loss_rate_train = 0.0
                loss_lift_train = 0.0
            else:
                hit_bad_train = self.y_train[mask_train].sum()
                hit_good_train = hit_count_train - hit_bad_train
                badrate_train = hit_bad_train / hit_count_train
                precision_train = hit_bad_train / hit_count_train
                recall_train = hit_bad_train / total_bad_train if total_bad_train > 0 else 0
                f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0
                lift_train = badrate_train / baseline_badrate_train if baseline_badrate_train > 0 else 0
                
                # 计算训练集损失率指标
                loss_rate_train = 0.0
                loss_lift_train = 0.0
                if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
                    train_subset = self.df.iloc[self.X_train.index][mask_train][[self.amount_col, self.ovd_bal_col, self.target_col]].dropna(subset=[self.amount_col, self.ovd_bal_col])
                    if len(train_subset) > 0:
                        # 计算命中样本的总放款金额（所有用户）
                        total_amount_selected = train_subset[self.amount_col].sum()
                        if total_amount_selected > 0:
                            # 计算命中样本的逾期总金额（坏样本）
                            total_ovd_bal_bad_selected = train_subset[train_subset[self.target_col] == 1][self.ovd_bal_col].sum()
                            loss_rate_train = total_ovd_bal_bad_selected / total_amount_selected
                            loss_lift_train = loss_rate_train / overall_loss_rate_train if overall_loss_rate_train > 0 else 0.0 if overall_loss_rate_train > 0 else 0.0
            
            # 在测试集上应用规则
            mask_test = np.ones(len(self.X_test), dtype=bool)
            for condition in rule['conditions']:
                feature = condition['feature']
                threshold = condition['threshold']
                operator = condition['operator']
                
                # 处理category类型（卡方分箱后的特征）
                feature_values = self.X_test[feature]
                if pd.api.types.is_categorical_dtype(feature_values):
                    # 将category类型转换为float类型
                    feature_values = feature_values.astype(float)
                
                if operator == '<=':
                    mask_test &= (feature_values <= threshold)
                elif operator == '>':
                    mask_test &= (feature_values > threshold)
                elif operator == '==':
                    mask_test &= (feature_values == threshold)
            
            # 计算测试集上的指标
            hit_count_test = mask_test.sum()
            if hit_count_test == 0:
                hit_bad_test = 0
                hit_good_test = 0
                badrate_test = 0
                precision_test = 0
                recall_test = 0
                f1_test = 0
                lift_test = 0
                loss_rate_test = 0.0
                loss_lift_test = 0.0
            else:
                hit_bad_test = self.y_test[mask_test].sum()
                hit_good_test = hit_count_test - hit_bad_test
                badrate_test = hit_bad_test / hit_count_test
                precision_test = hit_bad_test / hit_count_test
                recall_test = hit_bad_test / total_bad_test if total_bad_test > 0 else 0
                f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
                lift_test = badrate_test / baseline_badrate_test if baseline_badrate_test > 0 else 0
                
                # 计算测试集损失率指标
                loss_rate_test = 0.0
                loss_lift_test = 0.0
                if self.amount_col and self.ovd_bal_col and self.amount_col in self.df.columns and self.ovd_bal_col in self.df.columns:
                    test_subset = self.df.iloc[self.X_test.index][mask_test][[self.amount_col, self.ovd_bal_col, self.target_col]].dropna(subset=[self.amount_col, self.ovd_bal_col])
                    if len(test_subset) > 0:
                        # 计算命中样本的总放款金额（所有用户）
                        total_amount_selected = test_subset[self.amount_col].sum()
                        if total_amount_selected > 0:
                            # 计算命中样本的逾期总金额（坏样本）
                            total_ovd_bal_bad_selected = test_subset[test_subset[self.target_col] == 1][self.ovd_bal_col].sum()
                            loss_rate_test = total_ovd_bal_bad_selected / total_amount_selected
                            loss_lift_test = loss_rate_test / overall_loss_rate_test if overall_loss_rate_test > 0 else 0.0 if overall_loss_rate_test > 0 else 0.0
            
            # 计算整体badrate（基准badrate）
            baseline_badrate_train = self.y_train.mean()
            baseline_badrate_test = self.y_test.mean()
            
            # 计算命中率（规则命中样本占总样本的比例）
            train_hit_rate = hit_count_train / len(self.X_train) if len(self.X_train) > 0 else 0
            test_hit_rate = hit_count_test / len(self.X_test) if len(self.X_test) > 0 else 0
            
            # 计算压降后badrate（拦截后剩余badrate）
            badrate_after_interception_train = 0.0
            badrate_after_interception_test = 0.0
            
            if len(self.X_train) - hit_count_train > 0:
                badrate_after_interception_train = (total_bad_train - hit_bad_train) / (len(self.X_train) - hit_count_train)
            if len(self.X_test) - hit_count_test > 0:
                badrate_after_interception_test = (total_bad_test - hit_bad_test) / (len(self.X_test) - hit_count_test)
            
            # 计算badrate降低幅度
            badrate_reduction_train = 0.0
            badrate_reduction_test = 0.0
            
            if baseline_badrate_train > 0 and train_hit_rate > 0:
                badrate_reduction_train = ((baseline_badrate_train - badrate_after_interception_train) / baseline_badrate_train) / train_hit_rate
            if baseline_badrate_test > 0 and test_hit_rate > 0:
                badrate_reduction_test = ((baseline_badrate_test - badrate_after_interception_test) / baseline_badrate_test) / test_hit_rate
            
            # 计算效度差异值（训练集和测试集lift的差异）
            badrate_diff = lift_train - lift_test
            
            # 混淆矩阵
            true_positive = hit_bad_test
            false_positive = hit_good_test
            true_negative = ((~mask_test) & (self.y_test == 0)).sum()
            false_negative = ((~mask_test) & (self.y_test == 1)).sum()
            
            # 生成规则描述
            rule_description = ' AND '.join([
                f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f}"
                for cond in rule['conditions']
            ])
            
            result = {
                'rule': rule_description,
                'rule_id': rule['rule_id'],
                'train_hit_count': hit_count_train,
                'train_bad_count': hit_bad_train,
                'train_good_count': hit_good_train,
                'train_badrate': badrate_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train,
                'train_lift': lift_train,
                'train_loss_rate': loss_rate_train,
                'train_loss_lift': loss_lift_train,
                'train_hit_rate': train_hit_rate,
                'train_baseline_badrate': baseline_badrate_train,
                'train_badrate_after_interception': badrate_after_interception_train,
                'train_badrate_reduction': badrate_reduction_train,
                'test_hit_count': hit_count_test,
                'test_bad_count': hit_bad_test,
                'test_good_count': hit_good_test,
                'test_badrate': badrate_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test,
                'test_lift': lift_test,
                'test_loss_rate': loss_rate_test,
                'test_loss_lift': loss_lift_test,
                'test_hit_rate': test_hit_rate,
                'test_baseline_badrate': baseline_badrate_test,
                'test_badrate_after_interception': badrate_after_interception_test,
                'test_badrate_reduction': badrate_reduction_test,
                'badrate_diff': badrate_diff,
                'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative,
                'bad_count': hit_bad_test,
                'good_count': hit_good_test,
                'sample_count': hit_count_test
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        if not df.empty and 'test_lift' in df.columns:
            df = df.sort_values(by='test_lift', ascending=False)
        return df
    
    def get_rules_as_dataframe(self, deduplicate: bool = False, sort_by_lift: bool = False) -> pd.DataFrame:
        """
        获取规则DataFrame
        
        参数:
            deduplicate: 是否去重，默认为False
            sort_by_lift: 是否按lift倒序排序，默认为False（新增）
            
        返回:
            包含规则的DataFrame
        """
        rules_to_export = self.rules if self.algorithm not in ['rf', 'gbdt'] else self.all_rules
        
        if deduplicate and self.algorithm in ['rf', 'gbdt']:
            rules_to_export = self._deduplicate_rules(rules_to_export)
        
        # 按lift倒序排序（新增）
        if sort_by_lift and len(rules_to_export) > 0 and 'importance' in rules_to_export[0].keys():
            rules_to_export = sorted(rules_to_export, key=lambda x: x.get('importance', 0), reverse=True)
        
        # 转换为DataFrame
        df_list = []
        for rule in rules_to_export:
            # 生成规则描述
            rule_description = ' AND '.join([
                f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f}"
                for cond in rule['conditions']
            ])
            
            df_list.append({
                'rule_id': rule['rule_id'],
                'rule': rule_description,
                'predicted_class': rule['predicted_class'],
                'class_name': rule['class_name'],
                'class_probability': rule['class_probability'],
                'sample_count': rule['sample_count'],
                'importance': rule.get('importance', 0),
                'class_distribution': rule['class_distribution'],
                'tree_id': rule.get('tree_id', 0)
            })
        
        df = pd.DataFrame(df_list)
        
        # 如果DataFrame为空，返回空DataFrame
        if df.empty:
            return df
        
        # 按重要性排序
        if 'importance' in df.columns:
            df = df.sort_values(by='importance', ascending=False)
        
        return df
    
    def _deduplicate_rules(self, rules: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        基于规则相似度去重
        
        参数:
            rules: 规则列表
            similarity_threshold: 相似度阈值，默认为0.8
            
        返回:
            去重后的规则列表
        """
        # 将规则条件转换为集合
        rule_sets = []
        for rule in rules:
            conditions = set([
                f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f}"
                for cond in rule['conditions']
            ])
            rule_sets.append(conditions)
        
        # 去重
        unique_rules = []
        used_indices = set()
        
        for i, rule in enumerate(rules):
            if i in used_indices:
                continue
            
            unique_rules.append(rule)
            used_indices.add(i)
            
            # 查找相似规则
            for j in range(i + 1, len(rules)):
                if j in used_indices:
                    continue
                
                # 计算相似度
                similarity = len(rule_sets[i] & rule_sets[j]) / max(len(rule_sets[i]), len(rule_sets[j]))
                
                if similarity >= similarity_threshold:
                    # 保留重要性更高的规则
                    if rule.get('importance', 0) >= rules[j].get('importance', 0):
                        used_indices.add(j)
        
        return unique_rules
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        返回:
            包含特征重要性的DataFrame
        """
        if self.algorithm == 'isf':
            raise ValueError("孤立森林不支持特征重要性计算")
        
        feature_importance = self.model.feature_importances_
        feature_names = self.X.columns.tolist()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values(by='importance', ascending=False)
        
        return df
    
    def plot_feature_importance(self, save_path: str = None, 
                              figsize: Tuple[int, int] = (10, 6)):
        """
        绘制特征重要性图
        
        参数:
            save_path: 保存路径，默认为None
            figsize: 图表大小
        """
        if self.algorithm == 'isf':
            print("孤立森林不支持特征重要性可视化")
            return
        
        feature_importance = self.get_feature_importance()
        
        plt.figure(figsize=figsize)
        sns.barplot(data=feature_importance, x='importance', y='feature', 
                   palette='viridis')
        plt.title(f'Feature Importance ({self.algorithm.upper()})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        return plt
    
    def plot_rule_evaluation(self, save_path: str = None,
                         figsize: Tuple[int, int] = (16, 12)):
        """
        绘制规则评估结果的可视化图表
        
        参数:
            save_path: 保存路径，默认为None
            figsize: 图表大小
        """
        if self.algorithm == 'isf':
            print("孤立森林不支持规则评估可视化")
            return
        
        eval_results = self.evaluate_rules()
        
        if eval_results.empty:
            print("没有规则可以评估")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 规则拦截用户数
        axes[0, 0].bar(range(len(eval_results)), eval_results['sample_count'])
        axes[0, 0].set_title('Sample Count per Rule')
        axes[0, 0].set_xlabel('Rule')
        axes[0, 0].set_ylabel('Sample Count')
        
        # 规则badrate
        axes[0, 1].bar(range(len(eval_results)), eval_results['test_badrate'])
        axes[0, 1].set_title('Badrate per Rule')
        axes[0, 1].set_xlabel('Rule')
        axes[0, 1].set_ylabel('Badrate')
        
        # 规则精确率
        axes[1, 0].bar(range(len(eval_results)), eval_results['test_precision'])
        axes[1, 0].set_title('Precision per Rule')
        axes[1, 0].set_xlabel('Rule')
        axes[1, 0].set_ylabel('Precision')
        
        # 规则召回率
        axes[1, 1].bar(range(len(eval_results)), eval_results['test_recall'])
        axes[1, 1].set_title('Recall per Rule')
        axes[1, 1].set_xlabel('Rule')
        axes[1, 1].set_ylabel('Recall')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"规则评估图已保存到: {save_path}")
        
        return plt
    
    def plot_decision_tree(self, save_path: str = None, 
                        figsize: Tuple[int, int] = (20, 10)):
        """
        绘制决策树结构（仅适用于dt、chi2算法）
        
        参数:
            save_path: 保存路径，默认为None
            figsize: 图表大小
        """
        if self.algorithm in ['rf', 'gbdt', 'isf']:
            print(f"{self.algorithm.upper()}算法不支持决策树结构可视化")
            return
        
        # chi2算法使用RandomForestClassifier，不支持决策树结构可视化
        if self.algorithm == 'chi2':
            print(f"{self.algorithm.upper()}算法不支持决策树结构可视化（使用RandomForestClassifier）")
            return
        
        # 使用sklearn.tree.plot_tree绘制决策树
        from sklearn.tree import plot_tree
        plt.figure(figsize=figsize)
        plot_tree(self.model, feature_names=self.X.columns.tolist(), 
                 class_names=['good', 'bad'], filled=True, rounded=True)
        plt.title(f'Decision Tree ({self.algorithm.upper()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"决策树图已保存到: {save_path}")
        
        return plt
    
    def print_rules(self, top_n: int = 10):
        """
        打印Top规则
        
        参数:
            top_n: 打印的规则数量，默认为10
        """
        if self.algorithm in ['rf', 'xgb']:
            rules_to_print = self.all_rules
        else:
            rules_to_print = self.rules
        
        # 按重要性排序
        sorted_rules = sorted(rules_to_print, 
                         key=lambda x: x.get('importance', 0), 
                         reverse=True)
        
        print(f"\n=== Top {min(top_n, len(sorted_rules))} Rules ({self.algorithm.upper()}) ===\n")
        
        for i, rule in enumerate(sorted_rules[:top_n]):
            # 生成规则描述
            rule_description = ' AND '.join([
                f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f}"
                for cond in rule['conditions']
            ])
            
            print(f"Rule {rule['rule_id']} (Importance: {rule.get('importance', 0):.4f}):")
            print(f"  {rule_description}")
            print(f"  Predicted Class: {rule['class_name']} (Probability: {rule['class_probability']:.4f})")
            print(f"  Sample Count: {rule['sample_count']}")
            print(f"  Class Distribution: {rule['class_distribution']}")
            
            if self.algorithm != 'isf':
                # 在训练集上计算规则命中情况
                mask_train = np.ones(len(self.X_train), dtype=bool)
                for condition in rule['conditions']:
                    feature = condition['feature']
                    threshold = condition['threshold']
                    operator = condition['operator']
                    
                    if operator == '<=':
                        mask_train &= (self.X_train[feature] <= threshold)
                    elif operator == '>':
                        mask_train &= (self.X_train[feature] > threshold)
                    elif operator == '==':
                        mask_train &= (self.X_train[feature] == threshold)
                
                # 计算训练集上的统计信息
                hit_count_train = mask_train.sum()
                if hit_count_train > 0:
                    hit_bad_train = self.y_train[mask_train].sum()
                    hit_good_train = hit_count_train - hit_bad_train
                    badrate_train = hit_bad_train / hit_count_train if hit_count_train > 0 else 0
                    total_badrate_train = self.y_train.mean()
                    lift_train = badrate_train / total_badrate_train if total_badrate_train > 0 else 0
                    
                    print(f"  训练集 - 拦截用户数: {hit_count_train}, 坏客户数: {hit_bad_train}, 好客户数: {hit_good_train}")
                    print(f"  训练集 - Badrate: {badrate_train:.4f}, Lift: {lift_train:.4f}")
                else:
                    print(f"  训练集 - 拦截用户数: 0")
                
                # 在测试集上计算规则命中情况
                mask_test = np.ones(len(self.X_test), dtype=bool)
                for condition in rule['conditions']:
                    feature = condition['feature']
                    threshold = condition['threshold']
                    operator = condition['operator']
                    
                    if operator == '<=':
                        mask_test &= (self.X_test[feature] <= threshold)
                    elif operator == '>':
                        mask_test &= (self.X_test[feature] > threshold)
                    elif operator == '==':
                        mask_test &= (self.X_test[feature] == threshold)
                
                # 计算测试集上的统计信息
                hit_count_test = mask_test.sum()
                if hit_count_test > 0:
                    hit_bad_test = self.y_test[mask_test].sum()
                    hit_good_test = hit_count_test - hit_bad_test
                    badrate_test = hit_bad_test / hit_count_test if hit_count_test > 0 else 0
                    total_badrate_test = self.y_test.mean()
                    lift_test = badrate_test / total_badrate_test if total_badrate_test > 0 else 0
                    
                    print(f"  测试集 - 拦截用户数: {hit_count_test}, 坏客户数: {hit_bad_test}, 好客户数: {hit_good_test}")
                    print(f"  测试集 - Badrate: {badrate_test:.4f}, Lift: {lift_test:.4f}")
                else:
                    print(f"  测试集 - 拦截用户数: 0")
            
            print()
    
    def get_rules_evaluation_summary(self) -> pd.DataFrame:
        """
        获取规则评估摘要，包含训练集和测试集的规则效度情况
        
        返回:
            包含训练集和测试集规则效度情况的DataFrame
        """
        if self.algorithm == 'isf':
            raise ValueError("孤立森林不支持规则评估摘要")
        
        rules_to_evaluate = self.rules if self.algorithm not in ['rf', 'xgb'] else self.all_rules
        results = []
        
        # 训练集总坏样本数
        total_bad_train = self.y_train.sum()
        # 测试集总坏样本数
        total_bad_test = self.y_test.sum()
        
        for rule in rules_to_evaluate:
            # 在训练集上应用规则
            mask_train = np.ones(len(self.X_train), dtype=bool)
            for condition in rule['conditions']:
                feature = condition['feature']
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '<=':
                    mask_train &= (self.X_train[feature] <= threshold)
                elif operator == '>':
                    mask_train &= (self.X_train[feature] > threshold)
                elif operator == '==':
                    mask_train &= (self.X_train[feature] == threshold)
            
            # 计算训练集上的指标
            hit_count_train = mask_train.sum()
            if hit_count_train == 0:
                hit_bad_train = 0
                hit_good_train = 0
                badrate_train = 0
                precision_train = 0
                recall_train = 0
                f1_train = 0
                lift_train = 0
            else:
                hit_bad_train = self.y_train[mask_train].sum()
                hit_good_train = hit_count_train - hit_bad_train
                badrate_train = hit_bad_train / hit_count_train if hit_count_train > 0 else 0
                precision_train = hit_bad_train / hit_count_train if hit_count_train > 0 else 0
                recall_train = hit_bad_train / total_bad_train if total_bad_train > 0 else 0
                f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0
                total_badrate_train = self.y_train.mean()
                lift_train = badrate_train / total_badrate_train if total_badrate_train > 0 else 0
            
            # 在测试集上应用规则
            mask_test = np.ones(len(self.X_test), dtype=bool)
            for condition in rule['conditions']:
                feature = condition['feature']
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '<=':
                    mask_test &= (self.X_test[feature] <= threshold)
                elif operator == '>':
                    mask_test &= (self.X_test[feature] > threshold)
                elif operator == '==':
                    mask_test &= (self.X_test[feature] == threshold)
            
            # 计算测试集上的指标
            hit_count_test = mask_test.sum()
            if hit_count_test == 0:
                hit_bad_test = 0
                hit_good_test = 0
                badrate_test = 0
                precision_test = 0
                recall_test = 0
                f1_test = 0
                lift_test = 0
            else:
                hit_bad_test = self.y_test[mask_test].sum()
                hit_good_test = hit_count_test - hit_bad_test
                badrate_test = hit_bad_test / hit_count_test if hit_count_test > 0 else 0
                precision_test = hit_bad_test / hit_count_test if hit_count_test > 0 else 0
                recall_test = hit_bad_test / total_bad_test if total_bad_test > 0 else 0
                f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
                total_badrate_test = self.y_test.mean()
                lift_test = badrate_test / total_badrate_test if total_badrate_test > 0 else 0
            
            # 生成规则描述
            rule_description = ' AND '.join([
                f"{cond['feature']} {cond['operator']} {cond['threshold']:.4f}"
                for cond in rule['conditions']
            ])
            
            result = {
                'rule_id': rule['rule_id'],
                'rule': rule_description,
                'importance': rule.get('importance', 0),
                'train_hit_count': hit_count_train,
                'train_bad_count': hit_bad_train,
                'train_good_count': hit_good_train,
                'train_badrate': badrate_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train,
                'train_lift': lift_train,
                'test_hit_count': hit_count_test,
                'test_bad_count': hit_bad_test,
                'test_good_count': hit_good_test,
                'test_badrate': badrate_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test,
                'test_lift': lift_test
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by='importance', ascending=False)
        return df
