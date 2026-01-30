from .utils import load_example_data, preprocess_data
from .analysis import (
    VariableAnalyzer,
    analyze_rules,
    analyze_rule_correlation,
    get_user_rule_matrix,
    calculate_strategy_gain
)
from .mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    DecisionTreeRuleExtractor,
    XGBoostRuleMiner,
    TreeRuleExtractor
)
from .visualization import (
    RuleVisualizer,
    plot_rule_comparison,
    plot_rule_distribution,
    plot_lift_precision_scatter,
    plot_decision_tree,
    plot_rule_network,
    plot_heatmap,
    generate_rule_report
)

__version__ = '1.3.0'
__all__ = [
    # Utils
    'load_example_data',
    'preprocess_data',
    
    # Analysis
    'VariableAnalyzer',
    'analyze_rules',
    'analyze_rule_correlation',
    'get_user_rule_matrix',
    'calculate_strategy_gain',
    
    # Mining
    'SingleFeatureRuleMiner',
    'MultiFeatureRuleMiner',
    'DecisionTreeRuleExtractor',
    'XGBoostRuleMiner',  # Deprecated: 请使用 TreeRuleExtractor(algorithm='gbdt')
    'TreeRuleExtractor',
    
    # Visualization
    'RuleVisualizer',
    'plot_rule_comparison',
    'plot_rule_distribution',
    'plot_lift_precision_scatter',
    'plot_decision_tree',
    'plot_rule_network',
    'plot_heatmap',
    'generate_rule_report'
]