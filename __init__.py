from .core import (
    analyze_rules,
    analyze_rule_correlation,
    preprocess_data,
    get_user_rule_matrix
)
from .utils import load_example_data
from .metrics import (
    calculate_rule_correlation
)

__version__ = '0.3.0'
__all__ = [
    # Core functions
    'analyze_rules',
    'analyze_rule_correlation',
    'preprocess_data',
    'get_user_rule_matrix',
    
    # Utils
    'load_example_data',
    
    # Metrics
    'calculate_rule_correlation'
]

